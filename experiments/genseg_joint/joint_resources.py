"""New joint-only resource guard: one selected GPU query and primed CPU delta."""
import csv
import math
import os
import subprocess
import time
from experiments.genseg30.contract import require


class ResourceGuard:
    def __init__(self,index=1,*,min_free_ram_gib=8.,min_free_vram_gib=12.):
        import psutil
        require(type(index) is int and index>=0,'Explicit physical GPU required')
        self.psutil=psutil; self.index=index; self.uuid=None
        self.min_ram=min_free_ram_gib*1024**3; self.min_vram=min_free_vram_gib*1024**3
        require(all(math.isfinite(v) and v>0 for v in (self.min_ram,self.min_vram)),'Positive headroom required')
        self.last_sample=time.perf_counter()
        self.initial_cpu=psutil.cpu_percent(interval=.5)
        self.startup_interval=time.perf_counter()-self.last_sample
        self.last_sample=time.perf_counter()
        self.initial=True

    def __call__(self,reserve_bytes=0):
        require(type(reserve_bytes) is int and reserve_bytes>=0,'Nonnegative RAM reserve required')
        now=time.perf_counter()
        cpu=self.initial_cpu if self.initial else self.psutil.cpu_percent(interval=None)
        interval=self.startup_interval if self.initial else now-self.last_sample
        self.initial=False; self.last_sample=now
        ram=self.psutil.virtual_memory()
        result=subprocess.run(['nvidia-smi','-i',str(self.index),
            '--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader,nounits'],check=True,capture_output=True,text=True)
        rows=list(csv.reader(result.stdout.strip().splitlines()))
        require(len(rows)==1 and len(rows[0])==5,'Selected GPU telemetry unavailable')
        idx,uid,util,used,total=[v.strip() for v in rows[0]]
        util,used,total=float(util),float(used),float(total)
        require(int(idx)==self.index and uid and all(math.isfinite(v) and v>=0 for v in (cpu,util,used,total))
            and cpu<=100 and util<=100 and total>0 and used<=total and interval>0,'Invalid resource telemetry')
        require(cpu<80 and math.isfinite(ram.percent) and ram.percent<80 and util<80 and 100*used/total<80,
            '80 percent resource guard refused')
        require(ram.total>0 and ram.available>=0 and ram.available<=ram.total and
            100*(ram.total-ram.available+reserve_bytes)/ram.total<80 and
            ram.available-reserve_bytes>=self.min_ram,'Insufficient projected snapshot/model RAM headroom')
        require((total-used)*1024**2>=self.min_vram,'Insufficient VRAM headroom')
        require(self.uuid is None or uid==self.uuid,'Selected GPU UUID changed')
        fraction=.79-used/total
        require(math.isfinite(fraction) and fraction>0,
                'Nonpositive/nonfinite CUDA allocator fraction; insufficient allocator headroom')
        self.uuid=uid
        return dict(cpu_pct=cpu,cpu_sample_interval_seconds=interval,ram_used_pct=ram.percent,
            ram_available=ram.available,reserve_bytes=reserve_bytes,gpu_index=self.index,gpu_uuid=uid,
            gpu_util_pct=util,vram_used_mib=used,vram_total_mib=total,allocator_fraction=fraction,
            policy='selected_card_unit_boundary_external_load_check; no_wait_no_retry_no_empty_cache')

    def bind(self):
        require(self.uuid is not None,'Initial resource sample required')
        os.environ['CUDA_VISIBLE_DEVICES']=self.uuid
