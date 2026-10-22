"""New joint-only resource guard: one selected GPU query and primed CPU delta."""
import json
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
        self.allocator_fraction=None

    def __call__(self,reserve_bytes=0,*,phase='startup'):
        require(phase in ('startup','between_units'),'Unknown resource phase')
        require(phase=='startup' or self.allocator_fraction is not None,'Startup resource sample required')
        require(type(reserve_bytes) is int and reserve_bytes>=0,'Nonnegative RAM reserve required')
        snapshot=dict(phase=phase,cpu_pct=None,cpu_sample_interval_seconds=None,
            ram_used_pct=None,ram_available=None,ram_total_bytes=None,reserve_bytes=reserve_bytes,
            ram_min_free_bytes=self.min_ram,gpu_index=self.index,gpu_uuid=None,
            gpu_util_pct=None,vram_total_gib=None,vram_used_gib=None,vram_free_gib=None,
            global_vram_pct=None,violated_rules=[],telemetry_reason=None)
        def number(value):
            try:
                value=float(value)
                return value if math.isfinite(value) else None
            except (ValueError,TypeError): return None
        def check(ok,rule):
            if not ok: snapshot['violated_rules'].append(rule)
        try:
            now=time.perf_counter()
            cpu=self.initial_cpu if self.initial else self.psutil.cpu_percent(interval=None)
            interval=self.startup_interval if self.initial else now-self.last_sample
            self.initial=False; self.last_sample=now
            snapshot.update(cpu_pct=number(cpu),cpu_sample_interval_seconds=number(interval))
            ram=self.psutil.virtual_memory()
            snapshot.update(ram_used_pct=number(ram.percent),ram_available=number(ram.available),
                ram_total_bytes=number(ram.total))
            result=subprocess.run(['nvidia-smi','-i',str(self.index),
                '--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total',
                '--format=csv,noheader,nounits'],check=True,capture_output=True,text=True)
            rows=list(csv.reader(result.stdout.strip().splitlines()))
            require(len(rows)==1 and len(rows[0])==5,'Selected GPU telemetry unavailable')
            idx,uid,util,used,total=[v.strip() for v in rows[0]]
            util,used,total=number(util),number(used),number(total)
            snapshot.update(gpu_uuid=uid or None,gpu_util_pct=util,
                vram_used_gib=None if used is None else used/1024,
                vram_total_gib=None if total is None else total/1024,
                vram_free_gib=None if used is None or total is None else (total-used)/1024,
                global_vram_pct=None if used is None or total is None or total<=0 else 100*used/total)
            selected=int(idx)
            cpu=snapshot['cpu_pct']; interval=snapshot['cpu_sample_interval_seconds']
            require(all(v is not None for v in (cpu,interval,util,used,total,
                snapshot['ram_used_pct'],snapshot['ram_available'],snapshot['ram_total_bytes']))
                and selected==self.index and bool(uid) and 0<=cpu<=100 and 0<=util<=100
                and total>0 and 0<=used<=total and interval>0 and 0<=ram.percent<=100
                and ram.total>0 and 0<=ram.available<=ram.total,'Invalid resource telemetry')
        except Exception as exc:
            snapshot['telemetry_reason']=str(exc)
            snapshot['violated_rules'].append('valid_selected_gpu_telemetry')
        if not snapshot['violated_rules']:
            check(cpu<80,'cpu_pct<80')
            check(ram.percent<80,'ram_used_pct<80')
            if phase=='startup': check(util<80,'gpu_util_pct<80')
            check(100*used/total<80,'global_vram_pct<80')
            check(100*(ram.total-ram.available+reserve_bytes)/ram.total<80,'projected_ram_pct<80')
            check(ram.available-reserve_bytes>=self.min_ram,'RAM_headroom')
            if phase=='startup': check((total-used)*1024**2>=self.min_vram,'startup_free_VRAM>=12GiB')
            check(self.uuid is None or uid==self.uuid,'selected_gpu_uuid_unchanged')
            fraction=self.allocator_fraction if self.allocator_fraction is not None else .79-used/total
            check(math.isfinite(fraction) and fraction>0,'finite_positive_initial_allocator')
        context=(f'phase={phase} selected_index={self.index} selected_uuid={snapshot["gpu_uuid"]} '
            f'total_gib={snapshot["vram_total_gib"]} used_gib={snapshot["vram_used_gib"]} free_gib={snapshot["vram_free_gib"]}; '
            + ('startup_free_VRAM>=12GiB; GPUutil<80' if phase=='startup' else
               'between_units_no_12GiB_requirement; GPUutil_record_only_whole_card_sample_cannot_attribute_self_or_others')
            + '; CPU_RAM_global_VRAM<80%; RAM_headroom; fixed_startup_allocator')
        if snapshot['violated_rules']:
            label=('Invalid resource telemetry' if snapshot['telemetry_reason'] else
                'Selected GPU UUID changed' if 'selected_gpu_uuid_unchanged' in snapshot['violated_rules'] else
                '80 percent resource guard refused or insufficient allocator/headroom')
            exc=ValueError(label+'; '+context+'; snapshot='+json.dumps(snapshot,allow_nan=False))
            exc.resource_snapshot=snapshot
            raise exc
        self.uuid=uid
        self.allocator_fraction=fraction
        return dict(snapshot,vram_used_mib=used,vram_total_mib=total,allocator_fraction=fraction,
            policy=context+'; no_wait_no_retry_no_empty_cache')

    def bind(self):
        require(self.uuid is not None,'Initial resource sample required')
        os.environ['CUDA_VISIBLE_DEVICES']=self.uuid
