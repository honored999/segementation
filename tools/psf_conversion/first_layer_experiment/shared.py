"""Isolated experiment process and resource helpers."""
import hashlib, json, os, shutil, subprocess, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = None
PY=Path('C:/Users/user4/.conda/envs/nnunet5090/python.exe')
GPU='GPU-6c48a375-dd11-ff27-ed3d-0508532a9c6d'
KEY='encoder.stages.0.0.convs.0.conv.weight'
state={'status':'STARTING','stage':'INITIALIZE','pid':os.getpid(),'started_utc':datetime.now(timezone.utc).isoformat()}
def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    p=Path(p); temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    os.replace(temp,p)
def save(**v):
    state.update(v);state['updated_utc']=datetime.now(timezone.utc).isoformat()
    write(ROOT/'status.json',state)
def need(v,msg):
    if not v: raise RuntimeError(msg)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def pin(p):return {'path':str(p),'sha256':sha(p)}
def env_for(runtime):
    e=dict(os.environ)
    e.update(CUDA_VISIBLE_DEVICES=GPU,nnUNet_compile='false',PYTHONUTF8='1',
      PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg',
      OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
      ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS='1',nnUNet_n_proc_DA='1',nnUNet_def_n_proc='1',
      nnUNet_extTrainer=str(runtime/'trainers'))
    return e
def resources():
    import psutil
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
    row=next(x for x in rows.splitlines() if x.split(',')[0].strip()==GPU)
    _,util,used,total=[x.strip() for x in row.split(',')]
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True)
    competing=[x for x in apps.splitlines() if x.split(',')[0].strip()==GPU and any(t in x.lower() for t in ('python','cuda','triton'))]
    ram=psutil.virtual_memory()
    return dict(gpu=1,uuid=GPU,util=float(util),used_mib=float(used),total_mib=float(total),
      competing=competing,cpu=psutil.cpu_percent(interval=.5),ram_percent=ram.percent,
      ram_available_bytes=ram.available,disk_free_bytes=shutil.disk_usage(ROOT).free)
def gate(stage,free_mib):
    stable=0
    while stable<2:
        if free_mib:
            r=resources()
        else:
            import psutil
            ram=psutil.virtual_memory()
            r=dict(cpu=psutil.cpu_percent(interval=.5),ram_percent=ram.percent,
              ram_available_bytes=ram.available,disk_free_bytes=shutil.disk_usage(ROOT).free)
        ready=((not free_mib or (not r['competing'] and r['util']<10 and r['used_mib']<.2*r['total_mib']
          and r['total_mib']-r['used_mib']>free_mib)) and r['cpu']<80
          and r['ram_percent']<80 and r['ram_available_bytes']>8*2**30 and r['disk_free_bytes']>10*2**30)
        stable=stable+1 if ready else 0
        save(status='WAITING_RESOURCE',stage=stage,resources=r)
        if stable<2:time.sleep(5 if ready else 30)
    save(status='RUNNING',stage=stage)
def run(stage,argv,env,free_mib):
    gate(stage,free_mib)
    log=ROOT/(stage.lower()+'.log')
    began=time.monotonic()
    with log.open('xb') as stream:
        child=subprocess.Popen([str(PY),'-u',*map(str,argv)],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        save(child_pid=child.pid,log=str(log))
        code=child.wait()
    state.setdefault('stages',{})[stage]={'exit_code':code,'seconds':time.monotonic()-began,'log':str(log)}
    save(child_pid=None)
    need(code==0,f'{stage} exited {code}; see {log}')
