"""Task-owned, no-notification Windows scheduler worker: mode-selected training then evaluation."""
import argparse,datetime,hashlib,json,os,subprocess,sys,traceback
from pathlib import Path
from run_layout import job_layout


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',required=True);a=p.parse_args()
    root=Path(a.run_root);code=Path(__file__).resolve().parent
    spec=json.loads((root/'job_spec.json').read_text())
    layout=job_layout(root,spec)
    config=json.loads(layout['config'].read_text(encoding='utf-8-sig'))
    assert config.get('mode','baseline')==layout['mode'],'JOB_CONFIG_MODE_MISMATCH'
    state_path=root/'job_status.json'
    assert not state_path.exists(),'DUPLICATE_SUPERVISOR_FORBIDDEN'
    state={'status':'RUNNING','phase':'startup','pid':os.getpid(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'jobs':{}}
    def save():
        temporary=state_path.with_suffix('.pending.json')
        temporary.write_text(json.dumps(state,indent=2),encoding='utf-8');temporary.replace(state_path)
    save()
    try:
        for name,pin in spec['source_files'].items():
            assert (code/name).is_file() and hashlib.sha256((code/name).read_bytes()).hexdigest()==pin,'CODE_SNAPSHOT_CHANGED'
        executable=spec['python']
        env=os.environ.copy();env.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        stages=[('training',[executable,'-u','-B',str(code/'run.py'),'--config',str(layout['config']),'--stage','train','--gpu-uuid',spec['gpu_uuid'],'--output',str(layout['training'])]),
                ('evaluation',[executable,'-u','-B',str(code/'evaluate_run.py'),'--run-root',str(root)])]
        for name,args in stages:
            import psutil
            assert psutil.cpu_percent(interval=1)<80 and psutil.virtual_memory().percent<80,'HOST_RESOURCE_GUARD'
            state['phase']=name
            item={'status':'RUNNING','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};state['jobs'][name]=item;save()
            with (root/(name+'.stdout.log')).open('x',encoding='utf-8') as log:
                process=subprocess.Popen(args,cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
                item['pid']=process.pid;save();exit_code=process.wait()
            item.update(exit_code=exit_code,status='COMPLETED' if exit_code==0 else 'FAILED',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
            if exit_code:raise RuntimeError(name.upper()+'_FAILED')
        state.update(status='COMPLETED',phase='finished',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save()
    except BaseException as error:
        state.update(status='FAILED',error_type=type(error).__name__,error=str(error),finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save();(root/'controller_error.log').write_text(traceback.format_exc(),encoding='utf-8');raise


if __name__=='__main__':main()