"""Windows resource guard and evidence recorder for task-local CPU tests."""
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = Path(__file__).resolve().parent / 'validation'
EVIDENCE.mkdir(exist_ok=True)

class Memory(ctypes.Structure):
    _fields_ = [('length', wintypes.DWORD), ('load', wintypes.DWORD)] + [(n, ctypes.c_ulonglong) for n in ('total_phys', 'avail_phys', 'total_page', 'avail_page', 'total_virtual', 'avail_virtual', 'avail_ext')]

def cpu_times():
    values = [wintypes.FILETIME() for _ in range(3)]
    if not ctypes.windll.kernel32.GetSystemTimes(*(ctypes.byref(v) for v in values)):
        raise OSError('GetSystemTimes failed')
    return [v.dwLowDateTime + (v.dwHighDateTime << 32) for v in values]

a = cpu_times()
time.sleep(1)
b = cpu_times()
delta = [y-x for x,y in zip(a,b)]
cpu = 100 * (1-delta[0]/(delta[1]+delta[2]))
mem = Memory()
mem.length = ctypes.sizeof(mem)
if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)):
    raise OSError('GlobalMemoryStatusEx failed')
gpu_result = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], capture_output=True, text=True)
if gpu_result.returncode:
    raise RuntimeError('GPU preflight unavailable: ' + gpu_result.stderr)
gpus = []
for row in gpu_result.stdout.strip().splitlines():
    index, util, used, total = [float(v.strip()) for v in row.split(',')]
    gpus.append({'index': int(index), 'gpu_percent': util, 'vram_percent': 100*used/total, 'used_MiB': used, 'total_MiB': total})
label, *args = sys.argv[1:]
preflight = {'cpu_percent': cpu, 'ram_percent': 100*(1-mem.avail_phys/mem.total_phys), 'ram_total_GiB': mem.total_phys/2**30, 'gpus': gpus}
high = cpu >= 80 or preflight['ram_percent'] >= 80 or any(g['gpu_percent'] >= 80 or g['vram_percent'] >= 80 for g in gpus)
preflight['decision'] = 'LIGHTWEIGHT' if high else 'NORMAL'
cpu_control = any(args[i:i+2] == ['-k', 'cpu'] for i in range(len(args)))
cuda_smoke = any('test_server_cuda_smoke.py' in arg for arg in args) and not cpu_control
preflight['load'] = ('GPU0 tiny 17x17/four-stage/two-channel encoder; single-device FP32; no real data'
                     if cuda_smoke else 'CPU only, one thread, serial synthetic; no real data or CUDA allocations')
print('RESOURCE PREFLIGHT ' + json.dumps(preflight), flush=True)
# At high pressure CUDA probes stop; only exact lightweight CPU nodes are permitted.
if high and (cuda_smoke or not any('::' in arg for arg in args)):
    (EVIDENCE / (label + '.json')).write_text(json.dumps({'preflight': preflight, 'requested': args, 'status': 'NOT_RUN_RESOURCE_GUARD'}, indent=2), encoding='utf-8')
    sys.exit(80)
env = os.environ.copy()
env.update(PYTHONIOENCODING='utf-8', PYTHONUTF8='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', nnUNet_compile='false', nnUNet_extTrainer=str(ROOT/'nnunet_ext_trainers'))
# Pytest artifacts stay in an isolated E: temp directory outside tracked source.
temp = ROOT.parent / (ROOT.name + '-test-temp') / label
temp.parent.mkdir(exist_ok=True)
command = [sys.executable, '-m', 'pytest', *args, '-q', '--basetemp', str(temp)]
print('COMMAND ' + repr(command), flush=True)
start = time.time()
result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, encoding='utf-8', errors='replace')
print(result.stdout, flush=True)
print(result.stderr, flush=True)
print('EXIT_CODE ' + str(result.returncode), flush=True)
snapshot = {str(p.relative_to(ROOT)).replace('\\','/'): hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'nnunet_ext_trainers').rglob('*.py') if '__pycache__' not in str(p)}
snapshot[str(Path(__file__).relative_to(ROOT)).replace('\\','/')] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
probe = Path(__file__).with_name('test_server_cuda_smoke.py')
if probe.exists():
    snapshot[str(probe.relative_to(ROOT)).replace('\\','/')] = hashlib.sha256(probe.read_bytes()).hexdigest()
(EVIDENCE/(label+'.json')).write_text(json.dumps({'preflight': preflight, 'command': command, 'exit_code': result.returncode, 'duration_seconds': time.time()-start, 'stdout': result.stdout, 'stderr': result.stderr, 'source_sha256': snapshot}, ensure_ascii=False, indent=2), encoding='utf-8')
sys.exit(result.returncode)
