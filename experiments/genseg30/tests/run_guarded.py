"""Guarded tiny serial CPU tests; no dependency installation or patient access."""
import json
import os
import subprocess
import sys
import time

if os.name == 'nt':
    script = '$c = @(); 1..3 | ForEach-Object { $c += (Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor | Where-Object Name -eq "_Total").PercentProcessorTime; Start-Sleep -Milliseconds 400 }; $m=Get-CimInstance Win32_OperatingSystem; @{cpu=($c | Measure-Object -Average).Average; ram=100*(1-$m.FreePhysicalMemory/$m.TotalVisibleMemorySize); available=[double]$m.FreePhysicalMemory*1024} | ConvertTo-Json -Compress'
    stats = json.loads(subprocess.check_output(['powershell', '-NoProfile', '-Command', script], text=True))
else:
    def cpu_ticks():
        with open('/proc/stat') as stream:
            vals = [int(x) for x in stream.readline().split()[1:]]
        return sum(vals), sum(vals[3:5])
    a, idle_a = cpu_ticks()
    time.sleep(1)
    b, idle_b = cpu_ticks()
    with open('/proc/meminfo') as stream:
        mem = {line.split(':')[0]: int(line.split()[1]) * 1024 for line in stream}
    stats = {'cpu': 100 * (1 - (idle_b - idle_a) / (b - a)),
             'ram': 100 * (1 - mem['MemAvailable'] / mem['MemTotal']), 'available': mem['MemAvailable']}
rows = []
try:
    query = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True,
                           timeout=10, check=True)
    rows = [[float(x.strip()) for x in row.split(',')] for row in query.stdout.strip().splitlines()]
except (FileNotFoundError, subprocess.SubprocessError, ValueError) as exc:
    print('GPU/VRAM unavailable:', type(exc).__name__)
print(f"CPU={stats['cpu']:.1f}% RAM={stats['ram']:.1f}% available={stats['available'] / 2**30:.2f}GiB")
for i, (gpu, used, total) in enumerate(rows):
    print(f'GPU{i}={gpu:.1f}% VRAM={100 * used / total:.1f}% ({used}/{total}MiB)')
if stats['cpu'] >= 80 or stats['ram'] >= 80 or stats['available'] < 512 * 2**20 or any(
        gpu >= 80 or used / total >= .8 for gpu, used, total in rows):
    print('DEFERRED_RESOURCE_GUARD: retry later or select a pure metadata check.')
    sys.exit(3)
print('NORMAL: tiny serial CPU fixtures, one thread; no production model/real data.', flush=True)
os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                  CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
sys.exit(subprocess.call([sys.executable, '-m', 'pytest', '-p', 'no:cacheprovider', *sys.argv[1:]]))
