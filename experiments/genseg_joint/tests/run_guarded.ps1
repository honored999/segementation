param([Parameter(ValueFromRemainingArguments=$true)][string[]]$TestArgs)
$ErrorActionPreference = 'Stop'
$cpu = (Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor -Filter "Name='_Total'").PercentProcessorTime
$os = Get-CimInstance Win32_OperatingSystem
$ram = 100 * (1 - $os.FreePhysicalMemory / $os.TotalVisibleMemorySize)
$gpu = & nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
if ($LASTEXITCODE -ne 0) { throw 'GPU telemetry unavailable' }
Write-Output "RESOURCE PREFLIGHT CPU=$cpu RAM=$ram GPUs=$gpu; single thread CPU tiny tests"
if ($cpu -ge 80 -or $ram -ge 80 -or $os.FreePhysicalMemory -lt 3GB/1KB) { throw 'RESOURCE GUARD: CPU/RAM/headroom' }
foreach ($row in $gpu) {
  $cols = $row.Split(',') | ForEach-Object { [double]$_.Trim() }
  if ($cols[1] -ge 80 -or 100*$cols[2]/$cols[3] -ge 80) { throw 'RESOURCE GUARD: GPU busy' }
}
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
& D:/Anaconda/envs/newconda/python.exe -m pytest -q -p no:cacheprovider @TestArgs
exit $LASTEXITCODE
