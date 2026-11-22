$ErrorActionPreference = 'Stop'
$dashboardRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../../..')).Path
$dashboardUrl = 'http://127.0.0.1:6020/live'
$dashboardPython = Join-Path $dashboardRoot '.tensorboard-venv/Scripts/python.exe'
$dashboardReady = $false
try {
    $dashboardResponse = Invoke-WebRequest -Uri $dashboardUrl -UseBasicParsing -TimeoutSec 2
    $dashboardReady = $dashboardResponse.StatusCode -eq 200
} catch { }
if (-not $dashboardReady) {
    $dashboardArguments = @('-B', '-u', 'monitor_training.py', '--port', '6020',
        '--ppo-logdir', 'baseline_dynamic_waypoint/results/ppo_maneuver69_lateral_side10_seed7_1m',
        '--latest', '1')
    $dashboardProcess = Start-Process -FilePath $dashboardPython -ArgumentList $dashboardArguments `
        -WorkingDirectory $dashboardRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $PSScriptRoot 'tensorboard_stdout.log') `
        -RedirectStandardError (Join-Path $PSScriptRoot 'tensorboard_stderr.log')
    Write-Output "TensorBoard process: $($dashboardProcess.Id)"
    for ($dashboardAttempt = 0; $dashboardAttempt -lt 30; $dashboardAttempt++) {
        Start-Sleep -Seconds 1
        try {
            $dashboardResponse = Invoke-WebRequest -Uri $dashboardUrl -UseBasicParsing -TimeoutSec 1
            if ($dashboardResponse.StatusCode -eq 200) { $dashboardReady = $true; break }
        } catch { }
        if ($dashboardProcess.HasExited) { throw 'TensorBoard exited; inspect tensorboard_stderr.log.' }
    }
}
if (-not $dashboardReady) { throw 'TensorBoard has not responded yet; inspect its logs.' }
Write-Output "TensorBoard ready: $dashboardUrl"
Start-Process $dashboardUrl
