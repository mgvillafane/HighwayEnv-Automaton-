$ErrorActionPreference = 'Stop'
$experimentRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../../..')).Path
$experimentName = 'ppo_maneuver69_lateral_side10_seed7_1m'
$experimentPython = Join-Path $experimentRoot 'baseline_waypoint/.venv/Scripts/python.exe'
$experimentResults = Join-Path $experimentRoot "baseline_dynamic_waypoint/results/$experimentName"
if (Test-Path -LiteralPath $experimentResults) {
    throw "Results already exist: $experimentResults. Choose a new experiment name."
}
$experimentArguments = @(
    '-B', '-u', '-m', 'baseline_dynamic_waypoint.train_single_car',
    '--run-name', $experimentName,
    '--timesteps', '1000000',
    '--seed', '7',
    '--num-envs', '4',
    '--rollout-steps', '250',
    '--observation-schema', 'legacy',
    '--reward-profile', 'maneuver_lateral_side10',
    '--dropout-rate', '0.05',
    '--learning-rate', '0.0001',
    '--gamma', '0.99',
    '--time-penalty', '0.05',
    '--eval-episodes', '100',
    '--eval-seed', '10000',
    '--eval-every', '250000',
    '--quiet'
)
$experimentSnapshot = Join-Path $PSScriptRoot 'source'
New-Item -ItemType Directory -Path $experimentSnapshot -Force | Out-Null
foreach ($experimentFile in @('config.py', 'maneuver_guidance.py', 'single_car.py', 'environment.py',
                              'train.py', 'train_single_car.py', 'dropout_policy.py', 'evaluate.py')) {
    Copy-Item -LiteralPath (Join-Path $experimentRoot "baseline_dynamic_waypoint/$experimentFile") `
        -Destination (Join-Path $experimentSnapshot $experimentFile)
}
$experimentProcess = Start-Process -FilePath $experimentPython -ArgumentList $experimentArguments `
    -WorkingDirectory $experimentRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $PSScriptRoot 'training_stdout.log') `
    -RedirectStandardError (Join-Path $PSScriptRoot 'training_stderr.log')
$experimentLaunch = [ordered]@{
    run_name = $experimentName
    process_id = $experimentProcess.Id
    launched_utc = [DateTime]::UtcNow.ToString('o')
    python = $experimentPython
    arguments = $experimentArguments
    comparison_run = 'ppo_maneuver69_notrack_side10_seed7_1p5m'
    initialization = 'fresh actor, critic and optimizer; no checkpoint loaded'
    changes = 'route cost = 2 * abs(goal-lane lateral error); longitudinal and preferred-speed route weights zero; no FRONT lateral blend'
    retained = 'original distance progress 0.1, route progress 0.5, FRONT milestone +10 once, tracking zero, all 69 observations'
    timeout_penalty = 10.0
    dropout_rate = 0.05
    training_timesteps = 1000000
    source_snapshot = $experimentSnapshot
    results_directory = $experimentResults
}
$experimentLaunch | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'launch.json') -Encoding UTF8
$experimentLaunch | ConvertTo-Json -Depth 5
