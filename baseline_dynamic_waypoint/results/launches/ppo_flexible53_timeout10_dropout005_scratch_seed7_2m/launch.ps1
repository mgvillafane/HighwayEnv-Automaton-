$ErrorActionPreference = 'Stop'
$experimentRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../../..')).Path
$experimentName = 'ppo_flexible53_timeout10_dropout005_scratch_seed7_2m'
$experimentPython = Join-Path $experimentRoot 'baseline_waypoint/.venv/Scripts/python.exe'
$experimentResults = Join-Path $experimentRoot "baseline_dynamic_waypoint/results/$experimentName"
if (Test-Path -LiteralPath $experimentResults) {
    throw "Results already exist: $experimentResults. Choose a new experiment name."
}
$experimentArguments = @(
    '-B', '-u', '-m', 'baseline_dynamic_waypoint.train_single_car',
    '--run-name', $experimentName,
    '--timesteps', '2000000',
    '--seed', '7',
    '--num-envs', '4',
    '--rollout-steps', '250',
    '--observation-schema', 'legacy',
    '--reward-profile', 'timeout10',
    '--dropout-rate', '0.05',
    '--learning-rate', '0.0001',
    '--gamma', '0.99',
    '--time-penalty', '0.05',
    '--eval-episodes', '100',
    '--eval-seed', '10000',
    '--eval-every', '250000',
    '--quiet'
)
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
    source_config = 'baseline_dynamic_waypoint/results/ppo_singlecar_flexible30s_seed7_200k/run.json'
    initialization = 'fresh actor, critic and optimizer; no checkpoint loaded'
    configuration_changes = @{
        timeout_penalty = 10.0
        dropout_rate = 0.05
        training_timesteps = 2000000
    }
    results_directory = $experimentResults
    stdout = (Join-Path $PSScriptRoot 'training_stdout.log')
    stderr = (Join-Path $PSScriptRoot 'training_stderr.log')
}
$experimentLaunch | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'launch.json') -Encoding UTF8
$experimentLaunch | ConvertTo-Json -Depth 5
