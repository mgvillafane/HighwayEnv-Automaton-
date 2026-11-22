# Restore archived checkpoints to their recorded paths, without overwriting files.
$ErrorActionPreference = 'Stop'
$workspaceRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$manifest = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'manifest.json') -Raw | ConvertFrom-Json
$sourceRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot 'models')) + [IO.Path]::DirectorySeparatorChar
$destinationRoot = [IO.Path]::GetFullPath((Join-Path $workspaceRoot 'baseline_dynamic_waypoint/models')) + [IO.Path]::DirectorySeparatorChar
foreach ($entry in $manifest.entries) {
    $source = [IO.Path]::GetFullPath((Join-Path $workspaceRoot $entry.destination))
    $destination = [IO.Path]::GetFullPath((Join-Path $workspaceRoot $entry.source))
    if (-not $source.StartsWith($sourceRoot, [StringComparison]::OrdinalIgnoreCase)) { throw "Source outside archive: $source" }
    if (-not $destination.StartsWith($destinationRoot, [StringComparison]::OrdinalIgnoreCase)) { throw "Destination outside model directory: $destination" }
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { throw "Missing archive file: $source" }
    if (Test-Path -LiteralPath $destination) { throw "Refusing to overwrite: $destination" }
    if ((Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha256) { throw "Archive hash mismatch: $source" }
}
$archiveResultsRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot 'results')) + [IO.Path]::DirectorySeparatorChar
$resultsRoot = [IO.Path]::GetFullPath((Join-Path $workspaceRoot 'baseline_dynamic_waypoint/results')) + [IO.Path]::DirectorySeparatorChar
foreach ($entry in $manifest.directories) {
    $source = [IO.Path]::GetFullPath((Join-Path $workspaceRoot $entry.destination))
    $destination = [IO.Path]::GetFullPath((Join-Path $workspaceRoot $entry.source))
    if (-not $source.StartsWith($archiveResultsRoot, [StringComparison]::OrdinalIgnoreCase)) { throw "Directory outside archive: $source" }
    if (-not $destination.StartsWith($resultsRoot, [StringComparison]::OrdinalIgnoreCase)) { throw "Directory outside results: $destination" }
    if (-not (Test-Path -LiteralPath $source -PathType Container)) { throw "Missing archive directory: $source" }
    if (Test-Path -LiteralPath $destination) { throw "Refusing to overwrite directory: $destination" }
}
foreach ($entry in $manifest.entries) {
    Move-Item -LiteralPath (Join-Path $workspaceRoot $entry.destination) -Destination (Join-Path $workspaceRoot $entry.source)
}
foreach ($entry in $manifest.directories) {
    Move-Item -LiteralPath (Join-Path $workspaceRoot $entry.destination) -Destination (Join-Path $workspaceRoot $entry.source)
}
Write-Output "Restored $($manifest.archived_files) files."
