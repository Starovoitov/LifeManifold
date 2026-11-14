# Run any LifeManifold experiment inside the Linux container.
# Artifacts stay in this repo. API keys are read from .env when that file exists.
#
#   .\scripts\docker-experiment.ps1 bash scripts/run_cvt_baseline.sh
#   .\scripts\docker-experiment.ps1 uv run python -m worldspace --illuminator mapelites --scheduler worldspace/specs/map_elites_scheduler_mini.yaml --output-dir artifacts/map_elites_smoke --seed 42 --steps 200 --grid 8
#   .\scripts\docker-experiment.ps1 -Gpu uv run python scripts/train_surrogate.py
#   .\scripts\docker-experiment.ps1
# The last form opens a shell in the container.

param(
    [switch]$Gpu,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Command
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Error "Docker is not on PATH. Start Docker Desktop, then run this script again."
}

$composeFiles = @("-f", "docker-compose.yml")
if ($Gpu) {
    $composeFiles += @("-f", "docker-compose.gpu.yml")
}

if (-not $Command -or $Command.Count -eq 0) {
    & docker compose @composeFiles run --rm -it experiment bash
    exit $LASTEXITCODE
}

if ($Command[0] -like "*.sh") {
    $Command = @("bash") + $Command
}

& docker compose @composeFiles run --rm experiment @Command
exit $LASTEXITCODE
