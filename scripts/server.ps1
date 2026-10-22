#Requires -Version 7.0
[CmdletBinding()]
param(
    [ValidateSet('start','stop','status')][string]$Action = 'start',
    [int]$Port = 8800,
    [Alias('Host')][string]$BindHost = '127.0.0.1',
    [switch]$NoBrowser,
    [string]$Python = ''
)
$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$Candidates = [System.Collections.Generic.List[string]]::new()
if ($Python) { $Candidates.Add($Python) }
if ($env:VIDEO_POLISH_PYTHON) { $Candidates.Add($env:VIDEO_POLISH_PYTHON) }
$SavedPython = Join-Path $ProjectRoot '.runtime\python.json'
if (Test-Path -LiteralPath $SavedPython) {
    $Saved = Get-Content -LiteralPath $SavedPython -Raw -Encoding utf8 | ConvertFrom-Json
    if ($Saved.platform -eq 'win32') { $Candidates.Add($Saved.python) }
}
$Candidates.Add((Join-Path $ProjectRoot '.venv\Scripts\python.exe'))
$Candidates.Add((Join-Path (Split-Path $ProjectRoot -Parent) 'pic_polish\.venv\Scripts\python.exe'))
foreach ($EnvRoot in @((Join-Path $env:USERPROFILE '.conda\envs'), 'E:\ProgramData\anaconda3new\envs')) {
    if (Test-Path -LiteralPath $EnvRoot) {
        foreach ($Item in Get-ChildItem -LiteralPath $EnvRoot -Directory) { $Candidates.Add((Join-Path $Item.FullName 'python.exe')) }
    }
}
$Selected = $null
foreach ($Candidate in ($Candidates | Select-Object -Unique)) {
    if (-not (Test-Path -LiteralPath $Candidate -PathType Leaf)) { continue }
    # Stop/status need only the standard library, even if dependencies are damaged.
    if ($Action -ne 'start') { $Selected = $Candidate; break }
    $null = & $Candidate (Join-Path $PSScriptRoot 'check_env.py') 2>$null
    if ($LASTEXITCODE -eq 0) { $Selected = $Candidate; break }
}
if (-not $Selected) {
    throw 'No suitable existing Python found. Set VIDEO_POLISH_PYTHON to Python 3.10+ with torch, cv2, numpy, FastAPI, uvicorn and python-multipart. Existing environments will not be modified.'
}
$ServiceArgs = @((Join-Path $PSScriptRoot 'runner.py'), (Join-Path $PSScriptRoot 'service.py'), $Action, '--port', "$Port", '--host', $BindHost)
if ($NoBrowser) { $ServiceArgs += '--no-browser' }
Push-Location -LiteralPath $ProjectRoot
try {
    & $Selected @ServiceArgs
    $ResultCode = $LASTEXITCODE
} finally { Pop-Location }
exit $ResultCode
