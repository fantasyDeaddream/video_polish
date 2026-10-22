@echo off
setlocal EnableExtensions
set "PS_EXE=pwsh"
where.exe pwsh >nul 2>nul
if errorlevel 1 (
  if exist "%ProgramFiles%\PowerShell\7\pwsh.exe" (
    set "PS_EXE=%ProgramFiles%\PowerShell\7\pwsh.exe"
  ) else if exist "%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\powershell\pwsh.exe" (
    set "PS_EXE=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\powershell\pwsh.exe"
  ) else (
    echo [ERROR] PowerShell 7+ ^(pwsh^) is required.
    exit /b 1
  )
)
"%PS_EXE%" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\server.ps1" -Action start %*
exit /b %ERRORLEVEL%
