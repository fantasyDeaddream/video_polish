#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
action="${1:-start}"
shift || true
# Under WSL, reuse Windows CUDA and Python instead of installing a second stack.
if grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null && [[ "${VIDEO_POLISH_NATIVE_WSL:-0}" != 1 ]]; then
  windows_script="$(wslpath -w "$project_root/scripts/server.ps1")"
  pwsh_path="$(command -v pwsh.exe || true)"
  if [[ -z "$pwsh_path" ]]; then
    for candidate in '/mnt/c/Program Files/PowerShell/7/pwsh.exe' /mnt/c/Users/*/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/powershell/pwsh.exe; do
      if [[ -f "$candidate" ]]; then pwsh_path="$candidate"; break; fi
    done
  fi
  if [[ -z "$pwsh_path" ]]; then echo 'PowerShell 7 not found; put pwsh.exe on WSL PATH.' >&2; exit 1; fi
  translated=()
  while (($#)); do
    case "$1" in
      --port) translated+=(-Port "$2"); shift 2 ;;
      --host) translated+=(-BindHost "$2"); shift 2 ;;
      --no-browser) translated+=(-NoBrowser); shift ;;
      --python) translated+=(-Python "$(wslpath -w "$2")"); shift 2 ;;
      *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
  done
  exec "$pwsh_path" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "$windows_script" -Action "$action" "${translated[@]}"
fi
python_path="${VIDEO_POLISH_PYTHON:-}"
if [[ -z "$python_path" ]]; then
  for candidate in "$project_root/.venv/bin/python" "$project_root/../pic_polish/.venv-linux/bin/python" "${CONDA_PREFIX:-/nonexistent}/bin/python" "$(command -v python3 || true)"; do
    [[ -x "$candidate" ]] || continue
    if [[ "$action" != start ]] || "$candidate" "$project_root/scripts/check_env.py" >/dev/null 2>&1; then python_path="$candidate"; break; fi
  done
fi
if [[ ! -x "$python_path" ]]; then echo 'Set VIDEO_POLISH_PYTHON to an existing compatible Python.' >&2; exit 1; fi
cd -- "$project_root"
export PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1
exec "$python_path" "$project_root/scripts/runner.py" "$project_root/scripts/service.py" "$action" "$@"
