#!/usr/bin/env bash
set -euo pipefail

CODE_DIR="${SM_CODE_DIR:-/opt/ml/processing/input/code}"

sm_pip_bootstrap() {
  python -m pip install --upgrade pip >/dev/null 2>&1 || true
}

sm_install_requirements() {
  if [[ -f "${CODE_DIR}/requirements_sagemaker.txt" ]]; then
    python -m pip install --no-cache-dir -r "${CODE_DIR}/requirements_sagemaker.txt"
  else
    python -m pip install --no-cache-dir -r "${CODE_DIR}/requirements.txt"
  fi
}

sm_pip_bootstrap
sm_install_requirements

exec python "${CODE_DIR}/infer.py" "$@"
