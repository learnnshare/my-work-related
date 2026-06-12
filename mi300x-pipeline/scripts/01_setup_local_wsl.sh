#!/usr/bin/env bash
# 01_setup_local_wsl.sh — set up the LOCAL Ubuntu 22.04 / WSL box.
#
# The local box runs the normalize/predict/publish side + the dashboard. It does
# NOT need ROCm or a GPU (the MI300X lives in the cloud). This installs Python,
# a venv, the pipeline deps, runs the demo, and prints how to view the dashboard.
#
# Usage:  ./01_setup_local_wsl.sh
source "$(dirname "$0")/lib.sh"

banner "Local WSL / Ubuntu 22.04 setup"
is_wsl && ok "WSL detected" || warn "not WSL — fine, this works on plain Ubuntu too"

# 1. system packages
step "Installing base packages (python3, venv, pip, git, rsync)"
SUDO apt-get update -y
SUDO apt-get install -y python3 python3-venv python3-pip git rsync openssh-client
ok "base packages installed"

# 2. python venv
VENV="$REPO_DIR/.venv"
step "Creating virtualenv at $VENV"
[ -d "$VENV" ] || python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip >/dev/null
ok "venv ready ($(python --version))"

# 3. pipeline deps
step "Installing pipeline requirements"
pip install -r "$REPO_DIR/requirements.txt"
ok "requirements installed"

# 4. run the demo end-to-end (fixtures — no hardware needed)
step "Running the demo pipeline (fixtures)"
( cd "$REPO_DIR" && python orchestrator.py --config pipeline.yaml )
ok "demo published data to ../mi300x-dashboard/data/"

# 5. how to view the dashboard
DASH="$(cd "$REPO_DIR/../mi300x-dashboard" 2>/dev/null && pwd || true)"
banner "Next steps"
echo "Activate the venv in new shells:   ${BLD}source $VENV/bin/activate${RST}"
if [ -n "$DASH" ]; then
  echo "Open the dashboard (WSL → Windows browser):"
  echo "   ${BLD}cd $DASH && explorer.exe index.html${RST}"
  echo "or serve it (then browse http://localhost:8000):"
  echo "   ${BLD}cd $DASH && python3 -m http.server 8000${RST}"
fi
echo "Set up SSH to the cloud MI300X:     ${BLD}./scripts/00_setup_ssh.sh${RST}"
