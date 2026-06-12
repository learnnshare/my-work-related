#!/usr/bin/env bash
# 00_setup_ssh.sh — generate an SSH key (WSL side) and add a Host entry for the
# cloud MI300X box, so you can `ssh mi300x` and copy files with scp/rsync.
#
# Usage:
#   ./00_setup_ssh.sh                       # interactive
#   ./00_setup_ssh.sh --host 1.2.3.4 --user ubuntu --alias mi300x
#
# Run this on your LOCAL Ubuntu/WSL machine.
source "$(dirname "$0")/lib.sh"

ALIAS="mi300x"; HOST=""; USER_NAME=""; PORT="22"; KEY="$HOME/.ssh/id_ed25519_mi300x"
while [ $# -gt 0 ]; do
  case "$1" in
    --alias) ALIAS="$2"; shift 2;;
    --host)  HOST="$2"; shift 2;;
    --user)  USER_NAME="$2"; shift 2;;
    --port)  PORT="$2"; shift 2;;
    --key)   KEY="$2"; shift 2;;
    *) die "unknown arg: $1";;
  esac
done

banner "SSH setup for the cloud MI300X"

# 1. ensure ssh client
have ssh-keygen || die "openssh-client missing — run: sudo apt-get install -y openssh-client"

# 2. generate key (idempotent)
mkdir -p "$HOME/.ssh"; chmod 700 "$HOME/.ssh"
if [ -f "$KEY" ]; then
  ok "key already exists: $KEY"
else
  step "Generating ed25519 key at $KEY"
  ssh-keygen -t ed25519 -f "$KEY" -N "" -C "mi300x-$(whoami)@$(hostname)"
  ok "created $KEY and ${KEY}.pub"
fi

# 3. prompt for host/user if not given
[ -n "$HOST" ] || read -rp "Cloud host/IP: " HOST
[ -n "$USER_NAME" ] || read -rp "Cloud username [ubuntu]: " USER_NAME
USER_NAME="${USER_NAME:-ubuntu}"

# 4. write ~/.ssh/config entry (idempotent)
CFG="$HOME/.ssh/config"; touch "$CFG"; chmod 600 "$CFG"
if grep -qE "^Host[[:space:]]+$ALIAS\$" "$CFG"; then
  warn "Host '$ALIAS' already in $CFG — leaving it unchanged"
else
  step "Adding Host '$ALIAS' to $CFG"
  {
    echo ""
    echo "Host $ALIAS"
    echo "    HostName $HOST"
    echo "    User $USER_NAME"
    echo "    Port $PORT"
    echo "    IdentityFile $KEY"
    echo "    ServerAliveInterval 60"
  } >> "$CFG"
  ok "added"
fi

banner "Next steps"
echo "1. Give this PUBLIC key to your cloud provider (or append to the box's ~/.ssh/authorized_keys):"
echo
echo "${BLD}$(cat "${KEY}.pub")${RST}"
echo
echo "   If you can already log in with a password, push it automatically:"
echo "     ssh-copy-id -i ${KEY}.pub ${USER_NAME}@${HOST}"
echo
echo "2. Test the connection:   ${BLD}ssh $ALIAS${RST}"
echo "3. Copy this repo across:  ${BLD}rsync -av --exclude runs/ $REPO_DIR/ $ALIAS:~/mi300x-pipeline/${RST}"
