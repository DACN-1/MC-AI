#!/usr/bin/env bash
# Run this script ON the LMU cluster login node (after `ssh remote.cip.ifi.lmu.de`)
# to push the pre-extracted trajectory data to a rented vast.ai 5090 box.
#
# Usage:
#   bash scripts/push_data_to_5090.sh <vast_ssh_host> <vast_ssh_port>
#
# Example:
#   bash scripts/push_data_to_5090.sh ssh7.vast.ai 10678
#
# What it does:
#   1. Copy this user's repo + ssh pubkey to the vast.ai box (rsync -avz).
#   2. Push ~/BIG/trajectories/ to /workspace/trajectories/ on the box.
#
# The transfer is ~120 GB. At ~50 MB/s cluster outbound ≈ 40 min; at $16/TB
# vast.ai inbound ≈ $1.92 in bandwidth charges. The cluster's outbound link is
# usually faster than your home upload, so push from here rather than via Mac.
#
# Prereq: the vast.ai box must accept your cluster user's SSH key. On the
# vast.ai web console "Open SSH" prints the host/port; copy your cluster
# ~/.ssh/id_ed25519.pub into the box's ~/.ssh/authorized_keys first.

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "usage: $0 <vast_ssh_host> <vast_ssh_port>" >&2
    echo "example: $0 ssh7.vast.ai 10678" >&2
    exit 1
fi

VAST_HOST="$1"
VAST_PORT="$2"
VAST_USER="${VAST_USER:-root}"
REPO_ROOT="${REPO_ROOT:-$HOME/BIG}"
REMOTE_TRAJ="${REMOTE_TRAJ:-/workspace/trajectories}"
REMOTE_REPO="${REMOTE_REPO:-/workspace/r1-va}"

ssh_opts="-p $VAST_PORT -o StrictHostKeyChecking=accept-new"

echo "=== Verifying box reachability ==="
ssh $ssh_opts "$VAST_USER@$VAST_HOST" "nvidia-smi --query-gpu=name --format=csv,noheader"

echo
echo "=== Step 1/2: pushing repo source to $REMOTE_REPO ==="
# Exclude large/derived artifacts; rsync is idempotent so repeats are cheap.
rsync -avz --progress \
    -e "ssh $ssh_opts" \
    --include='*/' \
    --include='*.py' --include='*.sh' --include='*.txt' --include='Dockerfile' \
    --include='*.json' --include='*.md' --include='*.toml' --include='*.yml' \
    --exclude='trajectories/' --exclude='caches/' --exclude='output/' \
    --exclude='output_cluster/' --exclude='logs/' --exclude='models/' \
    --exclude='rollout_logs/' --exclude='rollout_clip_indist/' --exclude='paper/' \
    --exclude='docs/' --exclude='.venv/' --exclude='.git/' \
    --exclude='llava_model_cache/' --exclude='__pycache__/' --exclude='*.pyc' \
    --exclude='.DS_Store' --exclude='*.mp4' --exclude='*.npy' \
    --exclude='debug/' --exclude='rollouts/' \
    "$REPO_ROOT/" \
    "$VAST_USER@$VAST_HOST:$REMOTE_REPO/"

echo
echo "=== Step 2/2: pushing trajectories to $REMOTE_TRAJ (~120 GB; ~40 min) ==="
date
rsync -avz --progress --partial --partial-dir=.rsync-partial \
    -e "ssh $ssh_opts" \
    "$REPO_ROOT/trajectories/" \
    "$VAST_USER@$VAST_HOST:$REMOTE_TRAJ/"
date

echo
echo "=== Done. Next: SSH to the box and launch the Phase C build ==="
echo "  ssh $ssh_opts $VAST_USER@$VAST_HOST"
echo "  cd $REMOTE_REPO && bash scripts/launch_5090_phase_c.sh"
