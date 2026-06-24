#!/bin/bash
# Robust chunked upload of a large file to the vast box over a flaky uplink.
# Per-chunk HARD TIMEOUT (background-kill) so a mid-transfer STALL is killed and
# retried instead of hanging forever. Small chunks => a stall is cheap to retry.
# Usage: chunk_upload_generic.sh <local_src> <box_dest_abs> <want_sha1>
set -uo pipefail
cd /Users/diego/VSCode/r1-va
SRC="$1"; DEST="$2"; WANT_SHA="$3"
CHUNK=4m; TMO=60
CDIR="$(dirname "$DEST")/.chunks_$(basename "$DEST")"
# Target configurable via env (defaults to the vast box). For the cluster:
#   R1VA_SSH_HOST=cuencanieto@remote.cip.ifi.lmu.de R1VA_SSH_PORT=22 R1VA_SSH_KEY=~/.ssh/id_lmu
: "${R1VA_SSH_HOST:=root@ssh5.vast.ai}"; : "${R1VA_SSH_PORT:=11258}"; : "${R1VA_SSH_KEY:=}"
KEYOPT=""; [ -n "$R1VA_SSH_KEY" ] && KEYOPT="-i $R1VA_SSH_KEY"
SSHB() { ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o BatchMode=yes \
  -o ServerAliveInterval=10 -o ServerAliveCountMax=3 $KEYOPT -p "$R1VA_SSH_PORT" "$R1VA_SSH_HOST" "$@"; }

# bounded transfer: ssh in bg, kill if it exceeds $TMO seconds
put_chunk() { # $1=localfile $2=destpath
  SSHB "cat > $2" < "$1" & local pid=$!
  ( sleep "$TMO"; kill -9 $pid 2>/dev/null ) & local killer=$!
  wait $pid 2>/dev/null; local rc=$?
  kill "$killer" 2>/dev/null; wait "$killer" 2>/dev/null
  return $rc
}

TMP=$(mktemp -d)
split -b "$CHUNK" "$SRC" "$TMP/chunk_"
N=$(ls "$TMP"/chunk_* | wc -l | tr -d ' ')
echo "split into $N chunks ($(du -h "$SRC"|cut -f1)), chunk=$CHUNK timeout=${TMO}s"
SSHB "rm -f $DEST; mkdir -p $CDIR; rm -f $CDIR/*" >/dev/null

i=0
for f in "$TMP"/chunk_*; do
  i=$((i+1)); name=$(basename "$f"); want=$(stat -f%z "$f"); got=0
  for att in $(seq 1 12); do
    put_chunk "$f" "$CDIR/$name"
    got=$(SSHB "stat -c%s $CDIR/$name 2>/dev/null || echo 0" 2>/dev/null | tr -d ' ')
    [ "$got" = "$want" ] && break
    echo "  retry $name att=$att got=$got/$want"; sleep 1
  done
  [ "$got" = "$want" ] || { echo "FAILED chunk $name after retries"; rm -rf "$TMP"; exit 1; }
  echo "[$i/$N] OK $name"
done

echo "reassembling..."
SSHB "cat $CDIR/chunk_* > $DEST && rm -rf $CDIR"
boxsha=$(SSHB "sha1sum $DEST | awk '{print \$1}'" | tr -d ' ')
rm -rf "$TMP"
echo "box sha1: $boxsha"; echo "want sha1: $WANT_SHA"
[ "$boxsha" = "$WANT_SHA" ] && echo "UPLOAD_OK" || echo "UPLOAD_SHA_MISMATCH"
