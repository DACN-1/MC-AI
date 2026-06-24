#!/bin/bash
# Robust chunked upload of the fixAB slim head to the vast box over a flaky
# tethered uplink. Each chunk is a short, fresh ssh transfer (survives stalls);
# size-verified with per-chunk retry; reassembled + sha1-verified on the box.
set -uo pipefail
cd /Users/diego/VSCode/r1-va
SRC=output/llava_combined_lang_stride4_fixAB/model_best.slim.pt
BDIR=/workspace/r1-va/output/llava_combined_lang_stride4_fixAB
DEST=$BDIR/model_best.pt
CDIR=$BDIR/chunks
WANT_SHA=f488ea84fbfda236f010e2c5acaadf0b14d6d693
SSHB() { ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o BatchMode=yes \
  -o ServerAliveInterval=8 -o ServerAliveCountMax=3 -p 11258 root@ssh5.vast.ai "$@"; }

TMP=$(mktemp -d)
split -b 8m "$SRC" "$TMP/chunk_"
N=$(ls "$TMP"/chunk_* | wc -l | tr -d ' ')
echo "split into $N chunks"
SSHB "rm -f $DEST; mkdir -p $CDIR; rm -f $CDIR/*" >/dev/null

i=0
for f in "$TMP"/chunk_*; do
  i=$((i+1)); name=$(basename "$f"); want=$(stat -f%z "$f")
  got=0
  for att in 1 2 3 4 5 6 7 8; do
    SSHB "cat > $CDIR/$name" < "$f" 2>/dev/null
    got=$(SSHB "stat -c%s $CDIR/$name 2>/dev/null || echo 0" 2>/dev/null | tr -d ' ')
    [ "$got" = "$want" ] && break
    echo "  retry $name att=$att got=$got want=$want"; sleep 2
  done
  [ "$got" = "$want" ] || { echo "FAILED chunk $name"; rm -rf "$TMP"; exit 1; }
  echo "[$i/$N] OK $name ($want B)"
  sleep 1
done

echo "reassembling on box..."
SSHB "cat $CDIR/chunk_* > $DEST && rm -rf $CDIR"
boxsha=$(SSHB "sha1sum $DEST | awk '{print \$1}'" | tr -d ' ')
boxsz=$(SSHB "stat -c%s $DEST")
rm -rf "$TMP"
echo "box sha1: $boxsha"
echo "want sha1: $WANT_SHA"
echo "box size:  $boxsz"
[ "$boxsha" = "$WANT_SHA" ] && echo "UPLOAD_OK" || echo "UPLOAD_SHA_MISMATCH"
