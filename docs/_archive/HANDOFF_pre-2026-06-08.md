> **SUPERSEDED — 2026-06-08.** This is an archived handoff. The current handoff
> is `docs/HANDOFF.md`. Specific technical details below (Phase C plan, recipe
> arguments) are still referenced from code comments (e.g. `VLAAgent.py`,
> `slurm_train.sh`, `cluster_pipeline.py`) and remain accurate, but the
> proposed timeline / pending tasks are stale.

---

# Cluster training — handoff (2026-05-28)

## What's running right now

| Cell | Job | Status |
|---|---|---|
| combined_clip_lang | 151786 | RUNNING, abakus21 |
| combined_clip_nolang | 151787 | queued |
| combined_llava_lang | 151790 | queued (batch=16) |
| combined_llava_nolang | 151791 | queued (batch=16) |

4-cell ablation grid: `{LLaVA, CLIP} × {lang, nolang}`. All cells train **a single model on the combined chop_a_tree + collect_dirt dataset** (6.24 M overlapping frame windows from 2081 trajectories).

(A 5th task-ID cell was prototyped and then reverted at user request — see commit log if you want to resurrect.)

Cron `100e7e91` runs `check for errors in queue` every 10 min in this session. Will not survive Claude exiting.

## Open problems

### P1. LLaVA cache build is the bottleneck — 240 h ETA per cell

LLaVA encoder forward at `cache_batch_size=16` is ~136 ms/sample on the A5000. Combined dataset is 6.24 M samples → ~240 h per cache, well past the 48 h SLURM ceiling. Each cell needs ~5 sequential resume-jobs (`feature_cache.precompute` resumes via `.progress` cursor written every 100 batches).

Wall-clock budget with one contended GPU: **~2-3 weeks for the LLaVA half of the grid**, broken across multiple weekend reservation cycles. CLIP half finishes inside one weekend (~64 h for both cells).

**Why not bigger batch?** OOM. LLaVA-7B fp16 = 14 GB weights; the other 10 GB on A5000 has to hold per-layer activations. `VLAAgent.encode` requests `output_hidden_states=True` so it can grab `hidden_states[-1]` — this retains ALL 32 layers simultaneously (~10 GB at batch=64). Batches 32 and 64 both OOM'd in practice (23.3 GiB and 22.8 GiB peak).

### P2. abakus22 contested, abakus11/12 unresponsive

We effectively have one GPU (abakus21). The 1N nodes (abakus11/12) have been `idle*` (down/unresponsive) for the entire run. SLURM hasn't auto-recovered them. abakus22 has been allocated to another user the whole time.

### P3. compvis26 weekly reservation kills jobs every weekend

Reservation `compvis26` holds all four Abaki nodes Sat 06:00 → Mon 06:00 every week. `IGNORE_JOBS` flag: any job still running gets killed when the reservation starts. The `--time=2-00:00:00` setting was chosen to keep jobs inside one weekday window; but for LLaVA's multi-resume builds, jobs WILL be killed by the reservation. Resumability handles it gracefully — just adds wall-clock latency.

### P4. `transformers` warns on `TRANSFORMERS_CACHE` deprecation

Cosmetic. Will be removed in transformers 5.x. Drop the `export TRANSFORMERS_CACHE=...` line in `slurm_train.sh` whenever you next touch it; `HF_HOME` covers it.

## Optimization opportunities (by ROI)

### O1. Hidden-states forward hook → unlocks batch=64 for LLaVA (4× speedup) — RECOMMENDED

The big one. One-line change to `VLAAgent.encode`: instead of `output_hidden_states=True`, attach a forward hook on `self.llava.language_model.model.norm` to capture only the final hidden state. Frees ~10 GB, batch=64 will fit (probably batch=96 too).

```python
captured = {}
def _grab(mod, args, out):
    captured["h"] = out
handle = self.llava.language_model.model.norm.register_forward_hook(_grab)
try:
    _ = self.llava(input_ids=..., attention_mask=..., pixel_values=...)
finally:
    handle.remove()
pooled = captured["h"].mean(dim=1)
```

Verify: load a checkpoint, run `encode([img], ["chop a tree"])` before/after the change, confirm the output is bitwise-equal (it should be — `hidden_states[-1]` IS `norm` output).

**Expected impact**: LLaVA cache build 240 h → ~60 h per cell. Still needs ~2 jobs per cell but finishes in one weekend instead of three.

### O2. Frame subsampling (stride 4 or 8) — defer until O1 tried

`enumerate_samples` and `TrajectoryDataset._add_samples` both iterate `range(len(actions))`. Changing to `range(0, len(actions), STRIDE)` cuts N proportionally. Standard BC practice — adjacent frames are highly redundant (Δt = 33 ms at 30 fps). At stride=4 → 1.5 M samples → 60 h per LLaVA cache (with current batch=16). At stride=4 + O1 → ~15 h per LLaVA cache (single job).

Catches:
- Cache tag must include stride to avoid cross-stride mixing (e.g. `llava_combined_lang_s4`).
- `chunk_size` interaction: with stride 4 and chunk 8, target windows are 32 demo frames wide.
- `past_action_k=8`: same, past window grows.

Worth the code change only if O1 alone isn't fast enough.

### O3. Bigger CLIP batch — small gain, low risk

CLIP at batch=64 uses tiny VRAM (~5 GB total). Could try batch=256 or larger. Probably not bottlenecked by GPU at that point — decord I/O takes over. Marginal: each CLIP cell is already only ~32 h.

### O4. Decoder worker count tuning

Current `--num-workers=8` with `decoder_cache_size=64` per worker → up to 512 open MP4 handles per job. Could try `num_workers=16` for the encode loop. Probably won't move the needle since LLaVA forward dominates per-batch time.

### O5. Stage data once at job submit, not per-job

The script extracts both tarballs on first job per node. Each (node × first-time-job) costs ~10 min of tar extraction. If the cluster admin enables persistent `/var/tmp1` or gives us a project-scratch mount, we could pre-stage once and never re-extract.

### O6. Multi-GPU per cell via DataParallel

Not applicable: we have effectively 1 GPU, jobs are single-node single-GPU.

## Operational gotchas

- `/var/tmp1` wipes on weekly reboot → caches lost → next job rebuilds via `.progress` resume.
- `~/BIG` (home) quota: 144 GB total, currently ~124 GB used. Almost full because both 64 GB tarballs live there. **Don't store any new big files in `~/BIG`.** Use `/var/tmp1` (per-node, ephemeral).
- SLURM `RealMemory=1` and `Gres=(null)` are placeholders — never request `--mem` or `--gres`, jobs reject instantly.
- The login node is `aquamarin` (no GPU). All `/var/tmp1` inspection has to happen via `srun --jobid=<X> --overlap` while a job is running on that node.
- `slurm_train.sh` extracts tarballs idempotently. First job per node = ~20 min staging. Subsequent jobs = 0 min.
- `_ensure_cache` was buggy and would silently train a head on zeros if cache was partial. Fixed in commit `84c138e` to consult `.progress` vs `n_samples`.

## Where things live

- **Repo (local Mac)**: `/Users/diego/VSCode/r1-va` — git remote `DACN-1/MC-AI`, branch `main`
- **Repo (cluster)**: `cuencanieto@remote.cip.ifi.lmu.de:~/BIG/` — keep in sync via `rsync` of changed files
- **SSH key**: `~/.ssh/id_lmu` (no passphrase). No alias yet; long-form host.
- **Trajectory tarballs**: `~/BIG/trajectories/trajectory_task_{chop_a_tree,collect_dirt}_length_3000.tar.gz` (118 GB combined)
- **Per-node working dir**: `/var/tmp1/cuencanieto/{trajectories,caches,hf_cache}` (ephemeral)
- **Output (persistent, per-cell)**: `~/BIG/output/<cell>/{model.pt,metrics.json}`
- **SLURM logs**: `~/BIG/logs/slurm_<jobid>.{out,err}`
- **Monitor**: `bash ~/BIG/monitor.sh [snapshot|tail|watch]`

## Common commands

```bash
# Monitor cluster
ssh -i ~/.ssh/id_lmu cuencanieto@remote.cip.ifi.lmu.de 'bash ~/BIG/monitor.sh'

# Tail running job
ssh -i ~/.ssh/id_lmu cuencanieto@remote.cip.ifi.lmu.de 'bash ~/BIG/monitor.sh tail'

# Sync code change to cluster
rsync -avh -e "ssh -i ~/.ssh/id_lmu" <files> cuencanieto@remote.cip.ifi.lmu.de:~/BIG/

# Cancel all my jobs
ssh -i ~/.ssh/id_lmu cuencanieto@remote.cip.ifi.lmu.de 'scancel -u $USER'

# Submit one cell (combined dataset; TASK_FILTER unset = combined)
ssh -i ~/.ssh/id_lmu cuencanieto@remote.cip.ifi.lmu.de \
  'cd ~/BIG && BACKBONE=llava USE_LANGUAGE=1 sbatch slurm_train.sh'

# Pull results back to Mac
rsync -avh -e "ssh -i ~/.ssh/id_lmu" \
  cuencanieto@remote.cip.ifi.lmu.de:~/BIG/output/ \
  /Users/diego/VSCode/r1-va/output_cluster/
```

## Confound flagged in 2026-05-28 conversation (research note, not a code issue)

In the 4-cell base ablation, `nolang` cells have no signal to disambiguate
chop_a_tree vs collect_dirt — the model averages a single policy over both
tasks. So the `lang vs nolang` comparison conflates "language inside backbone
helps" with "any task disambiguation helps." Cell 5 (`combined_llava_nolang_taskid`)
gives a third reference point: no language but explicit task ID into the head
→ measures pure task-disambiguation contribution. Discuss in §3.6 of the paper.

## Recent commits

```
4bd5e35  fix: LLaVA cache_batch_size 32 -> 16 (A5000 OOMs again under transformers 4.49)
18b7dd9  feat: combined-dataset ablations + optional task-ID conditioning
e5ef163  fix: cap LLaVA cache_batch_size at 32 to avoid A5000 OOM
84c138e  fix: cache-reuse check must consult .progress, not just file existence
101f0f4  fix: bump transformers to >=4.45 so tokenizers >=0.20 lands
a303da4  chore: drop --mem and trim --time to 48h for Abaki reservation window
180aabb  chore: bump deps for Python 3.12 on LMU CIP
14a19fa  chore: migrate SLURM + download scripts from JURECA to LMU CIP Abaki
d07d5bb  chore: drop chunk_frames.py + h5py dep, refresh docs
```

## Immediate next actions (recommended)

1. Implement O1 (hidden_states hook), local-test, sync, cancel queued LLaVA cells, resubmit at batch=64. **Single biggest win available.**
2. If still too slow: O2 (frame stride=4).
3. Leave CLIP cells alone — they're working.
