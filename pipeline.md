# Pipeline summary (as of 2026-05-28)

End-to-end view of the r1v-a behavioural-cloning pipeline: from BASALT
contractor tarballs to a trained action head and an in-MineRL rollout. For
deeper detail on individual stages see `CLAUDE.md` (codebase) and
`HANDOFF.md` (live cluster state).

## What we're training

A lightweight MLP **action head** on top of a frozen vision-language
backbone (LLaVA-1.5-7B or CLIP), predicting MineRL gameplay actions from
single RGB frames. 21 binary actions (BCE) + 2 mu-law-quantized camera axes
(11-way categorical CE per axis). Optional past-action context (K=8) and
chunked target prediction (N=8) compose with a `--no-language` baseline,
forming the ablation grid below.

## Ablation grid (this run)

Combined `chop_a_tree + collect_dirt` dataset = **6.24 M overlapping frame
windows from 2,081 trajectories**, training one head per cell:

|                   | use_language=1 | use_language=0 |
|-------------------|----------------|----------------|
| LLaVA-1.5-7B (fp16) | cell A       | cell B         |
| CLIP-ViT-L/14     | cell C         | cell D         |

Past-action K=8, chunk_size=8 in all cells. Each cell answers: does the
language prompt inside the backbone help BC on this combined task?

## Stage-by-stage flow

### 1. Data staging (one-off per cluster node)
- Tarballs (`trajectory_task_{chop_a_tree,collect_dirt}_length_3000.tar.gz`,
  ~118 GB combined) live under `~/BIG/trajectories` on the cluster home
  filesystem.
- `slurm_train.sh` extracts both tarballs to `/var/tmp1/cuencanieto/
  trajectories/` on the assigned compute node. **Idempotent per-node**:
  first job extracts (~20 min), subsequent jobs skip.
- `/var/tmp1` wipes on the weekly compvis26 reboot — re-extract needed
  after that.

### 2. Feature-cache build (the bottleneck)
- `feature_cache.precompute` (in `feature_cache.py`) iterates every (mp4,
  frame, task_text) tuple, runs the frozen backbone once, mean-pools the
  final hidden state, writes `(N_samples, feature_dim)` fp16 to a memmapped
  `.npy` plus a JSON metadata sidecar and a `.progress` cursor file.
- **Resumable**: `.progress` is rewritten atomically every 100 batches; a
  killed job resumes from the last checkpoint after metadata-match.
- Per-cell sizes:
  - LLaVA tag: 6.24 M × 4096 × fp16 ≈ **48 GB**
  - CLIP tag:  6.24 M × 1536 × fp16 ≈ **18 GB**
- Per-cell wall-time on A5000 24 GB (measured this run):
  - **CLIP @ batch=64**: 1.16 s/iter → ~32 h
  - **LLaVA @ batch=64** (O1 hook + FA2, post-optimization): target ≤35
    ms/sample → ~60 h (vs. ~240 h at the original batch=16)

### 3. Head training (cheap)
- `imitation_learning.train_cached_head` loads the cache memmap directly
  into a `CachedFeatureDataset`, trains the MLP head for 10 epochs at
  batch=256, lr=1e-3.
- ~50 µs per sample of head compute; **~2 h/cell** at 10 epochs.
- Epoch-level atomic checkpoints + auto-resume on restart.

### 4. Rollout (eval, optional)
- `run_rollout.py` loads `model.pt`, instantiates a fresh MineRL env, runs
  N episodes with logits→action mapping (`action_mapping.py`). Records
  per-episode metrics via `eval_logger.py`.
- Runs in Docker on Apple Silicon dev machine (`docker compose run minerl
  run_rollout.py …`) — MineRL doesn't install on Python 3.12 cluster venv.

## The speed-critical path (where this session's work landed)

Before today, LLaVA's `encode()` asked the backbone for
`output_hidden_states=True`, which retained all 32 transformer layer
activations (~10 GB at batch=64) and forced `cache_batch_size=16` to avoid
A5000 OOM. The two changes pushed in:

1. **O1 — Forward-hook for the final hidden state.** Drop
   `output_hidden_states=True`; register a hook on the LLaMA final RMSNorm
   (`getattr(language_model, "model", language_model).norm`) to capture
   only the last layer. Frees ~10 GB. Code in `VLAAgent.encode`.
2. **FA2 — FlashAttention-2.** `attn_implementation="flash_attention_2"`
   on `from_pretrained`. Eliminates the quadratic attention activation
   tensor and speeds up prefill. flash-attn 2.6.3 cu123/torch2.4/cp312
   wheel pinned in `requirements.txt`. Try/except falls back to sdpa on
   macOS (no flash-attn there).

Net: `cache_batch_size` cap lifted from 16 → 64; 240 h per LLaVA cell →
~60 h target. Slurm script default updated to 64 across all backbones.

Verification: `tests/test_encode_equivalence.py` (opt-in via
`R1VA_RUN_LLAVA_INTEGRATION=1`) asserts bitwise-equal pooled output between
old and new code paths on the same input.

## Cluster orchestration

- **Partition**: `Abaki` on LMU CIP. 4 nodes (abakus11, 12, 21, 22), each
  with 1 × RTX A5000 (24 GB), 32-CPU layout.
- **Reservation**: `compvis26` holds the entire partition Sat 06:00 → Mon
  06:00 every week → jobs in flight get killed at the boundary; relies on
  feature_cache + train resumability to handle this gracefully.
- **One job per cell**: `slurm_train.sh` reads `BACKBONE`,
  `USE_LANGUAGE`, `TASK_FILTER` env vars, names output dir accordingly,
  invokes `cluster_pipeline.py` (which calls `feature_cache.precompute`
  then `train_cached_head`).
- **Effective concurrency right now**: 1 GPU. abakus11/12 have been
  `IDLE+NOT_RESPONDING` (flapping to DOWN occasionally) the entire run;
  abakus22 is allocated to another user. So all 4 cells run serially on
  abakus21.

## Current state (snapshot ~2026-05-28 ~14:50 local)

| Cell | Job | Status |
|---|---|---|
| clip_combined_lang   | 151786 | RUNNING on abakus21 (14h 47m, **~47%**, ~17h remaining) |
| clip_combined_nolang | 151787 | PENDING (Resources) |
| llava_combined_lang  | 151794 | PENDING (Priority) — uses O1+FA2 |
| llava_combined_nolang| 151795 | PENDING (Priority) — uses O1+FA2 |

ETA for full grid: **~6 days out** (early week-after-next) assuming
abakus21 stays the only GPU, with one weekend reservation cycle absorbed
mid-LLaVA. If abakus11/12 or abakus22 free up, the LLaVA half parallelizes
and the back end compresses by ~3 days.

## Session-only monitors (this Claude session)

Three crons keep visibility going while the cluster crunches:

- `100e7e91` — every 10 min: queue + active stderr error scan
- `f8a5d737` — hourly at :07: throughput + run progress
- `93b1584e` — every 30 min at :13/:43: node-availability state changes

All session-only; die when this Claude exits. Auto-expire after 7 days.

## Where things live

- **Repo (local Mac)**: `/Users/diego/VSCode/r1-va` — git remote
  `DACN-1/MC-AI`, branch `main`.
- **Repo (cluster)**: `cuencanieto@remote.cip.ifi.lmu.de:~/BIG/`.
- **Trajectory tarballs**: `~/BIG/trajectories/` on cluster (do NOT add
  more — 144 GB quota, ~124 GB used).
- **Per-node working dir**: `/var/tmp1/cuencanieto/{trajectories,caches,
  hf_cache}` — ephemeral, wiped weekly.
- **Output (persistent, per-cell)**: `~/BIG/output/<cell>/{model.pt,
  metrics.json}`.
- **SLURM logs**: `~/BIG/logs/slurm_<jobid>.{out,err}`.

## Pointers

- Codebase walkthrough + workflows: `CLAUDE.md`
- Cluster state + known issues + optimization opportunities: `HANDOFF.md`
- Cluster ops cheatsheet (rsync, monitor, scancel, sbatch): `HANDOFF.md`
  §"Common commands"
- Implementation plan for O1+FA2: `.claude/plans/look-at-the-handoff-
  inherited-toucan.md`
