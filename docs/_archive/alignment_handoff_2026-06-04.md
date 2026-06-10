> **SUPERSEDED — 2026-06-08.** This is an archived handoff. The current handoff
> is `docs/HANDOFF.md`. Specific technical details below (Phase C plan, recipe
> arguments) are still referenced from code comments (e.g. `VLAAgent.py`,
> `slurm_train.sh`, `cluster_pipeline.py`) and remain accurate, but the
> proposed timeline / pending tasks are stale.

---

# Handoff: aligning the LLaVA and CLIP backbones for the 4-cell ablation

## Why this exists

The original ablation (May 28) is a 2x2 grid:

|              | with prompt | without prompt |
|--------------|-------------|----------------|
| **LLaVA**    | exp 1       | exp 3          |
| **CLIP**     | exp 2       | exp 4          |

The four pairwise comparisons each answer a specific question — *does the
prompt do anything inside LLaVA* (1 vs 3), *inside CLIP* (2 vs 4), *which
backbone is the better vision-only encoder* (3 vs 4), and *who wins with full
language* (1 vs 2). The spec explicitly says **"all four cells use identical
training hyperparameters, dataset, and action head architecture."**

The implementations currently in `VLAAgent.py` (LLaVA) and
`frozen_vision_baseline.py` (CLIP) violate that — three asymmetries below.
Until they are fixed, none of the four pairwise comparisons mean what the
spec claims they mean. Today's empirical finding ("LLaVA has weak feature
discrimination, fires every action under sample decoding") may instead be
a finding about *implementation asymmetry* — we don't know which.

## The three asymmetries

### 1. Pooling philosophy (the big one)

| | What `encode()` returns | Pool over what |
|---|---|---|
| **CLIP** (`frozen_vision_baseline.py:54-67`) | `[image_proj || text_proj]`, 2x embed_dim = 1536 | two dedicated projection heads (vision pooler + text pooler), concat |
| **LLaVA** (`VLAAgent.py:89-132`) | one joint vector, 4096 | mean of post-norm hidden state across **all** (image + text) tokens |

CLIP gives the head a clean `[vision || text]` separation. LLaVA collapses
both modalities into one arithmetic mean over a heterogeneous token sequence
(typically ~576 image tokens and only ~5 prompt tokens — the text gets
washed out before the head even sees it). That dilution alone is a strong
candidate for "LLaVA's prompt doesn't matter in the head's output."

### 2. `use_language=False` semantics

CLIP nolang: `text_features = th.zeros_like(image_features)`
  (`frozen_vision_baseline.py:66`) — head sees `[image_proj || zeros]`,
  truly image-only.

LLaVA nolang: `effective_texts = [""] * len(texts)` (`VLAAgent.py:96`) →
prompt becomes `"<image>\n"` → the mean pool still includes those (few but
nonzero) text-side hidden states. Not image-only.

Consequence: exp 3 vs exp 4 ("which backbone is the better vision-only
encoder?") isn't comparing vision-only systems.

### 3. Head capacity scales differently — but maybe not as bad as it looks

End-to-end heads in the two modules:

| | feature_dim | head hidden | compression |
|---|---:|---:|---:|
| CLIP (`frozen_vision_baseline.py:48-52`) | 1536 | 768 | **2x** |
| LLaVA (`VLAAgent.py:78-82`) | 4096 | 4096 | **1x** |

**Caveat:** cached-feature training (the path actually used for every run
this branch has produced) uses `HeadOnlyAgent` in `feature_cache.py:451`,
which defaults `hidden_dim = feature_dim`. So under the cached path *both*
backbones end up at the 1x compression — CLIP cache head is 1536 -> 1536,
LLaVA cache head is 4096 -> 4096. That's the same ratio, just different
absolute width. Whether the spec's "identical" requires *same ratio* or
*same absolute hidden* is the call to make. If the latter, both should be
pinned to one number (say 2048) regardless of backbone.

## Order of operations (cheap -> expensive)

This respects what's already on disk. Don't rebuild caches before knowing
whether a head-only change is enough.

### Step 1 — Pin a single head-hidden across all 4 cells (no cache rebuild)

`HeadOnlyAgent(hidden_dim=H)` is already a parameter — just plumb a
constant through `train_cached_head`. Pick `H = 2048` (between 768 and
4096) and use it for **all** 4 cells. ~ 30 min head training per cell on
existing caches.

Why this step alone: if LLaVA's flat-logit behaviour was caused by *head
capacity differing from CLIP*, fixing this alone moves LLaVA's action
distribution toward CLIP's. If not, the asymmetry that matters is pooling
(step 2) or `use_language` (step 3) — and step 1 cost almost nothing to
rule out.

**Critical files:** `feature_cache.py:451-486` (HeadOnlyAgent — already
takes hidden_dim), `imitation_learning.py:train_cached_head` (plumb the
flag through; check existing signature for `hidden_dim`),
`cluster_pipeline.py` / `slurm_train.sh` (expose `--hidden-dim` and
default it to `2048`).

### Step 2 — Rebuild LLaVA cache with split image/text pooling (LLaVA cache only)

Rewrite `VLAAgent.encode()` so it:

1. Runs the forward pass exactly as today (FA2 + hidden-states hook on
   RMSNorm — `VLAAgent.py:107-126` is the load-bearing code, do not touch
   the hook or FA2 init).
2. After capture, slices the post-norm hidden state by **token role**:
   image tokens (the positions that got expanded from `<image>`) versus
   text tokens (everything else, excluding pad). The LlavaProcessor's
   `input_ids` carries this — `LlavaConfig.image_token_index` is the
   marker. Tokenizer's `pad_token_id` covers the rest.
3. Mean-pools the two groups separately, concatenates ->
   `[image_pool || text_pool]`. Feature dim becomes 8192 (2 * 4096).
4. When `use_language=False`: literally zero `text_pool` in this method
   (same as `frozen_vision_baseline.py:66`). No more `texts=[""]` hack —
   the empty-text path is identical to a zero text branch.

**Verify the slicing** against a known input before rebuilding the cache:
the LlavaProcessor expands `<image>` to N image tokens (N = number of
patches from the vision encoder, typically 576). Construct a test prompt
`"<image>\nchop a tree"`, run the processor, confirm `(input_ids ==
image_token_index).sum() == N` and the rest are text tokens. Test gate
this behind `R1VA_RUN_LLAVA_INTEGRATION=1` like
`tests/test_encode_equivalence.py` already does (`tests/...`).

This change increases cache feature_dim 4096 -> 8192, so `feature_cache.py`
will detect mismatch and rebuild from scratch (the existing
`meta_existing.get("feature_dim") == feature_dim` check on
`feature_cache.py:258` does this safely). Don't try to migrate the old
cache.

**Cost:** ~5 h to build one LLaVA cache (chop, stride 4, lang+nolang both
covered by zeroing the text branch at head input) on a rented RTX 3090
(~$0.25/h, ~ $1.25). Stride 8 would be half.

**On the CLIP side:** no change needed. CLIP `encode()` already returns
`[image_proj || text_proj]` with text=zeros when `use_language=False`.
Existing CLIP cache is reusable. Verify by reading
`output_cluster/clip_chop_a_tree_lang_nomove_stride8.brokenW/metrics.json`
or any synced CLIP cell.

### Step 3 — One aligned 4-cell rollout on chop, then decide

After step 2 lands and the new LLaVA head is trained against the new cache:

- Roll out all 4 cells on `MineRLChopATree640-v0` with the *same* decode
  settings (`--sample --temperature 1.0 --color-match auto`, 5 ep x 1000
  steps). Use the inference server + SSH tunnel pattern from this
  session.
- Compare action distributions (per-action fire rates, *not* the
  dominant-action picker — see the comparison table in section "Decoding
  findings" below).
- If LLaVA cells now look comparable to CLIP cells: the asymmetry was
  the implementation. If LLaVA still loses, the conclusion is genuine
  (and the paper has a clean finding either way).

### Out of scope for this handoff

- Retraining with the `no_move_fix` weighted-loss + history-dropout recipe.
  Today's chop-FIX runs showed `pos_weight[attack]=0.50` overcorrected on
  chop (attack -> 1.7%) while OK-ish on dirt (attack -> 42%). Whether the
  weighted recipe is the right answer for *aligned* LLaVA is unknown
  until step 3 completes.
- Retraining the dirt-cell LLaVA. Steps 1-3 are scoped to chop so the
  aligned-pooling cache rebuild is cheap. Dirt extends the same recipe
  once chop is validated.

## What's already on disk

### Heads (trained)

- `output_cluster/clip_chop_a_tree_lang_nomove_stride8/model.pt` —
  CLIP chop lang, weighted loss + history dropout (no_move_fix recipe).
  Trained Jun 4 morning.
- `output_cluster/clip_collect_dirt_lang_nomove_stride8/model.pt` —
  dirt counterpart.
- `output_cluster/clip_combined_nolang/model.pt` — May 29 baseline,
  *unweighted*, chop-only data (the "combined" tag is a misnomer per
  `docs/rollouts.md`). Produced the empirically-best rollout behaviour
  this branch has seen.
- `output/llava_chop_a_tree_lang_stride4/model.pt` — Jun 3 LLaVA, chop
  lang, unweighted recipe. **This** is the head we tested today via
  vast.ai SSH-tunnel + JPEG transport. config has no `weighted_loss` /
  `history_dropout` keys — defaults to off.

The 4-cell ablation is **not** complete — only one LLaVA cell exists
locally, and the CLIP cells use the no_move_fix recipe (different from
the LLaVA cell). No two cells in the current state share both the recipe
and the alignment, so cross-cell claims are not currently supported by
the data.

### Caches

- `caches/` on Mac — limited, mostly empty after the reorg.
- Cluster `/var/tmp1` had the LLaVA-chop-stride8 cache before the most
  recent compvis26 reservation reboot; assume gone.
- HF Hub cache: the rented vast.ai box (now destroyed) had the
  LLaVA-1.5-7B base weights cached. Re-renting takes the same ~11 min
  to redownload.

## Decoding findings (relevant for interpreting step 3's results)

Today's 4 decoding experiments on the chop FIX CLIP model (5000 steps
each, `MineRLChopATree640-v0`):

| Cell | attack | forward | back | left | right | jump | camera |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline CLIP + sample (replicate) | 58.7% | 48.3% | 2.3% | 24.5% | 11.2% | 10.3% | 67.8% |
| chop FIX + sample (uncorrected) | 1.7% | 83.3% | 36.3% | 53.0% | 58.7% | 80.3% | 98.3% |
| FIX + `T_attack=5.0` + sample | 39.8% | 51.7% | 52.8% | 72.6% | 44.9% | 45.2% | 98.2% |
| FIX + `thr_attack=0.005` + greedy | 100% | 82.1% | 0% | 59.5% | 0% | 78.1% | 60.0% |
| FIX + `bias_attack=+6.0` + sample | 98.9% | 41.3% | 58.9% | 65.3% | 59.1% | 38.4% | 96.6% |
| LLaVA + sample (ep 0 only) | 37.9% | 23.6% | **51.9%** | 22.3% | 24.1% | 13.4% | 71.0% |

LLaVA fires `back` at 51.9% — z mean **+0.07** in `steps_000.json`, near
even odds. That is the visible "goes backwards a lot" behaviour the user
flagged. The story isn't decoding miscalibration on one action — it's
the *flat* logit distribution across the entire binary block (std 1.3-2.7,
all means clustered near 0). Decoding fixes can re-shape per action; only
better-discriminative features fix the cause.

The CLI flags for these calibration experiments are live in `run_rollout.py`
already: `--binary-temperatures attack=5.0`, `--binary-thresholds attack=0.005`,
`--binary-logit-bias attack=6.0`. Helper: `action_mapping.build_per_action_vector`.

## Reward note

Every rollout this session ended with `total_reward=0`. Per
`eval_envs.py:99`, `MineRLChopATree640-v0` rewards on *log entering
inventory*, not block break. Bare-handed mining (`inventory=[]` per
`eval_envs.py:156`) requires ~5 sec of sustained `attack` on the same
block; with current camera/back stochasticity the agents never park long
enough for a log to drop and be walked-into. Behavioral comparisons via
action-distribution remain valid; reward isn't a useful signal for these
short rollouts yet.

## Infrastructure notes for next session

- `scripts/setup_vastai_inference.sh` (added today) — minimal inference-
  only stack (torch+cu121, transformers, no flash-attn, no decord). Works
  on any RTX 3090-class spot.
- `inference_server.py` (modified today) — accepts both raw `pov` and
  JPEG-compressed `pov_jpeg` payloads, runs HTTP/1.1 with keep-alive.
- `run_rollout.py:_RemoteAgent` (modified today) — persistent
  `http.client.HTTPConnection`, JPEG-encodes the frame before sending,
  retries once on dropped connection. ~1 s/step over a US-CA SSH tunnel
  from home upstream; was 16 s/step before keep-alive landed.
- Vast.ai pick: filter on `gpu_name=RTX_3090 reliability>=0.97
  inet_down>=400 disk_space>=40`, *exclude CN hosts* (Docker Hub
  rate-limited there — instance 39470383 wasted ~30 min stuck on layer
  retries). The `34108482 / 31323636 / 39477125` series is illustrative
  but offer IDs cycle quickly.

## Files to read first when picking this back up

1. `docs/no_move_fix.md` — context on the weighted-loss + history-dropout
   recipe and why it was applied.
2. `frozen_vision_baseline.py:24-82` and `VLAAgent.py:18-160` — the two
   `encode()` methods side by side; the alignment work is here.
3. `feature_cache.py:451-486` — `HeadOnlyAgent` (the cached-feature head;
   already takes `hidden_dim` — step 1 is just plumbing it).
4. `tests/test_encode_equivalence.py` — pattern for opt-in
   LLaVA-integration tests; copy this gating for the new pooling tests.
