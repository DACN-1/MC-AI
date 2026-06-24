# Method Chapter — Verification Report

**Date:** 2026-06-23
**Scope:** Every factual claim in the thesis *Method* chapter, checked against the
`r1-va` codebase (repo `DACN-1/MC-AI`) — source files **and** the four reported
trajectory-split checkpoints.
**Method:** Direct source reads for the load-bearing claims (architecture, loss,
optimiser, selection, split, decode) + a fan-out of independent verification
agents for the remaining clusters (caching, dataset/provenance, rollout,
eval metrics, reward/envs, deps, citations), each with an adversarial recheck of
any non-confirmed finding. Checkpoint configs were dumped from the `.pt` files
directly.

## Verdict legend

| Mark | Meaning |
|------|---------|
| ✅ | **Confirmed** — code/artefacts implement exactly what the chapter says. |
| 🟡 | **Confirmed for the reported runs, with a caveat** — true as written for the four reported cells / canonical eval path, but the value is a config/flag choice, not a hard code default (so the prose is accurate, the *code default* differs). |
| 🔧 | **Discrepancy — fix the text.** |
| ❌ | **Refuted.** |

---

## 1. Executive summary

**The chapter is overwhelmingly accurate.** Every architectural, loss,
optimiser, data-split, and decode claim checks out against the source, and — the
decisive result — the four reported cells exist on disk as a clean knob-free
2×2 whose stored configs match the chapter line-for-line.

**The four reported checkpoints** are
`{llava,clip}_combined_{lang,nolang}_stride4_tsplit[_base]`. Their stored
`config` blocks confirm, for all four: `hidden_dim=2048`,
`split_by_trajectory=true`, `keep_best=true`, `epochs=10`, `batch_size=256`,
`lr=0.001`, `cam_ce_weight=0.5`, `chunk_size=8`, `past_action_k=8`,
`val_split=test_split=0.1`, and **every** optional knob off
(`weighted_loss=false, learnable_bce_temp=false, focal_gamma=0,
past_action_slot_dropout=0, cam_weighted_loss=false, frame_history_k=0`,
`film`/`feature_norm`/`patch_grid` off). Head parameter counts are exact:
**LLaVA 18,188,632 ≈ 18.2 M, CLIP 4,557,144 ≈ 4.6 M.**

**Three things must be fixed before the chapter is final:**

1. 🔧 **Python version.** The chapter says *Python 3.10*; the stack requires/was
   validated on **Python 3.12** (`requirements-blackwell.txt`, `requirements.txt`;
   3.10 is only the floor of an accept-anything range in `setup_vastai_5090.sh`).
   → change "Python 3.10" to "Python 3.12".
2. 🔧 **Bibliography.** `paper/dbstmpl.bib` is still the **stock LMU template** (2
   real entries: `ABKS99`, `EKSX96` + 1 `@string`). **None** of the chapter's 10
   `\cite` keys exist, there is no `MineDreamer` entry, and there is no
   `shah2021basalt`. The committed `paper/*.tex` is also still template
   boilerplate (`Introduction.tex` still has the German DBSCAN example). → the
   `.bib` must be populated with all 10 keys + a MineDreamer entry before the
   chapter compiles. (This confirms the chapter's own `[VERIFY]` flags #4 and #6.)
3. 🔧 **FastAim note (open-item #7).** The note attributes the log-break ability
   and `BreakSpeedMultiplier=5.0` to the *FastAim* variant. In code the `5.0`
   multiplier is shared by **both** `MineRLChopATree640Fast-v0` and
   `...FastAim-v0` (`eval_envs.py:209-215, 228-235`); FastAim's *only* extra knob
   is `start_pitch=20.0`. → reword the note: the **Fast** variants (5.0
   multiplier) are what let a log break inside budget; FastAim merely adds a fixed
   downward pitch.

Everything else is either ✅ or a 🟡 where the prose is accurate for the reported
setup and the only nuance is "this is a config/flag value, not the bare code
default." Those nuances are listed so they can't surprise an examiner; none
require a text change.

---

## 2. Resolution of the chapter's own OPEN ITEMS

| # | Open item | Resolution |
|---|-----------|-----------|
| 1 | `config["hidden_dim"]=2048` in the four checkpoints | ✅ **Confirmed.** All four tsplit checkpoints store `hidden_dim: 2048`; head W1 shapes are `(2048, 8536)` (LLaVA) / `(2048, 1880)` (CLIP). |
| 2 | Reported 2×2 = `split_by_trajectory=True` + keep-best (the `*_tsplit` cells), not the legacy frame-level baseline | ✅ **Confirmed.** All four store `split_by_trajectory: true` and `keep_best: true`. The clean cells are `{llava,clip}_combined_{lang,nolang}_stride4_tsplit[_base]`. (Use *these* as the reported cells — the legacy `clip_combined_lang_stride4` is frame-level and must not be cited.) |
| 3 | Hardware per run (5090, torch 2.7/cu128) | ✅ **Confirmed** as the canonical path (`requirements-blackwell.txt`, `setup_vastai_5090.sh`, `ARCHITECTURE.md:658`). The reported cells are stride-4 LLaVA/CLIP cells consistent with the 5090 build. *(But fix Python version — open item separate, see §1.)* |
| 4 | Add a `\cite` for MineDreamer | 🔧 **Still open — no MineDreamer bib key exists.** Must be added. |
| 5 | Four anchor cells knob-free (`feature_norm=False, image_dropout=0, learnable_bce_temp=False, patch_grid=0, frame_history_k=0`) | ✅ **Confirmed.** Config blocks show `learnable_bce_temp=false, frame_history_k=0`, and the head param count equals the *pure* MLP with no LayerNorm, which independently proves `feature_norm=False` and `film=False`. CLIP `feature_dim=1536 = 2×768` proves `patch_grid=0`. `image_dropout` is train-time-only (no param/inference trace) and is absent from the config → default `0.0`. |
| 6 | `shah2021basalt` not cited | ✅ **Confirmed** absent from bib and uncited — consistent with deliberate non-citation (data are VPT-agent rollouts, not BASALT contractor data). Nothing to remove. |
| 7 | Prompt-swap protocol / FastAim envs note | 🔧 **Correction needed** (see §1.3): `5.0` is shared by Fast & FastAim; biome envs confirmed single-biome (forest/plains), no generic dual world. |

---

## 3. Claim-by-claim verification

### 3.1 Architecture (§Architecture)

| Claim | Verdict | Evidence |
|-------|---------|----------|
| LLaVA-1.5-7B backbone, frozen; head only is trained | ✅ | `VLAAgent.py:58-72` loads `llava-hf/llava-1.5-7b-hf`, `requires_grad_(False)` on all backbone params. |
| `<image>` placeholder expanded to **576** visual tokens, processed with prompt tokens | ✅ | `VLAAgent.py:124-126` "expands to `image_seq_length` (=576) tokens". |
| Final-layer post-norm hidden state read via **forward hook on the terminal RMSNorm** | ✅ | `VLAAgent.py:152-159` hooks `language_model…model.norm`; comment "same tensor `hidden_states[-1]` would return". |
| **Split pooling**: mask-mean image-role & text-role tokens separately, concat → 8192 (2×4096) | ✅ | `VLAAgent.py:169-174` `pooled = cat([image_pool, text_pool])`; `_split_pool` `_masked_mean` per role (`:178-226`); `_feature_dim = 2*hidden` (`:89`). |
| CLIP-ViT-L/14 baseline: 768 image + 768 text → **1536**, concat | ✅ | `frozen_vision_baseline.py:63` `embed_dim=projection_dim` (=768); `:80` `feature_dim = 2*embed_dim`. Checkpoint `feature_dim:1536`. |
| CLIP no-prompt: text branch → **zero vector** of same dim (head input shape held) | ✅ | `frozen_vision_baseline.py:104-109` zeros `(…, projection_dim)`. |
| Action space: 21 binary + 2 camera; 11 mu-law bins/axis; **D=43** | ✅ | `constants.py:58-64` `NUM_BINARY=21, NUM_CAMERA=2, NUM_CAMERA_BINS=11, NUM_OUTPUT_LOGITS=43`. |
| Camera quantizer `maxval=10, binsize=2, mu=10` → 11 bins; demo values are bin centres (lossless) | ✅ | `vpt_camera.py:75-77`, `n_bins=11`; empirically every recorded camera value equals a bin centre (round-trip lossless). |
| Past-action: last **K=8** actions, 43-d each (21 binary 0/1 + two 11-way one-hots), flattened (**K·D=344**), concat to feature, zero-padded & right-aligned | ✅ | `constants.py:70-71` `PAST_ACTION_DIM=43, DEFAULT_PAST_ACTION_K=8`; `action_to_onehot` (`:126-144`); checkpoint `past_action_dim:344`. |
| Chunking: head emits **N=8** future steps reshaped to (N,D); loss averages chunk; rollout executes only step 1 | ✅ | `VLAAgent.py:252-253` reshape `(B, chunk, output_dim)`; checkpoint `chunk_size:8`; chunk averaging in `vla_loss` (§3.2). |
| Head = 2-layer MLP ReLU: `Linear(F+K·D → H) → ReLU → Linear(H → N·D)`; H=2048; fp32; features cast to fp32; LLaVA bf16, CLIP fp32 | ✅ | `feature_cache.py:593-597` (HeadOnlyAgent); `:481-482` fp32 cast; `VLAAgent.py:40-48` bf16 default on CUDA; CLIP native fp32. **Reported cells use `hidden_dim=2048`** (see 🟡 below). |
| Head params ≈ 18.2 M (LLaVA) / 4.6 M (CLIP) | ✅ | Dumped from checkpoints: **18,188,632** and **4,557,144**. |
| The trained head is `HeadOnlyAgent` (hidden held constant across backbones), **not** the embed-dim-fixed (768) head in `frozen_vision_baseline.py` | ✅ | Cached path builds `HeadOnlyAgent` (`imitation_learning.py:1154-1164`); checkpoint W1 = `(2048, …)` for **both** backbones, ruling out the 768-wide `FrozenVisionAgent` head. |

🟡 **Caveat (no text change needed):** `HeadOnlyAgent`'s `hidden_dim` *code default*
is `None → feature_dim` (`feature_cache.py:541`), so `H=2048` is a value passed by
the launch config, not a bare default. The four reported configs all store
`hidden_dim:2048`, so the chapter's `H=2048` is correct **for the reported runs**.

### 3.2 Training objective (§Loss, §Optimisation)

| Claim | Verdict | Evidence |
|-------|---------|----------|
| Binary keys via independent BCE-with-logits; camera axes via CE over 11 bins | ✅ | `vla_loss` `imitation_learning.py:314-329`. |
| Total = `BCE + 0.5·(CE_x + CE_y)` (footnote form), chunk axis flattened so every step weighs equally | ✅ | `:330` `cam_ce = cam_ce_weight*(ce_x+ce_y)`, `:247` default `0.5`, `:332` `return bce+cam_ce`; `:283-286` flatten `(B,N,D)→(B·N,D)`. Checkpoint `cam_ce_weight:0.5`. |
| Adam, lr=1e-3, default betas, **no weight decay, no grad clip, no warmup**, batch 256, 10 epochs | ✅ | `:1165` `Adam(model.parameters(), lr=lr)` (wd defaults 0); signature `:931-934` `batch_size=256, epochs=10, lr=1e-3`; no clip/warmup in loop. |
| Constant LR schedule | 🟡 | `lr_schedule="constant"` default (`:934`); scheduler stays `None` unless `"cosine"` (`:1173-1178`). The reported cells have no `cos` tag → constant. ✅ for reported runs. |
| Only the head trained against pre-extracted features; no autocast at head stage; fp16 cache → fp32 | ✅ | Cached path trains `HeadOnlyAgent` only; no `autocast`/`GradScaler` in loop; cache fp16 (`feature_cache.py:303,315-318,338`), `__getitem__` → fp32 (`:482`). |
| Model selection = epoch with highest **validation movement-F1**, mean F1 over back/forward/jump/left/right/sprint (attack excluded, it saturates) | ✅ | `:1243-1246` `movement_idx` = exactly those 6 keys; `:1325-1326` F1; `:1341-1347` keep-best snapshot; comment `:1240-1241` "attack F1 saturates ~.96". Reported configs `keep_best:true`. |

🟡 **Caveat:** `keep_best` *code default* is `False` (`:957`); the reported cells set
it `true` (confirmed in config), and the kept "best" snapshot is a separate
`*_best.pt` alongside a rolling last-epoch file. Accurate for the reported runs.

### 3.3 Two-stage caching (§Two-Stage Training with Cached Features)

| Claim | Verdict | Evidence |
|-------|---------|----------|
| fp16 memmap + JSON sidecar pinning sample order & cache config | ✅ | `feature_cache.py:293-305` JSON `samples` (ordered `[stem,frame_idx]`), `:315-320` `np.memmap(dtype=float16)`. |
| Production **frame stride 4** | 🟡 | Code default is `1` (`:210`, both CLIs); stride 4 is an explicit production choice. The chapter says *"at the production frame-stride of 4"*, i.e. correctly frames it as a choice — and the reported cells are `…_stride4`. ✅ for reported runs. |
| **4 caches** (backbone × language); backbone runs once per cache | 🟡 | Tag is `(backbone, task_filter, lang)`; "4 total" holds for the combined path (`task_filter=None`), which is the chapter's design (all cells on combined data). Single backbone pass confirmed (`:323-339`). ✅ for the reported design. |
| No-prompt cache ≠ empty prompt: LLaVA zeros pooled **text** after pooling (prompt still decoded); CLIP zeros text branch | ✅ | `VLAAgent.encode:136-173`; `frozen_vision_baseline.encode:99-109`. |
| Resumability: `.progress` ~every 100 batches; `r+` memmap resume after JSON-metadata validation | ✅ | `feature_cache.py:211, 268-277, 318, 340-344`. Validates n_samples/feature_dim/backbone/use_language (+ task_filter/llava_id/frame_stride/patch_grid). |
| Cache-build / cost estimates (53×, ~15 ms/sample, ~6 h LLaVA / ~1.5 h CLIP stride-4, head ~0.5–2 h) | ✅ | All present in `ARCHITECTURE.md:660-672` and explicitly labelled **estimates** ("measure with `probe_5090_throughput.py`"). 53× and 15 ms match exactly; cache builds = stride-1 (~25 h / ~6 h) ÷4. |

### 3.4 Data pipeline (§Dataset, §Frame Loading, §Data Split)

| Claim | Verdict | Evidence |
|-------|---------|----------|
| Two tasks: chop_a_tree + collect_dirt | ✅ | `eval_envs.py:189-201`; two `trajectory_task_*` dirs on disk; `ARCHITECTURE.md:287-294`. |
| 20 FPS, 360×640 MP4 + per-frame JSON action log, 1:1 aligned | ✅ | cv2 probe: `640×360 @ 20 fps, 3000 frames` == 3000 action lines; `ARCHITECTURE.md:330`. |
| R/B channels swapped (OpenCV BGR artefact); policy trains on swapped frames; only systematic train/eval difference | ✅ | `eval_envs.py:18-23, 283-322`; `ARCHITECTURE.md:308-313`. *(The "only systematic difference" is an asserted/visually-verified claim, not re-derivable from code — accurate per the repo.)* |
| Provenance: real MineRL engine renders from STEVE-1/VPT via MineDreamer `play` (`vpt/2x.model`, `steve1.weights`) — not diffusion | ✅ | Real info-JSON records `in_model=vpt/2x.model`, `in_weights=steve1/steve1.weights`, MineDreamer play config; `eval_envs.py:3-7`, `ARCHITECTURE.md:300-304`. |
| decord.VideoReader on demand; per-worker LRU cache, default 64 | ✅ | `imitation_learning.py:62-72, 185-206`. |
| Camera magnitudes are mu-law bin centres → lossless 11-bin discretisation | ✅ | Empirically every recorded value is an exact bin centre. |
| Split 80/10/10, seed **42** | ✅ | `val_split=test_split=0.1` (`:936-937`, checkpoints); `seed=42` (`:874, 1038`). |
| Split at **trajectory level** (every frame of a trajectory in one set) | ✅ | `split_indices_by_trajectory` groups by stem, assigns whole stems (`:1036`, `split_indices_by_stem:889-917`); `tests/test_split.py` asserts disjoint, no stem spans subsets. Reported configs `split_by_trajectory:true`. |

🟡 **Caveats (no text change needed):** (a) realised split fractions are
*approximate* at whole-trajectory granularity (not exactly 80/10/10); the chapter's
"$80\%/10\%/10\%$" is the requested target — fine, but you may add "approximately"
if you want to be airtight. (b) Only the **cached** path (`train_cached_head`) has
the trajectory-level option; the end-to-end `train_vla` path is always frame-level.
The reported cells use the cached path, so this is consistent.

### 3.5 Rollout evaluation (§Inference Loop, §Evaluation Metrics)

| Claim | Verdict | Evidence |
|-------|---------|----------|
| Per-episode deque of last **K=8** actions, zero-init at t=0 | 🟡 | `run_rollout.py:387-390` builds the zero-init deque `maxlen=past_action_k`; K read from checkpoint config, =8 for the reported heads (`past_action_k:8`). ✅ for reported runs. |
| Chunked logits (1,N,D); only first step decoded, rest discarded, re-plan each tick | 🟡 | Default path `plan[0]` only (`:428-439`). Opt-in `--execute-steps>1` / `--chunk-ensemble` (default off) change this. ✅ for canonical/default rollout. |
| Stochastic decode: binary τ=1.0, camera τ_cam=2.0 (canonical) | 🟡 | Hardcoded in `eval_suite.sh:96-106` (`--sample --temperature 1.0 --camera-temperature 2.0`). Bare `run_rollout.py` defaults to greedy / `camera-temperature=None`. ✅ via the canonical eval suite. |
| Greedy variant (thr 0.5, argmax) exists, collapses camera onto null bin | ✅ | `action_mapping.py:107-114`; docstring `:45-48` "92%-majority 0° bin always argmaxed". |
| Merge into no-op template (pickItem/swapHands kept); executed action appended to buffer as one-hot | ✅ | `run_rollout.py:333, 457`; `action_mapping.py:118-121`; `:472-473` `action_to_onehot`. |
| R/B swap applied to live observation at rollout | 🟡 | `eval_envs.py:317-324` swap in `ReinhardColorWrapper`; fires for registered envs (the eval-suite Chop/Dirt envs) under `--color-match auto`; does **not** fire for unregistered ids (e.g. the FindCave default of `run_rollout.py`). ✅ for the canonical eval envs. |
| Offline metrics: per-key precision/recall/F1; camera top-1 bin accuracy + MAE (deg) | ✅ | `imitation_learning.py:1422-1433` (`evaluate_cached`) & `:1517-1520` (`evaluate`): `precision_recall_fscore_support`, `camera_{x,y}_bin_accuracy`, `camera_mae_degrees`. Matches the keys in `output/*/metrics.json`. |
| Online primary: per-action firing rate per condition | ✅ | `eval_compare.py:8, 125-130, 250`. |
| Wilson CI + two-proportion z-test + multiple-comparison correction + effect-size threshold | ✅ | `eval_compare.py:58` (Wilson), `:70-82` (z), `:285,310` (Bonferroni), `:85` (Cohen's h); significance requires both `p_adj<0.001` **and** `|h|≥0.5` (`:96-100`). |
| Peak inventory = running max of goal item; reward + success rate reported | ✅ | `eval_logger.py:39, 70-71, 86, 151-173`. |
| Compare trained agents against a **random-policy baseline** | 🟡 | The capability exists (`run_rollout.py` supports a random policy; `eval_compare` compares any rollout tags with the same statistics), but there is **no dedicated random-baseline code path** and no evidence in committed artefacts that a random baseline was logged. If you keep this sentence, ensure a random-policy run exists in the eval set; otherwise soften to "can be compared against a random-policy baseline." |

### 3.6 Reward & environments

| Claim | Verdict | Evidence |
|-------|---------|----------|
| Reward recomputed from inventory deltas | 🟡 | Offline `scripts/inventory_reward.py:21-45` fully recomputes from deltas. Live `InventoryRewardWrapper` returns `engine_reward + delta` (`run_rollout.py:213`) — but engine reward is ~always 0 for these tasks, so the effective value is the delta. Accurate in spirit. |
| Engine handler watches generic `log`; chop yields `oak_log`/`birch_log` → never increments | ✅ | `eval_envs.py:124,194` `RewardForCollectingItems(type="log")`; `run_rollout.py:218-224` per-wood names + "confirmed via probe: never bare `log`". |
| MineRL reset not bit-deterministic (reward unreliability) | ✅* | True per project record (`project_eval_nondeterminism.md`) and implied by seeding plumbed against RNG drift (`run_rollout.py:365-367, 584-587`). *Not asserted verbatim in the named source files; the in-file reward-unreliability cause is the broken Malmo handler. Empirical claim — accurate.* |
| FastAim variant `BreakSpeedMultiplier=5.0`; chop only breaks a log under FastAim | 🔧 ❌ | **5.0 is shared by `…Fast-v0` AND `…FastAim-v0`** (`eval_envs.py:209-215, 228-235`); it is the 5.0 multiplier (both Fast variants) that enables a break, not FastAim specifically. FastAim only *adds* `start_pitch=20.0`. **Reword open-item #7.** |
| Single-biome envs: Chop640=forest, Dirt640=plains; no generic dual world | ✅ | `eval_envs.py:46-47, 189-201`; no mixed-biome spec in `_SPECS`. |

### 3.7 Implementation details

| Claim | Verdict | Evidence |
|-------|---------|----------|
| PyTorch | ✅ | `import torch` throughout. |
| **Python 3.10** | 🔧 ❌ | **Stack requires Python 3.12.** `requirements-blackwell.txt:30` "Python 3.12 (cp312) is required"; `requirements.txt:9-11` "Tested with Python 3.12 … 3.10 not available on LMU CIP". `setup_vastai_5090.sh:50` accepts 3.10–3.12 (3.10 = floor only). **Change "Python 3.10" → "Python 3.12".** |
| RTX 5090, bf16 + SDPA for LLaVA; no FA2 on Blackwell (sm_120) | ✅ | `VLAAgent.py:47,64` (bf16 default, SDPA fallback); `requirements-blackwell.txt:15-19`; `ARCHITECTURE.md:658`. |
| torch 2.7 from cu128 index | ✅ | `requirements-blackwell.txt:11-13`; `setup_vastai_5090.sh:28,76` (`>=2.7,<2.9` from cu128 — floor of 2.7). |
| Dockerised MineRL rollout | ✅ | `Dockerfile`, `docker-compose.yml`, `run_minerl.sh` (project standard). |
| Repo at github.com/DACN-1/MC-AI | ✅ | `git remote -v` → `origin https://github.com/DACN-1/MC-AI.git`; `llava_5090_runbook.md:30`. |

### 3.8 Ablation design

| Claim | Verdict | Evidence |
|-------|---------|----------|
| 2×2 over backbone × language, all four on the **combined** two-task dataset | ✅ | Four `*_combined_*` tsplit cells exist; all configs combined (no `task_filter`/`stem_filter` restriction except `clip_nolang` which stores `stem_filter:null`). |
| Across all cells: temporal recipe (K=8,N=8), head width, optimiser, frame stride, trajectory split, selection held identical; only backbone & language vary | ✅ | The four configs are identical except `feature_dim` (8192 vs 1536) and the language flag — confirmed by direct config diff. |
| LLaVA no-prompt zeros pooled text after pooling (image still prompt-conditioned); CLIP no-prompt removes language entirely | ✅ | `VLAAgent.encode:172-173` vs `frozen_vision_baseline.encode:104-109` (separate encoders). |

---

## 4. Action items (prioritised)

**Must fix before submission:**
1. 🔧 §Implementation Details: **"Python 3.10" → "Python 3.12"** (stack is validated on 3.12; 3.10 is unavailable on LMU CIP).
2. 🔧 Populate `paper/dbstmpl.bib`: add all 10 Method `\cite` keys (`Bain1995AFF, ross2011, kumar2022finetuningdistortpretrainedfeatures, li2023blip2bootstrappinglanguageimagepretraining, liu2024improvedbaselinesvisualinstruction, radford2021clip, baker2022vpt, tavakoli2018actionbranching, lifshitz2023steve1, kanervisto2020`) **plus a `MineDreamer` entry**. The chapter will not compile against the current template bib. *(Also note the committed `paper/*.tex` is still LMU template boilerplate — the Method/Intro drafts are not yet in the repo.)*
3. 🔧 Open-item #7 wording: the log-break-enabling `BreakSpeedMultiplier=5.0` belongs to **both** Fast variants, not FastAim alone; FastAim adds `start_pitch=20.0`.

**Optional tightening (prose is defensible as-is):**
4. §Evaluation Metrics: the "random-policy baseline" comparison has no dedicated code path — either confirm a random baseline was actually logged, or soften the verb to "can be compared against."
5. §Data Split: consider "approximately 80/10/10" since trajectory-level assignment lands near, not exactly, those fractions.
6. The 🟡 items (H=2048, stride 4, K=8, τ_cam=2.0, R/B swap, keep-best) are all *config/flag* values rather than hard code defaults; the chapter describes the **reported runs**, for which every one is confirmed in the checkpoint configs / `eval_suite.sh`. No change needed, but cite the four `*_tsplit*` cells as the source of the reported numbers.

**Nothing material is wrong with the architecture, loss, optimiser, split, or decode descriptions** — they match the source and the four reported checkpoints exactly.

---

## Appendix A — Reported checkpoint configs (ground truth)

Dumped from the `model_best.pt` of each cell.

| Cell | file | feature_dim | hidden_dim | head params | split_by_traj | keep_best | knobs |
|------|------|------------:|-----------:|------------:|:---:|:---:|------|
| **Exp 1** LLaVA + prompt | `llava_combined_lang_stride4_tsplit` | 8192 | 2048 | 18,188,632 | true | true | all off |
| **Exp 3** LLaVA − prompt | `llava_combined_nolang_stride4_tsplit` | 8192 | 2048 | 18,188,632 | true | true | all off |
| **Exp 2** CLIP + prompt | `clip_combined_lang_stride4_tsplit_base` | 1536 | 2048 | 4,557,144 | true | true | all off |
| **Exp 4** CLIP − prompt | `clip_combined_nolang_stride4_tsplit_base` | 1536 | 2048 | 4,557,144 | true | true | all off |

Common to all four: `epochs=10, batch_size=256, lr=0.001, chunk_size=8,
past_action_k=8, past_action_dim=344, val_split=0.1, test_split=0.1,
cam_ce_weight=0.5, weighted_loss=false, history_dropout=0.0,
learnable_bce_temp=false, focal_gamma=0.0, past_action_slot_dropout=0.0,
cam_weighted_loss=false, chop_oversample_weight=1.0, frame_history_k=0,
frame_weight_multiplier=1.0, camera_quantizer={maxval:10, binsize:2, mu:10}`.
Head state-dict = exactly `action_head.{0,2}.{weight,bias}` (a single
`Linear→ReLU→Linear`), confirming a plain 2-layer MLP with no LayerNorm / FiLM.

## Appendix B — Verification provenance

- Direct source reads (this session): `VLAAgent.py`, `constants.py`, `vpt_camera.py`,
  `frozen_vision_baseline.py`, `action_mapping.py`, `imitation_learning.py`
  (`vla_loss`, `train_cached_head`, `evaluate`/`evaluate_cached`, split logic),
  `feature_cache.py` (`HeadOnlyAgent`), `paper/dbstmpl.bib`, git remote.
- Checkpoint configs dumped directly from the four `*_tsplit*` `.pt` files.
- Remaining clusters (caching internals, dataset/provenance, data split tests,
  rollout loop, eval-metric statistics, reward/envs, deps, citations) verified by
  independent agents with an adversarial recheck on every non-confirmed finding;
  the offline-metrics finding was re-resolved against `imitation_learning.evaluate`
  (the online-metrics files do not contain it, but the offline path does).
