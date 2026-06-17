# CLIP head recipes — catalog + objective task-performance screen (2026-06-17)

## Why this doc

The 2026-06-08/09 recipe leaderboard was ranked on a **broken reward** and later
shown to be noise. This screen re-evaluates every CLIP head recipe on an
**objective, reward-independent signal — items actually collected** — to decide
which recipe is worth the (paid) LLaVA rollout. See also `docs/trials.md`,
`docs/HANDOFF.md`.

## The metric (and why reward was broken)

MineRL's `RewardForCollectingItems` never fires for chopping because it watches
the item `"log"`, but chopping yields per-wood names (`oak_log`/`birch_log`/…) —
the `"log"` slot never increments. The `InventoryRewardWrapper` (meant to bypass
the Malmo handler) had the **same bug**. Both were watching a nonexistent item.

Fix + instrument (committed):
- `InventoryRewardWrapper` now rewards real wood-log items (`run_rollout.py`).
- `eval_logger` records per-step `obs["inventory"]` → per-episode **peak count of
  every collected item by real name** → `run_summary.json` fields
  `mean_peak_inventory`, `collect_rate`, `peak_inventory_per_episode`.
- The env already exposes the full inventory observation (no env change needed);
  the agent never reads it (head consumes only `obs["pov"]`), so behaviour is
  unchanged — it's a pure logging channel.

Rank with: `python scripts/rank_item_collection.py evaluations/test/recipe_sweep_v2`
(sums all `*_log` for chop, `dirt`/`coarse_dirt`/`grass_block` for dirt).

## Recipe catalog (deltas from baseline)

**Baseline** `clip_combined_lang_stride4`: `hidden_dim=2048, epochs=10, lr=1e-3,
past_action_k=8, chunk_size=8`, frame-level split, no keep-best, all shaping knobs
off. Knob values extracted from each checkpoint's stored `config`.

| family | recipe | delta from baseline |
|---|---|---|
| **slot-dropout** | `slot30` / `slot50` | `past_action_slot_dropout` 0.3 / 0.5 |
| **chop-oversample** | `chop3` | `chop_oversample_weight=3` |
| | `slot20_chop3` / `slot30_chop3` / `slot40_chop3` | slot-dropout 0.2/0.3/0.4 + oversample 3 |
| | `slot30_chop5` | slot 0.3 + oversample 5 |
| | `slot50_chop3` | slot 0.5 + oversample 3 (trained 2026-06-17, frame-level) |
| **train length / lr** | `ep20` | epochs 20 |
| | `slot30_chop3_ep20` | slot30_chop3 + epochs 20 |
| | `sc_ep14` / `sc_best20` | slot30_chop3 + epochs 14 / 20+keep-best |
| | `sc_lr5e4_ep20` | slot30_chop3 + lr 5e-4 + epochs 20 |
| **capacity / horizon** | `hd4096` | hidden_dim 4096 |
| | `sc_chunk4` | slot30_chop3 + chunk_size 4 |
| **loss shaping** | `focal2` | focal_gamma 2.0 |
| | `wloss` / `slot30_chop3_wloss` | weighted_loss (task-active frame weighting) |
| | `cwloss` / `slot30_chop3_cwloss` | cam_weighted_loss |
| | `slot30_chop3_camCE2x` | cam_ce_weight 2.0 |
| | `chop_onset5/15/40` | camera_onset_weight 5/15/40, window 8 |
| **history reg** | `sc_hd02` | slot30_chop3 + history_dropout 0.2 |
| **frame-weighted rounds** | `r1` | frame_weight 5× min_run 60 + history_dropout 0.5 + learnable_bce_temp |
| | `r2` | r1 but min_run 180 |
| | `r3a_noTemp` | frame_weight 5× min_run 180 + history_dropout 0.5, no temp |
| | `r3b_lowDrop` | frame_weight 5× min_run 180 + history_dropout 0.2 + temp |
| | `r3c_minrun300` | frame_weight 5× min_run 300 + history_dropout 0.5 + temp |
| **frame history** | `fhist4` / `fhist8` | frame_history_k 4 / 8 |
| **honest-split anchors** | `tsplit_base`, `slot30_chop3_tsplit`, `nolang_tsplit_base` | trajectory split + keep-best |
| **controls** | `slot30_chop3_replica` | slot30_chop3 re-trained (reproducibility) |
| | `pooled_chop_ctl` | baseline-equivalent (patch-grid control) |

Knob glossary: `slot_dropout`=zero random past-action slots; `chop_oversample`=upsample
sustained-attack frames; `frame_weight ×N min_run M`=WeightedRandomSampler upweights
frames in attack runs ≥M ticks; `history_dropout`=zero the whole past-action vector;
`weighted_loss`/`cam_weighted_loss`/`cam_ce_weight`=loss reweighting; `camera_onset_weight`=
upweight camera-CE before attack onset; `frame_history_k`=concat K prior frames.

## Screen results (n=6, C_chop_task + D_dirt_task, sampled decode)

`evaluations/test/recipe_sweep_v2/` — 19 cells. **logs/ep = mean peak logs (chop);
dirt/ep = mean peak dirt.**

| recipe | chop logs/ep | dirt/ep | dirt collect-rate |
|---|---|---|---|
| slot30_chop3 | 0 | 4.83 | 100% |
| r3c_minrun300 | 0 | 4.50 | 100% |
| slot50 | 0.17 (1/6) | 4.17 | 100% |
| hd4096 | 0 | 3.33 | 83% |
| slot30_chop5 | 0 | 3.33 | 83% |
| r3b_lowDrop | 0 | 2.67 | 83% |
| r3a_noTemp | 0 | 2.50 | 83% |
| r1 | 0 | 2.17 | 83% |
| slot30 | 0 | 1.83 | 100% |
| ep20 | 0 | 1.50 | 83% |
| slot20_chop3 | 0 | 0.50 | 33% |
| r2 / slot50_chop3 | 0 | 0.17 | 17% |
| baseline / focal2 / slot40_chop3 / wloss / slot30_chop3_ep20 / slot30_chop3_wloss | 0 | 0.00 | 0% |

## Findings

1. **Chop is floor for every recipe.** All ≈ 0 completed logs; `slot50`'s single
   log (1/6) is noise. No recipe completes a chop — the videos show *attempts*
   (attacking trunks), not completed breaks. Chop is unsolved by any head recipe.
2. **Dirt shows a real "digs vs doesn't" split.** A group digs (the r-family +
   `slot30`/`slot30_chop3`/`slot50`/`slot30_chop5`/`hd4096`) while `baseline`,
   `focal2`, `wloss`, and the `_ep20`/`_wloss` compounds collect ~0. **baseline
   collects zero dirt**, so this is a genuine recipe effect — overturning the old
   "recipes are just noise" verdict (that was the broken-reward artifact).
3. **n=6 cannot rank the diggers.** The slot×chop3 gradient is non-monotonic
   (`slot20_chop3` 0.5 / `slot30_chop3` 4.83 / `slot40_chop3` 0.0) — a 4.8→0 swing
   from a 0.1 knob change is variance, not signal. Dirt is high-variance per
   episode (spawn-dependent). Fine ranking needs high-n.
4. **`slot50_chop3` showed no synergy.** Combining slot-dropout 0.5 + oversample 3
   landed near floor (0.17 dirt) — if anything `chop3` hurt the strong `slot50`
   (4.17 → 0.17). Inconclusive at n=6, but no support for the hypothesis.
5. **Two knobs actively kill dirt:** adding `ep20` or `wloss` to `slot30_chop3`
   drops it 4.83 → 0. `focal2` collapses the policy (consistent with prior work).

## In progress / next

- **Full eval** (10 ep × 4 conditions, sampled decode) on the top-4 dirt diggers
  (`slot30_chop3`, `r3c_minrun300`, `slot50`, `hd4096`) + `baseline` control →
  `evaluations/test/recipe_full/`. Gives a trustworthy dirt ranking + the A/B/C
  conditioning firing rates.

## Implication for the LLaVA spend

- **Dirt is the only task that works.** Roll out the dirt-winning recipe on LLaVA;
  `slot30_chop3` is the leading candidate **and already has a trained LLaVA head**
  (runbook cells 4/5) — pending the full-eval confirmation.
- **Chop: spend nothing** — floor everywhere; no LLaVA chop story.
- The knob-free LLaVA/CLIP `tsplit` anchors remain the conditioning 2×2 spine,
  independent of recipe choice.
