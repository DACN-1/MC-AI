# Last Refinements — closing the gap between claims and evidence

**Date:** 2026-06-24
**Source:** strict-examiner colloquium review (another session) + verified data state this session.
**Scope:** the final tests/fixes that move the thesis from "would pass, not top-marked"
to "conclusions supported by the evidence presented." Scholarship and writing are not
the problem; the empirical foundation under the headline claims is. Most items below
are things the text already *flags* but does not *resolve*.

---

## Verdict being answered

Pass, not top mark. The gap an examiner presses in the colloquium is **are the
conclusions actually supported by the evidence**, and on several headline claims:
not quite. Eight attack points, mapped to concrete remedies below. Do **not** break
what is already strong (record, so it survives edits): Wilson/Cohen's-h/effect-size
discipline, the random floor, the offline→online dissociation (the single most
defensible result), the FiLM probe used as self-falsification, and the candour about
confounds.

---

## Current evidence state (verified this session — for whoever executes this)

- **The 2×2 = the four `*_anchor` checkpoints**, each a **single training run, seed 0**.
  Caches on disk: `caches/{llava,clip}_combined_{lang,nolang}_stride4.npy` (LLaVA 24 GB,
  CLIP 4.5 GB ea). Head training is cheap (0.5–2 h/cell on cached features). Eval =
  `eval_suite.sh`, 4 conds × 10 ep; **Mac/MPS is the bottleneck** (slow, thrashes — see
  memory `project_mac_eval_thrash`). Env **is** seed-deterministic (per-episode spread =
  10 distinct world seeds).
- **Conditioning (attack A→C):** CLIP+lang 84.9→37.0 (−48 pp, h=−1.04); CLIP−lang +6.1;
  every LLaVA cell flat (≤±5 pp). FiLM +3.0 pp, fixAB −0.2 pp. Load-bearing claim rests
  on **one cell, one run.**
- **Task (dirt/ep, D):** CLIP+lang 4.3, CLIP−lang 2.3, LLaVA anchors 0.0/0.1, FiLM 4.0,
  random 0. **Chop logs at floor for all four** → quantitative task story is dirt only.
- **Dirt confound (chop-env A vs dirt-env D):** CLIP+lang 0.8→4.3 (+3.5); CLIP−lang
  2.5→2.3 (flat). Difference-in-differences is defensible but **missing the
  dirt-env + empty-prompt cell** to nail it.
- **Offline metrics (1.25 M frames/cell):** four cells a near-tie (movement-F1
  0.368–0.431; CLIP−lang nominally highest) → the dissociation.
- **Probe:** pooled LLaVA text-half is 99.7 % task-decodable **but prompt⊕image
  confounded** (2 prompts ⟂ 2 biomes; text-half image-entangled). FiLM failing ⇒ head is
  not the lever; bottleneck upstream. **Not yet counterfactually tested.**
- **Missing cells:** `llava_nolang_film`, `llava_nolang_fixAB` (never trained);
  `clip_nolang_r3c` (incomplete). Not part of the clean 2×2, but the FiLM control now
  matters (see T7).

---

## Final tests — prioritized

### P0 — required for a strong defence

**T1 — Across-training-seed variance.** *(examiner #1, the hardest hit)*
Each cell is one seed; Chapter 4 reports bootstrap intervals over the 10 eval episodes
only — **no across-training-seed variance anywhere**, and Limitations never mentions it.
Retrain the 4 anchor cells (and FiLM) at 3–5 seeds (head init + data-order shuffle),
report mean ± sd of the conditioning Δ(attack A→C) and dirt/ep across seeds.
Head retrains are cheap; **the cost is the re-evals** (rollouts).
- Minimum viable: 3 seeds for CLIP+lang and one LLaVA cell, to show the −48 pp effect is
  stable and the LLaVA flats are not seed-luck.
- Regardless of whether the runs happen: **add the single-seed caveat to Limitations now.**
- Blocker: rollout volume on Mac → cluster or rented box.

**T2 — Reframe the RQ1 headline.** *(examiner #4; writing-only, free)*
"CLIP beats LLaVA" is undercut by your own FiLM tie (4.0 vs 4.3). The honest claim is
**"LLaVA loses under a minimal concat head and ties CLIP under a FiLM head"** — a
head-readout × backbone *interaction*, not a clean backbone ranking. Ch6 has the caveat;
push it up to the abstract / intro / section titles so "CLIP wins" stops doing
rhetorical work the evidence does not earn.

**T3 — Dirt-env + empty-prompt condition.** *(examiner #5; eval-only)*
A/B/C share the chop env (clean), but D is the only dirt-env condition and carries the
dirt prompt, so the 0.8→4.3 lift conflates **environment with prompt** (plains have more
dirt than forest). Add condition **E = dirt env + empty prompt** for at least CLIP+lang
and CLIP−lang. Closes the difference-in-differences. No training; rollouts only.

### P1 — high value, mostly cheap

**T4 — Counterfactual prompt-swap probe.** *(examiner #3 — tests the mechanism)*
The explanatory core (CLIP gives a separable prompt axis; LLaVA buries it) is **asserted,
not tested**; the cached probe (99.7 %) is confounded. Re-encode a fixed set of ~500
frames under all four prompts through LLaVA **and** CLIP; measure ‖Δ(pooled text-half)‖
and cosine. If LLaVA's barely moves while CLIP's jumps, the mechanism is demonstrated and
the confound quantified. **Needs LLaVA forwards (GPU) — on the table.** Cache-incompatible
(must re-encode). Pre-build the script so it fires on a box.

**T5 — Demonstrator anchor + chop-direction explanation.** *(examiner #6 and #7; cheap analysis)*
Every number is relative to your own cells. Compute the STEVE-1/VPT demonstrator's
per-task **attack rate** and **dirt collection** from the training actions (subset the
stems — the full `all_actions.json` is ~900 MB and will choke a naive parse). This gives
(a) an absolute reference for "4.3 dirt / 1000 steps," and (b) a test of the
counterintuitive chop result: does the *demonstrator* also attack less under "chop a
tree"? If the model is matching demo statistics, the attack-suppression is imitation, not
malfunction. Then write the paragraph that currently does not exist explaining why the
chop prompt suppresses the one action chopping requires.

**T6 — Chop aiming / supervision check.** *(examiner #2; cheap, demo-side)*
Run `scripts/analyze_chop_aiming.py`. It tests whether the demonstrations contain a
*learnable* "aim at the trunk" signal (directional pre-attack camera burst vs
sub-quantisation wander). If the supervision is dead, the chop failure is a data problem,
not an architecture one — which sharpens point #2 and gives chop a quantitative line
instead of only the qualitative "targets the tree, misses the block."

### P2 — rounds it out (LLaVA compute; explicitly on the table)

**T7 — Train + eval `llava_nolang_film` (the missing FiLM control).** *(examiner #4; user-endorsed)*
FiLM is currently a one-off probe with no matched no-prompt control, so it sits outside
the controlled 2×2. Training the no-language FiLM cell promotes it to a symmetric result
and answers the obvious question: **does the FiLM dirt gain (0→4.0) survive without
language?** Expected yes (all LLaVA conditioning is flat, so the gain is head-readout, not
language) — but the control *proves* the FiLM story is about the head, not a smuggled
language effect. Cache exists (`llava_combined_nolang_stride4`), so this is **head-only
training + rollouts**, no re-cache. Do `llava_nolang_fixAB` too for full symmetry if cheap.

*(T8 — LoRA / last-token-pooling — scrapped: it was the only item needing a Stage-1
recache. The frozen-vs-fine-tuned trade-off is therefore not measured; handle examiner
#8 by reframing instead, see below.)*

### Writing-only reframes — do regardless (no compute)

- **#1:** single-seed caveat in Limitations (until T1 lands).
- **#2:** state plainly that the quantitative task story is **one working task** (dirt);
  chop contributes firing-rate conditioning + the qualitative finding only.
- **#6:** the chop attack-suppression paragraph (pair with T5).
- **#8:** contribution = comparative + dissociation, not architectural novelty; cite PR2L
  as the precedent. Since T8 is scrapped the frozen-vs-fine-tuned trade-off is **not
  measured**, so do not advertise it: demote JARVIS-VLA from "comparator" to related work
  and scope every claim to the frozen regime. This is now the *whole* answer to #8.

---

## Execution notes

- **No-GPU, do-now:** T2, T5, T6, all writing reframes. None need a backbone or rollouts.
- **Rollout-bound (cluster / rented box):** T1, T3 — many eval suites; Mac MPS will not
  carry the volume.
- **LLaVA-forward-bound (GPU):** T4, T7 — `requirements-blackwell.txt` + a 5090 per the
  runbook, or any CUDA box. T7 is the cheaper of the two (head-only, cache exists).
- **No Stage-1 recache anywhere.** With T8 scrapped, every remaining test reuses the
  on-disk caches or needs none; T4 needs only ~2 k ad-hoc encodings, not a rebuild.
- **Minimal defensible package:** T1 (≥3 seeds, 2 cells) + T2 + T3 + T5 + the writing
  reframes. That directly answers the three hardest punches (#1 single seed, #4 FiLM
  framing, #5 prompt/biome confound) and supplies the missing anchor (#7). T4 and T7 are
  what would lift it from "defensible" toward a top mark.

## Where the marks are won or lost

"CLIP wins" survives **only** as "wins under the minimal head" — protect that qualifier
everywhere. The dissociation (RQ3) is already top-mark-grade; keep it central. The
conditioning claim (RQ2) is one cell from one run until T1, and its mechanism is asserted
until T4 — those two tests are the difference between a result and an anecdote.
