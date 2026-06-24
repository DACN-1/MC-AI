# Why CLIP conditions on language and LLaVA doesn't

**The question.** In the frozen-backbone + MLP-head setup, the CLIP-lang head
shifts its behavior when the prompt changes (attack-firing drops from 85% to 37%
on "chop a tree"), while every LLaVA-lang head — anchor, A+B, and FiLM — keeps a
flat, prompt-invariant policy. Both use the same head, same loss, same decode, same
cache pipeline, and LLaVA's vision tower *is* a CLIP ViT-L/14. So why the gap?

**The one-line answer.** CLIP hands the head a **contrastively-trained, image-aligned
text embedding** — exactly the format a frozen linear-ish probe needs to switch
behavior on. LLaVA hands the head a **mean-pooled next-token-prediction hidden
state**, which carries the instruction in a form that isn't linearly separable into
an actionable "task direction" — and the causal `<image>\ntext` ordering makes that
weak channel the *only* one carrying the prompt. It's a **readout problem**, not a
capability problem, and no head-side architecture can manufacture a signal the
pooled feature doesn't expose.

---

## 1. The evidence

Attack-firing % across prompt conditions (A `""` · B `"Play Minecraft."` ·
C `"chop a tree"`), from `scripts/eval_compare.py` over `evaluations/paper`:

| cell | A | B | C | **A→C Δ** | conditions? |
|------|--:|--:|--:|--:|---|
| `clip_lang_anchor` | 84.9 | 76.7 | 37.0 | **−47.9 pp** | ✅ strong |
| `clip_nolang_anchor` (control) | 57.8 | 57.8 | 64.0 | +6.1 pp | flat (as expected) |
| `llava_lang_anchor` | 34.9 | 32.6 | 29.8 | −5.1 pp | ❌ |
| `llava_lang_fixAB` (LayerNorm + img-dropout, concat) | 74.5 | 75.4 | 74.3 | −0.2 pp | ❌ |
| `llava_lang_film` (A+B+C, text modulates image) | 86.9 | 88.6 | 89.9 | +3.0 pp | ❌ |

CLIP-lang moves attack by ~48 pp on the on-task prompt; its nolang control stays
flat — that's the conditioning signal. **No LLaVA head moves beyond the ~5 pp noise
floor**, including FiLM, which *structurally forces* the text to participate.

This is not for lack of a working policy: once normalized, LLaVA is a strong
*unconditional* agent — `llava_lang_film` reaches **dirt 4.0/ep @ 90%**, essentially
matching CLIP's best (4.3/ep), with val F1 0.452. It uses the visual features well;
it just won't condition on the prompt.

---

## 2. What is NOT the cause (controlled away)

Both paths are identical except the backbone, so these are ruled out:

- **Backbone capacity / vision quality.** LLaVA-1.5-7B's vision tower *is*
  `openai/clip-vit-large-patch14-336` — the same family as the CLIP baseline
  (`openai/clip-vit-large-patch14`). LLaVA's features sit *downstream* of CLIP
  (CLIP embedding → 32 LLaMA layers), so they contain *more* visual information,
  not less. The gap is not "LLaVA sees worse."
- **Head / loss / decode / sampler.** Identical across cells (the 2×2 symmetry rule).
- **The degenerate-policy problem.** That was a *normalization* issue (the raw 8192-d
  feature's per-dim scale imbalance), fixed post-cache by LayerNorm (`feature_norm`):
  inventory-mashing 60% → ~1%, dirt 0.0 → 2.5–4.0/ep. **Fixing it did not bring back
  conditioning** — so degeneracy and conditioning are *separate* root causes.
- **The "image shortcut" alone.** A concat head can read task-identity off the
  always-present image and zero-weight text — that explains `fixAB`. But **FiLM
  removed that escape hatch** (see §4) and *still* failed. So the shortcut is an
  aggravator, not the whole story.

---

## 3. The mechanism: what each backbone hands the head as "text"

The head sees `[image_features ‖ text_features]`. The difference is entirely in how
`text_features` is produced.

### CLIP — a contrastive, image-aligned, probe-ready text embedding
`frozen_vision_baseline.encode` uses `self.clip.get_text_features(...)` — the output
of CLIP's **contrastive pre-training**, whose entire objective was to make the text
vector for "a photo of a tree" land *close to* images of trees in a shared embedding
space. Consequences:

- The text vector is **discriminative and image-aligned by construction**: changing
  the prompt moves it in a direction that is semantically meaningful *relative to the
  visual features the head also sees*.
- It varies **strongly and consistently** with the prompt (distinct captions → well-
  separated, L2-normalized embeddings). So "text points toward tree → lower attack"
  is a stable, high-magnitude, linearly-readable direction. A tiny head can learn it.

### LLaVA — a mean-pooled next-token-prediction hidden state
`VLAAgent.encode` produces `text_pool` = the **mean over the LLaMA final-layer hidden
states at the prompt-token positions** (captured via the RMSNorm hook). Note this is a
**split** pool — text tokens are pooled *separately* from the 576 image tokens (the
Phase C fix to the old joint-mean design, which existed precisely to stop the handful
of text tokens being swamped by the image tokens). So text is **not diluted**; the
problem is **not pooling arithmetic**. Different prompts genuinely yield different
`text_pool` vectors and the head sees the difference — it just can't act on it. The
real issue is **representation geometry**:

- Those hidden states are optimized for exactly one thing: **predicting the next
  token**. `text_pool` is therefore a *language-modeling* representation in the LM's
  generative manifold, not a discriminative, aligned task embedding. The prompt-
  distinguishing direction is present but **not aligned** to the visual/action space
  and **not linearly separable** into a behavior-relevant signal a frozen probe can
  read ("text says tree ⇒ lower attack").
- Contrastive alignment is the thing that *makes* CLIP's text vector probe-ready;
  LLaVA's decoder states were never trained to be a fixed-vector summary of "what to
  do," so cleanly pooling them still yields a vector whose task content isn't usable
  by the head. Clean pooling doesn't fix the *meaning* of the vector.
- (Minor residual: the mean over prompt tokens also discards word order and includes
  `\n`/format tokens — negligible for a 2–3-word prompt, not the cause. And a decoder
  concentrates its "what to do given this image+instruction" computation at the **last
  prompt-token** state, not the average — which is why *last-token* is the candidate
  readout, not because the mean diluted anything.)

This is the crux: **same instruction, but CLIP encodes it as an aligned global
embedding and LLaVA as a pooled generative state.** Only the former is usable by a
small frozen head to switch behavior.

---

## 4. The causal-ordering aggravator

LLaVA's prompt layout is `<image>\n{prompt}` (`VLAAgent.encode`): image tokens come
**first**, text **after**, under a **causal** decoder. So image tokens cannot attend
forward to the text, which means **`image_pool` is prompt-invariant** — it is bit-for-
bit the same whatever the prompt. Therefore *all* conditioning must flow through
`text_pool`, the weak channel from §3.

(In CLIP there's no such asymmetry — the text embedding is a strong standalone
channel. And if LLaVA put text *before* image, image tokens could absorb the
instruction — but that changes `encode()` and requires a re-cache, which is out of
scope here.)

---

## 5. Why the experiments prove it's the *signal*, not the architecture

We escalated the head's ability to use text and it still failed — which rules out
"the head just ignored an available signal":

- **`fixAB` (concat + image-dropout):** gave the head the text vector and a training
  pressure (drop the image 15–25% of the time) to use it. Result: flat. A concat head
  *can* zero-weight text, so this alone isn't conclusive.
- **`film` (FiLM):** text **structurally modulates** the image via per-channel scale
  γ and shift β; the MLP consumes *only* the modulated image, so text **cannot be
  zeroed out**. Trained with image-dropout so that when the image is dropped the
  output collapses to **β(text) alone** — forcing β to carry the task. Result: still
  flat (A→C +3 pp), *while reaching the best objective F1 and dirt collection of any
  LLaVA head.*

That is the decisive result. FiLM had every architectural incentive and the capacity
to use `text_pool`, demonstrably exploited the features well overall — and the prompt
*still* didn't move behavior. The conditioning signal needed to switch behavior is
**not linearly accessible in `text_pool`**. No head-side trick (normalization, gating,
modulation, dropout) can recover a signal the pooled feature doesn't expose.

---

## 6. Conclusion — it's a readout problem, deferred

- **CLIP conditions** because its text encoder was contrastively trained to emit a
  discriminative, image-aligned embedding — the exact format a frozen probe needs.
- **LLaVA doesn't** because the head only ever sees the prompt as a mean-pooled
  next-token hidden state, whose task content isn't linearly separable; and causal
  ordering makes that the *only* prompt-bearing channel.
- This is a **readout** limitation, not a capability one — LLaVA plainly "understands"
  the instruction generatively. Fixing it means changing how the text is read out of
  the backbone, which requires re-encoding the cache:
  - **Last-token pooling** — use the final-position hidden state (where a decoder
    concentrates its decision / instruction-following) instead of the mean. Cheapest
    feature-quality fix; still frozen + cache + head, but a one-pass re-encode
    (~6 h/tag stride-4 on a CUDA box).
  - **Text-before-image ordering** — let image tokens attend to the instruction.
  - **LoRA fine-tune** — let the backbone reshape its states to be action/instruction-
    relevant; most powerful, but abandons caching (end-to-end).

All three require a backbone re-encode (or end-to-end training), which is why they're
deferred. Within the existing mean-pool cache, conditioning is **not recoverable** —
the post-cache levers (normalization, dropout, FiLM) are exhausted, and they delivered
a strong *unconditional* LLaVA policy but no language conditioning.

---

*Verified against `frozen_vision_baseline.encode` (CLIP `get_text_features` →
`[image ‖ text]`, dim 1536), `VLAAgent.encode` (split mean-pool of the LLaMA last
hidden state → `[image_pool ‖ text_pool]`, dim 8192; prompt `<image>\n{prompt}`,
causal), and the rollout firing rates in `evaluations/paper`. See `llava_fix.md`
(fix A+B) and `llava_fix_v2.md` (FiLM) for the experiments behind §5.*
