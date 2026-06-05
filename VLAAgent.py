"""Frozen LLaVA backbone with a trainable action head.

The head emits `output_dim` raw logits per chunk step. With `chunk_size>1`
forward returns a 3-D tensor (B, chunk_size, output_dim) so the loss / mapping
code can treat single-step and multi-step heads uniformly. The interpretation
of `output_dim` is owned by `constants.py` and `action_mapping.py` — see
`NUM_OUTPUT_LOGITS` there for the binary / camera-bin layout.
"""

import torch as th
from torch import nn
from transformers import (
    LlavaProcessor,
    LlavaForConditionalGeneration,
)


class VLAAgent(nn.Module):
    def __init__(
        self,
        output_dim: int,
        backbone: str = "llava-hf/llava-1.5-7b-hf",
        use_language: bool = True,
        past_action_dim: int = 0,
        chunk_size: int = 1,
        head_hidden_dim: int | None = None,
    ):
        super().__init__()
        self.processor = LlavaProcessor.from_pretrained(backbone)
        # FlashAttention-2 cuts both prefill time and the quadratic attention
        # activation tensor. Combined with the hidden-state hook in encode(),
        # this is what lets the LLaVA cache build run at batch=64+ on a 24 GB
        # A5000. Fall back to sdpa on platforms without the flash-attn wheel
        # (e.g. the macOS dev machine) — the encode output is numerically the
        # same modulo standard fp16 reduction-order differences.
        # FA2 must be initialised on the GPU directly. `from_pretrained` first
        # places the model on CPU and then HF silently falls back to eager
        # attention if it sees CPU placement at init time — so a later
        # `.to("cuda")` does not re-enable FA2. Pass `device_map="cuda"` here
        # to skip the CPU staging step. On macOS (no CUDA) fall back to the
        # default CPU placement + sdpa attention.
        # On Blackwell (RTX 5090, sm_120) there is no flash-attn wheel — stock
        # flash-attn has no sm_120 kernel — so we run on sdpa there by design.
        # With flash_attn absent, HF raises ImportError (CPU-only → ValueError),
        # both caught below. If a dirty base image ships a stock flash-attn whose
        # kernel rejects sm_120, the failure surfaces as a RuntimeError at load —
        # caught too, so the sdpa retry still fires.
        cuda_available = th.cuda.is_available()
        try:
            self.llava = LlavaForConditionalGeneration.from_pretrained(
                backbone,
                torch_dtype=th.float16,
                attn_implementation="flash_attention_2",
                device_map="cuda" if cuda_available else None,
            )
        except (ImportError, ValueError, RuntimeError):
            self.llava = LlavaForConditionalGeneration.from_pretrained(
                backbone,
                torch_dtype=th.float16,
                device_map="cuda" if cuda_available else None,
            )
        # Freeze backbone
        for p in self.llava.parameters():
            p.requires_grad_(False)

        if hasattr(self.llava.config, "hidden_size"):
            hidden = self.llava.config.hidden_size
        else:
            hidden = self.llava.config.text_config.hidden_size

        self.use_language = use_language
        self.past_action_dim = past_action_dim
        self.chunk_size = chunk_size
        self.output_dim = output_dim

        # Split image/text pooling — see encode() docstring. Pooled features are
        # [image_pool || text_pool] of dim 2*hidden = 8192 for LLaVA-1.5-7B. Mirrors
        # frozen_vision_baseline.CLIP's [image_proj || text_proj] convention so the
        # 2×2 ablation's "language vs no-language" cells compare apples to apples.
        self._llava_hidden = hidden
        self._feature_dim = 2 * hidden
        self._head_hidden_dim = (
            head_hidden_dim if head_hidden_dim is not None else self._feature_dim
        )

        # Action head stays in fp32 — keeps optimizer steps numerically stable
        # regardless of where the (frozen) backbone runs. We cast pooled features
        # to fp32 in forward().
        self.action_head = nn.Sequential(
            nn.Linear(self._feature_dim + past_action_dim, self._head_hidden_dim),
            nn.ReLU(),
            nn.Linear(self._head_hidden_dim, output_dim * chunk_size),
        )

    @property
    def feature_dim(self) -> int:
        """Width of the pooled LLaVA feature consumed by the head (pre past-action).

        After the split-image/text pooling fix (`docs/alignment_handoff.md`
        Phase C step 2) this is 2 * llava_hidden_size = 8192 for LLaVA-1.5-7B.
        """
        return self._feature_dim

    def encode(self, images, texts) -> th.Tensor:
        """Run the frozen backbone and return pooled (B, 2*hidden) features.

        The pooled feature is ``[image_pool || text_pool]`` — image tokens and
        text tokens of the post-norm hidden state are mean-pooled separately,
        then concatenated. Mirrors CLIP's ``[image_proj || text_proj]``
        convention so the 2x2 ablation's no-language cells (Exp 3, Exp 4)
        compare apples to apples.

        Before this change, the entire (image + text) sequence was mean-pooled
        into one 4096-d vector. Because the LlavaProcessor emits a single
        ``<image>`` placeholder per image which the model expands to
        ``image_seq_length`` (=576) tokens at forward time, the joint mean
        was dominated by image tokens (~576) versus a handful of text tokens
        (~5), washing out the language signal entirely. See
        ``docs/alignment_handoff.md`` for the full diagnostic.

        ``use_language=False`` zeros ``text_pool`` after pooling — matching
        ``frozen_vision_baseline.py``'s zero-text-features behavior. We do NOT
        replace the input text with ``""``; the encoder still sees the prompt
        (image tokens attend to text tokens at every layer — that's a LLaVA
        architectural property that can't be closed by code), but the head
        input is image-only.
        """
        # Always pass the actual text. use_language is enforced at pooling time,
        # not at processor time — see docstring.
        prompts = [t if "<image>" in t else f"<image>\n{t}" for t in texts]

        inputs = self.processor(
            images=images,
            text=prompts,
            return_tensors="pt",
            padding=True,
        ).to(self.llava.device)

        # Capture the final-layer hidden state via a forward hook on the LLaMA
        # RMSNorm. Asking the model for `output_hidden_states=True` stashes all
        # 32 layers (~10 GB at batch=64 on LLaVA-7B) and forces the cache build
        # down to batch=16 on the A5000. The hook output is the same tensor
        # that `hidden_states[-1]` would have returned.
        lang_model = self.llava.language_model
        norm = getattr(lang_model, "model", lang_model).norm
        captured: dict[str, th.Tensor] = {}

        def _grab(_mod, _args, output):
            captured["h"] = output

        handle = norm.register_forward_hook(_grab)
        try:
            self.llava(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pixel_values=inputs.pixel_values,
            )
        finally:
            handle.remove()

        image_pool, text_pool = self._split_pool(
            captured["h"], inputs.input_ids, inputs.attention_mask
        )
        if not self.use_language:
            text_pool = th.zeros_like(text_pool)
        pooled = th.cat([image_pool, text_pool], dim=-1)
        head_dtype = self.action_head[0].weight.dtype
        return pooled.to(head_dtype)

    def _split_pool(self, h: th.Tensor, input_ids: th.Tensor, attention_mask: th.Tensor):
        """Mean-pool image-role and text-role tokens of the post-norm hidden state.

        ``h`` is the (B, T_out, hidden) tensor captured by the RMSNorm hook.
        ``input_ids`` (B, T_in) carries one image-token placeholder per sample
        which the model expands to ``N`` image tokens during forward, so
        ``T_out = T_in - 1 + N`` (single image per sample). The image tokens
        occupy positions ``[img_pos, img_pos + N)`` of the post-expansion
        sequence; everything else is text or pad.

        Returns ``(image_pool, text_pool)`` both shape ``(B, hidden)``.
        """
        B, T_in = input_ids.shape
        T_out = h.shape[1]
        if T_out < T_in:
            raise ValueError(
                f"hidden state seq_len {T_out} < input_ids seq_len {T_in} — "
                "expected the model to expand <image> at forward time"
            )
        n_per_img = T_out - T_in + 1  # assumes exactly one image placeholder per sample

        image_token_id = self.llava.config.image_token_index
        img_match = input_ids == image_token_id
        if not img_match.any(dim=1).all():
            raise ValueError(
                f"some samples lack image_token_id={image_token_id} in input_ids"
            )
        img_pos = img_match.int().argmax(dim=1)  # (B,) first occurrence per sample

        device = h.device
        positions = th.arange(T_out, device=device).unsqueeze(0)  # (1, T_out)
        img_start = img_pos.unsqueeze(1)                          # (B, 1)
        img_end = img_start + n_per_img                           # (B, 1)
        is_image = (positions >= img_start) & (positions < img_end)  # (B, T_out)

        # Map post-expansion positions back to pre-expansion positions so we can
        # look up the original attention mask (pads etc.).
        pre_pos = th.where(positions >= img_end, positions - (n_per_img - 1), positions)
        pre_pos = pre_pos.clamp(min=0, max=T_in - 1).long()
        attn_at_pre = th.gather(attention_mask, 1, pre_pos)  # (B, T_out)
        is_text = (attn_at_pre > 0) & (~is_image)

        def _masked_mean(values: th.Tensor, mask: th.Tensor) -> th.Tensor:
            m = mask.unsqueeze(-1).to(values.dtype)
            s = (values * m).sum(dim=1)
            c = m.sum(dim=1).clamp(min=1)
            return s / c

        return _masked_mean(h, is_image), _masked_mean(h, is_text)

    def forward(self, images, texts, past_actions=None):
        """Run a forward pass.

        Returns logits of shape (B, chunk_size, output_dim). For the original
        single-step head (`chunk_size=1`) this is (B, 1, output_dim); callers
        that want the legacy 2-D shape can index `[:, 0, :]`.

        Args:
            images: list[PIL.Image] of length B
            texts:  list[str] of length B (replaced with "" if use_language=False)
            past_actions: optional (B, past_action_dim) float tensor. Required
                iff past_action_dim > 0.
        """
        pooled = self.encode(images, texts)

        if self.past_action_dim > 0:
            if past_actions is None:
                raise ValueError(
                    "past_actions is required when past_action_dim > 0; "
                    f"got None (past_action_dim={self.past_action_dim})"
                )
            past = past_actions.to(pooled.device).to(pooled.dtype)
            pooled = th.cat([pooled, past], dim=-1)

        flat = self.action_head(pooled)  # (B, output_dim * chunk_size)
        return flat.view(flat.size(0), self.chunk_size, self.output_dim)
