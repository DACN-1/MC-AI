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
        cuda_available = th.cuda.is_available()
        try:
            self.llava = LlavaForConditionalGeneration.from_pretrained(
                backbone,
                torch_dtype=th.float16,
                attn_implementation="flash_attention_2",
                device_map="cuda" if cuda_available else None,
            )
        except (ImportError, ValueError):
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

        # Action head stays in fp32 — keeps optimizer steps numerically stable
        # regardless of where the (frozen) backbone runs. We cast pooled features
        # to fp32 in forward().
        self.action_head = nn.Sequential(
            nn.Linear(hidden + past_action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim * chunk_size),
        )

    @property
    def feature_dim(self) -> int:
        """Width of the pooled LLaVA feature consumed by the head (pre past-action)."""
        return self.action_head[0].in_features - self.past_action_dim

    def encode(self, images, texts) -> th.Tensor:
        """Run the frozen backbone and return pooled (B, feature_dim) features.

        Exposed for the feature-caching pipeline so a separate script can
        precompute LLaVA embeddings once and reuse them across many head-only
        training runs.
        """
        effective_texts = texts if self.use_language else [""] * len(texts)
        prompts = [t if "<image>" in t else f"<image>\n{t}" for t in effective_texts]

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

        # LLaMA has no CLS token — mean-pool the last hidden state across the full
        # (image + text) sequence so the head sees both modalities.
        pooled = captured["h"].mean(dim=1)
        head_dtype = self.action_head[0].weight.dtype
        return pooled.to(head_dtype)

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
