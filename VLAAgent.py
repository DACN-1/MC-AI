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
        self.llava = LlavaForConditionalGeneration.from_pretrained(
            backbone, torch_dtype=th.float16
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
        effective_texts = texts if self.use_language else [""] * len(texts)
        prompts = [t if "<image>" in t else f"<image>\n{t}" for t in effective_texts]

        inputs = self.processor(
            images=images,
            text=prompts,
            return_tensors="pt",
            padding=True,
        ).to(self.llava.device)

        out = self.llava(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pixel_values=inputs.pixel_values,
            output_hidden_states=True,
        )

        # LLaMA has no CLS token — mean-pool the last hidden state across the full
        # (image + text) sequence so the head sees both modalities.
        pooled = out.hidden_states[-1].mean(dim=1)
        head_dtype = self.action_head[0].weight.dtype
        pooled = pooled.to(head_dtype)

        if self.past_action_dim > 0:
            if past_actions is None:
                raise ValueError(
                    "past_actions is required when past_action_dim > 0; "
                    f"got None (past_action_dim={self.past_action_dim})"
                )
            past = past_actions.to(pooled.device).to(head_dtype)
            pooled = th.cat([pooled, past], dim=-1)

        flat = self.action_head(pooled)  # (B, output_dim * chunk_size)
        return flat.view(flat.size(0), self.chunk_size, self.output_dim)
