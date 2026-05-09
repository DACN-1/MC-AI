"""Frozen LLaVA backbone with a trainable action head.

The head emits `output_dim` raw logits per sample. The interpretation of those
logits is owned by `constants.py` and `action_mapping.py` — see `NUM_OUTPUT_LOGITS`
there for the binary / camera-bin layout.
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

        # Action head stays in fp32 — keeps optimizer steps numerically stable
        # regardless of where the (frozen) backbone runs. We cast pooled features
        # to fp32 in forward().
        self.action_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )
        self.use_language = use_language

    def forward(self, images, texts):
        """Run a forward pass and return logits of shape (B, output_dim).

        When use_language=False the text input is replaced with empty strings so
        only visual features contribute to the action head.
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
        pooled = pooled.to(self.action_head[0].weight.dtype)
        return self.action_head(pooled)
