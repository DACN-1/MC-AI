"""Vision-only baseline: frozen CLIP encoder + trainable MLP action head."""

import torch as th
from torch import nn
from transformers import CLIPModel, CLIPProcessor

from constants import NUM_OUTPUT_LOGITS as DEFAULT_OUTPUT_DIM


class FrozenVisionAgent(nn.Module):
    """CLIP-based vision-only baseline with the same forward interface as VLAAgent.

    The texts argument is accepted but ignored — only image features are used.
    Output layout matches VLAAgent: (B, chunk_size, output_dim). See
    constants.NUM_OUTPUT_LOGITS.
    """

    def __init__(
        self,
        output_dim: int = DEFAULT_OUTPUT_DIM,
        backbone: str = "openai/clip-vit-large-patch14",
        past_action_dim: int = 0,
        chunk_size: int = 1,
    ):
        super().__init__()
        self.processor = CLIPProcessor.from_pretrained(backbone)
        self.clip = CLIPModel.from_pretrained(backbone)
        for p in self.clip.parameters():
            p.requires_grad_(False)

        embed_dim = self.clip.config.projection_dim
        self.past_action_dim = past_action_dim
        self.chunk_size = chunk_size
        self.output_dim = output_dim

        self.action_head = nn.Sequential(
            nn.Linear(embed_dim + past_action_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, output_dim * chunk_size),
        )

    def forward(self, images, texts, past_actions=None):
        """Return logits of shape (B, chunk_size, output_dim). texts is ignored."""
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(
            next(self.clip.parameters()).device
        )
        image_features = self.clip.get_image_features(**inputs)  # (B, embed_dim)
        if self.past_action_dim > 0:
            if past_actions is None:
                raise ValueError(
                    "past_actions is required when past_action_dim > 0; "
                    f"got None (past_action_dim={self.past_action_dim})"
                )
            past = past_actions.to(image_features.device).to(image_features.dtype)
            image_features = th.cat([image_features, past], dim=-1)
        flat = self.action_head(image_features)
        return flat.view(flat.size(0), self.chunk_size, self.output_dim)


if __name__ == "__main__":
    agent = FrozenVisionAgent()
    print("ok")
