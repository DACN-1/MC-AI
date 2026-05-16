"""CLIP-based baseline with the same forward interface as VLAAgent.

When `use_language=True` both the image and text encoders contribute features
to the action head (concatenated). When `use_language=False` the text branch
is zeroed — image features are concatenated with a zero-vector of matching
shape so the head input dimension stays constant. This keeps the
`use_language` axis a runtime flag rather than a separate architecture.
"""

import torch as th
from torch import nn
from transformers import CLIPModel, CLIPProcessor

from constants import NUM_OUTPUT_LOGITS as DEFAULT_OUTPUT_DIM


class FrozenVisionAgent(nn.Module):
    """Frozen CLIP encoder(s) + trainable MLP action head.

    Output layout matches VLAAgent: (B, chunk_size, output_dim). See
    constants.NUM_OUTPUT_LOGITS.
    """

    def __init__(
        self,
        output_dim: int = DEFAULT_OUTPUT_DIM,
        backbone: str = "openai/clip-vit-large-patch14",
        use_language: bool = True,
        past_action_dim: int = 0,
        chunk_size: int = 1,
    ):
        super().__init__()
        self.processor = CLIPProcessor.from_pretrained(backbone)
        self.clip = CLIPModel.from_pretrained(backbone)
        for p in self.clip.parameters():
            p.requires_grad_(False)

        embed_dim = self.clip.config.projection_dim
        self.use_language = use_language
        self.past_action_dim = past_action_dim
        self.chunk_size = chunk_size
        self.output_dim = output_dim
        # Always-on concat: image features + (text features or zeros). Lets
        # the same head shape serve both use_language settings, so a single
        # checkpoint can be loaded for either ablation cell.
        self.feature_dim = 2 * embed_dim

        self.action_head = nn.Sequential(
            nn.Linear(self.feature_dim + past_action_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, output_dim * chunk_size),
        )

    def encode(self, images, texts) -> th.Tensor:
        """Run the frozen CLIP encoders and return the concatenated feature
        vector (B, feature_dim). Useful for feature caching."""
        device = next(self.clip.parameters()).device
        img_inputs = self.processor(images=images, return_tensors="pt", padding=True).to(device)
        image_features = self.clip.get_image_features(**img_inputs)  # (B, embed_dim)
        if self.use_language:
            txt_inputs = self.processor(
                text=texts, return_tensors="pt", padding=True, truncation=True
            ).to(device)
            text_features = self.clip.get_text_features(**txt_inputs)  # (B, embed_dim)
        else:
            text_features = th.zeros_like(image_features)
        return th.cat([image_features, text_features], dim=-1)

    def forward(self, images, texts, past_actions=None):
        """Return logits of shape (B, chunk_size, output_dim)."""
        features = self.encode(images, texts)
        if self.past_action_dim > 0:
            if past_actions is None:
                raise ValueError(
                    "past_actions is required when past_action_dim > 0; "
                    f"got None (past_action_dim={self.past_action_dim})"
                )
            past = past_actions.to(features.device).to(features.dtype)
            features = th.cat([features, past], dim=-1)
        flat = self.action_head(features)
        return flat.view(flat.size(0), self.chunk_size, self.output_dim)


if __name__ == "__main__":
    agent = FrozenVisionAgent()
    print("ok")
