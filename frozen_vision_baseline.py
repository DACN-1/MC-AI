"""Vision-only baseline: frozen CLIP encoder + trainable MLP action head."""

from torch import nn
from transformers import CLIPModel, CLIPProcessor

from constants import NUM_OUTPUT_LOGITS as DEFAULT_OUTPUT_DIM


class FrozenVisionAgent(nn.Module):
    """CLIP-based vision-only baseline with the same forward interface as VLAAgent.

    The texts argument is accepted but ignored — only image features are used.
    Output layout matches VLAAgent — see constants.NUM_OUTPUT_LOGITS.
    """

    def __init__(
        self,
        output_dim: int = DEFAULT_OUTPUT_DIM,
        backbone: str = "openai/clip-vit-large-patch14",
    ):
        super().__init__()
        self.processor = CLIPProcessor.from_pretrained(backbone)
        self.clip = CLIPModel.from_pretrained(backbone)
        for p in self.clip.parameters():
            p.requires_grad_(False)

        embed_dim = self.clip.config.projection_dim
        self.action_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, output_dim),
        )

    def forward(self, images, texts):
        """Return action logits of shape (B, NUM_ACTIONS). texts is ignored."""
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(
            next(self.clip.parameters()).device
        )
        image_features = self.clip.get_image_features(**inputs)  # (B, embed_dim)
        return self.action_head(image_features)


if __name__ == "__main__":
    agent = FrozenVisionAgent()
    print("ok")
