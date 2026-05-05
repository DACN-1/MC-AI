import torch as th
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import (
    LlavaProcessor,
    LlavaForConditionalGeneration
)



class VLAAgent(nn.Module):
    def __init__(self, NUM_ACTIONS: int, backbone: str = "llava-hf/llava-1.5-7b-hf", use_language: bool = True):
        super().__init__()
        self.processor = LlavaProcessor.from_pretrained(backbone)
        self.llava     = LlavaForConditionalGeneration.from_pretrained(
            backbone, torch_dtype=th.float16
        )
        # Freeze backbone 
        for p in self.llava.parameters():
            p.requires_grad_(False)


        if hasattr(self.llava.config, "hidden_size"):
            hidden = self.llava.config.hidden_size
        else:  
            hidden = self.llava.config.text_config.hidden_size
        self.action_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, NUM_ACTIONS),
        )
        
        self.use_language = use_language

        if self.llava.device.type == "cpu":
            self.action_head = self.action_head.float()  # float32 on CPU
        else:
            self.action_head = self.action_head.half()   # float16 on GPU

    def forward(self, images, texts):
        """Run a forward pass and return action logits of shape (B, NUM_ACTIONS).

        When use_language=False the text input is replaced with empty strings so
        only visual features contribute to the action head.
        """
        effective_texts = texts if self.use_language else [""] * len(texts)
        # Ensure each prompt includes the <image> placeholder token required by LLaVA
        texts = [t if "<image>" in t else f"<image>\n{t}" for t in effective_texts]
        inputs = self.processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding=True,
        ).to(self.llava.device) # type: ignore

        out = self.llava(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pixel_values=inputs.pixel_values,
            output_hidden_states=True,
        )

        pooled = out.hidden_states[-1][:, 0]  # CLS token
        # Cast pooled features to the same dtype as the head weights
        pooled = pooled.to(self.action_head[0].weight.dtype)
        return self.action_head(pooled)

    # Dummy forward pass for testing
    # def forward(self, images, texts):
    #     batch_size = len(images)
    #     dummy_logits = th.rand(batch_size, NUM_ACTIONS,
    #                 device=self.action_head[0].weight.device,
    #                 requires_grad=True)
    #     return dummy_logits
