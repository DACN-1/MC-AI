"""Verify the encode() forward-hook path matches the old output_hidden_states path.

Opt-in integration test: loading LLaVA-1.5-7B in fp16 needs ~14 GB of RAM and
a populated HF cache, so this is skipped by default. Run with

    R1VA_RUN_LLAVA_INTEGRATION=1 python -m unittest tests.test_encode_equivalence

on a machine where the model is already cached (cluster or any box that has
previously run training / rollout).
"""

import os
import sys
import unittest
from pathlib import Path

import numpy as np
import torch as th
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@unittest.skipUnless(
    os.environ.get("R1VA_RUN_LLAVA_INTEGRATION") == "1",
    "Set R1VA_RUN_LLAVA_INTEGRATION=1 to run the LLaVA encode-equivalence test",
)
class EncodeForwardHookEquivalenceTests(unittest.TestCase):
    """The forward hook on language_model[.model].norm must capture the same
    tensor that the old `output_hidden_states=True; hidden_states[-1]` path
    produced. Same model, same input, same RNG → bitwise-equal pooled output."""

    @classmethod
    def setUpClass(cls):
        from VLAAgent import VLAAgent

        from constants import NUM_OUTPUT_LOGITS, PAST_ACTION_DIM

        cls.agent = VLAAgent(
            output_dim=NUM_OUTPUT_LOGITS,
            backbone="llava-hf/llava-1.5-7b-hf",
            use_language=True,
            past_action_dim=PAST_ACTION_DIM,
            chunk_size=1,
        )
        cls.agent.eval()

        rng = np.random.default_rng(0)
        cls.img = Image.fromarray(
            rng.integers(0, 256, (336, 336, 3), dtype=np.uint8)
        )
        cls.text = "chop a tree"

    def test_hook_matches_output_hidden_states(self):
        agent = self.agent

        with th.no_grad():
            new_pooled = agent.encode([self.img], [self.text])

            prompts = [f"<image>\n{self.text}"]
            inputs = agent.processor(
                images=[self.img],
                text=prompts,
                return_tensors="pt",
                padding=True,
            ).to(agent.llava.device)
            out = agent.llava(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pixel_values=inputs.pixel_values,
                output_hidden_states=True,
            )
            ref_pooled = out.hidden_states[-1].mean(dim=1).to(
                agent.action_head[0].weight.dtype
            )

        self.assertEqual(new_pooled.shape, ref_pooled.shape)
        self.assertTrue(
            th.equal(new_pooled, ref_pooled),
            f"hook path diverged from output_hidden_states path: "
            f"max abs diff = {(new_pooled - ref_pooled).abs().max().item():.3e}",
        )


if __name__ == "__main__":
    unittest.main()
