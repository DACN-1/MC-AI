"""Tests for ``VLAAgent.encode`` and its split image/text pooling.

Two layers:

1. **Synthetic-input unit tests** (`SplitPoolSyntheticTests`) — exercise
   ``_split_pool`` directly with handcrafted tensors. Verifies the post-
   expansion role-mask logic without loading LLaVA.

2. **LLaVA integration tests** (`SplitPoolLLaVAIntegrationTests`) — opt-in,
   load the real LLaVA-1.5-7B in fp16 (~14 GB RAM + populated HF cache).
   Skipped by default. Run with::

       R1VA_RUN_LLAVA_INTEGRATION=1 python -m unittest tests.test_encode_equivalence

   Verifies that the forward hook on the post-norm layer captures the same
   tensor as ``output_hidden_states=True; hidden_states[-1]``, that the
   image-token expansion shape matches the model's ``image_seq_length``,
   and that ``encode()`` returns split-pooled features of the expected
   dim with ``use_language=False`` properly zeroing the text half.
"""

import os
import sys
import unittest
from pathlib import Path

import numpy as np
import torch as th

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Synthetic unit tests — no LLaVA, no network, fast.
# ---------------------------------------------------------------------------


class SplitPoolSyntheticTests(unittest.TestCase):
    """Drive ``VLAAgent._split_pool`` with handcrafted tensors that mimic the
    post-expansion hidden state the LLaVA forward hook captures."""

    @staticmethod
    def _build_fake_agent(hidden=8, image_token_id=32000):
        """Construct a VLAAgent-shaped object with just enough attributes for
        ``_split_pool`` to work, bypassing the (heavyweight) ``__init__``."""
        from VLAAgent import VLAAgent

        agent = VLAAgent.__new__(VLAAgent)
        # _split_pool only needs self.llava.config.image_token_index
        class _Cfg:
            pass
        class _Llava:
            pass
        agent.llava = _Llava()
        agent.llava.config = _Cfg()
        agent.llava.config.image_token_index = image_token_id
        agent._llava_hidden = hidden
        agent._feature_dim = 2 * hidden
        return agent

    def test_split_pool_basic(self):
        """One image-placeholder per sample, no padding, no edge cases."""
        agent = self._build_fake_agent(hidden=4, image_token_id=999)
        # input_ids: [BOS=1, <image>=999, text_tok=10, text_tok=11]   (B=1, T_in=4)
        input_ids = th.tensor([[1, 999, 10, 11]])
        attention_mask = th.ones_like(input_ids)
        # Model expands <image> to N=3 image tokens; T_out = 4 - 1 + 3 = 6
        # Position layout in post-expansion sequence:
        #   0: BOS, 1-3: image, 4-5: text
        h = th.tensor([[
            [10.0, 10.0, 10.0, 10.0],   # 0  BOS  -> text role
            [ 1.0,  1.0,  1.0,  1.0],   # 1  image
            [ 2.0,  2.0,  2.0,  2.0],   # 2  image
            [ 3.0,  3.0,  3.0,  3.0],   # 3  image
            [ 5.0,  5.0,  5.0,  5.0],   # 4  text
            [ 7.0,  7.0,  7.0,  7.0],   # 5  text
        ]])
        image_pool, text_pool = agent._split_pool(h, input_ids, attention_mask)
        # image_pool = mean([1,2,3]) = 2.0
        # text_pool = mean([10, 5, 7]) = 22/3
        self.assertEqual(image_pool.shape, (1, 4))
        self.assertEqual(text_pool.shape, (1, 4))
        th.testing.assert_close(image_pool, th.full((1, 4), 2.0))
        th.testing.assert_close(text_pool, th.full((1, 4), 22.0 / 3.0))

    def test_split_pool_respects_padding(self):
        """Pad tokens (attention_mask=0) must NOT contribute to text_pool."""
        agent = self._build_fake_agent(hidden=2, image_token_id=999)
        # B=1, T_in=4: BOS, <image>, text, pad
        input_ids = th.tensor([[1, 999, 10, 0]])
        attention_mask = th.tensor([[1, 1, 1, 0]])   # pad at last position
        # T_out = 4 - 1 + 3 = 6.  Positions: 0 BOS, 1-3 image, 4 text, 5 pad.
        h = th.tensor([[
            [10.0, 10.0],  # BOS (text)
            [ 1.0,  1.0],  # image
            [ 2.0,  2.0],  # image
            [ 3.0,  3.0],  # image
            [ 5.0,  5.0],  # text
            [99.0, 99.0],  # PAD — must be excluded
        ]])
        image_pool, text_pool = agent._split_pool(h, input_ids, attention_mask)
        # image_pool = mean([1,2,3]) = 2.0; text_pool = mean([10, 5]) = 7.5
        th.testing.assert_close(image_pool, th.full((1, 2), 2.0))
        th.testing.assert_close(text_pool, th.full((1, 2), 7.5))

    def test_split_pool_image_at_different_positions(self):
        """Image placeholder need not be at a fixed offset across the batch."""
        agent = self._build_fake_agent(hidden=2, image_token_id=999)
        # B=2: sample 0 has <image> at pos 1, sample 1 has it at pos 2
        input_ids = th.tensor([
            [1, 999, 10,  0],
            [1,   5, 999, 10],
        ])
        attention_mask = th.tensor([
            [1, 1, 1, 0],   # last is pad
            [1, 1, 1, 1],
        ])
        # T_out = 4 - 1 + 3 = 6. n_per_img = 3.
        # Sample 0: positions [0,1,2,3,4,5]; image positions [1,2,3]; text positions [0, 4].  Pos 5 is pad.
        # Sample 1: positions [0,1,2,3,4,5]; image positions [2,3,4]; text positions [0, 1, 5].
        h = th.tensor([
            [[10.0, 10.0],  # s0 pos 0 — text (BOS)
             [ 1.0,  1.0],  # s0 pos 1 — image
             [ 2.0,  2.0],  # s0 pos 2 — image
             [ 3.0,  3.0],  # s0 pos 3 — image
             [ 5.0,  5.0],  # s0 pos 4 — text
             [99.0, 99.0]], # s0 pos 5 — pad
            [[20.0, 20.0],  # s1 pos 0 — text (BOS)
             [30.0, 30.0],  # s1 pos 1 — text
             [ 4.0,  4.0],  # s1 pos 2 — image
             [ 5.0,  5.0],  # s1 pos 3 — image
             [ 6.0,  6.0],  # s1 pos 4 — image
             [40.0, 40.0]], # s1 pos 5 — text
        ])
        image_pool, text_pool = agent._split_pool(h, input_ids, attention_mask)
        # s0: image mean = (1+2+3)/3 = 2.0, text mean = (10 + 5)/2 = 7.5
        # s1: image mean = (4+5+6)/3 = 5.0, text mean = (20 + 30 + 40)/3 = 30.0
        th.testing.assert_close(image_pool, th.tensor([[2.0, 2.0], [5.0, 5.0]]))
        th.testing.assert_close(text_pool, th.tensor([[7.5, 7.5], [30.0, 30.0]]))


# ---------------------------------------------------------------------------
# LLaVA integration tests — opt-in, heavy.
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    os.environ.get("R1VA_RUN_LLAVA_INTEGRATION") == "1",
    "Set R1VA_RUN_LLAVA_INTEGRATION=1 to run the LLaVA integration tests",
)
class SplitPoolLLaVAIntegrationTests(unittest.TestCase):
    """End-to-end with the real LLaVA-1.5-7B weights."""

    @classmethod
    def setUpClass(cls):
        from PIL import Image
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
        """RMSNorm hook captures the same tensor as output_hidden_states[-1].

        The encode() path uses a forward hook to grab the post-norm hidden state
        without storing all 32 layers; this verifies the hook output matches the
        cheap-but-memory-heavy ``output_hidden_states=True`` reference path.
        """
        agent = self.agent
        with th.no_grad():
            # Run the new encode and grab the hook's tensor for inspection.
            captured = {}
            lang_model = agent.llava.language_model
            norm = getattr(lang_model, "model", lang_model).norm

            def grab(_m, _a, out):
                captured["h"] = out

            handle = norm.register_forward_hook(grab)
            prompts = [f"<image>\n{self.text}"]
            inputs = agent.processor(
                images=[self.img],
                text=prompts,
                return_tensors="pt",
                padding=True,
            ).to(agent.llava.device)
            try:
                ref_out = agent.llava(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    pixel_values=inputs.pixel_values,
                    output_hidden_states=True,
                )
            finally:
                handle.remove()

            hook_h = captured["h"]
            ref_h = ref_out.hidden_states[-1]

        self.assertEqual(hook_h.shape, ref_h.shape)
        self.assertTrue(
            th.equal(hook_h, ref_h),
            f"hook diverged from output_hidden_states[-1]: "
            f"max abs diff = {(hook_h - ref_h).abs().max().item():.3e}",
        )

    def test_image_token_expansion_count_matches_seq_length(self):
        """Verify the post-expansion image-token span is exactly image_seq_length.

        LlavaProcessor emits ONE ``image_token_index`` placeholder per image in
        ``input_ids``. The model expands it to ``config.image_seq_length`` tokens
        during forward (576 for LLaVA-1.5-7B). If transformers ever changes that
        contract — or pre-expands at processor time instead — ``_split_pool``'s
        ``n_per_img = T_out - T_in + 1`` assumption breaks. Fail loud here.
        """
        agent = self.agent
        with th.no_grad():
            captured = {}
            lang_model = agent.llava.language_model
            norm = getattr(lang_model, "model", lang_model).norm
            handle = norm.register_forward_hook(lambda _m, _a, o: captured.update(h=o))
            prompts = [f"<image>\n{self.text}"]
            inputs = agent.processor(
                images=[self.img], text=prompts, return_tensors="pt", padding=True,
            ).to(agent.llava.device)
            try:
                agent.llava(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    pixel_values=inputs.pixel_values,
                )
            finally:
                handle.remove()

        T_in = inputs.input_ids.shape[1]
        T_out = captured["h"].shape[1]
        # n_per_img derived from the shape delta; should equal the model's
        # advertised image_seq_length.
        n_derived = T_out - T_in + 1
        expected = getattr(agent.llava.config, "image_seq_length", None)
        if expected is None:
            # Older configs put it on vision_config.
            expected = (
                agent.llava.config.vision_config.image_size
                // agent.llava.config.vision_config.patch_size
            ) ** 2
        self.assertEqual(
            n_derived,
            expected,
            f"image-token expansion mismatch: T_out - T_in + 1 = {n_derived}, "
            f"but model.config.image_seq_length = {expected}. _split_pool's "
            f"role mask will be wrong if these don't match.",
        )
        # And input_ids must contain exactly one image placeholder per sample.
        img_id = agent.llava.config.image_token_index
        self.assertEqual(
            int((inputs.input_ids == img_id).sum().item()),
            inputs.input_ids.shape[0],
        )

    def test_encode_shape_and_no_language_zeros_text_pool(self):
        """``encode()`` returns ``(B, 2 * hidden)``; ``use_language=False`` zeros the text half."""
        agent = self.agent
        with th.no_grad():
            pooled = agent.encode([self.img], [self.text])
        self.assertEqual(pooled.shape, (1, agent.feature_dim))
        self.assertEqual(agent.feature_dim, 2 * agent._llava_hidden)

        agent.use_language = False
        try:
            with th.no_grad():
                pooled_nolang = agent.encode([self.img], [self.text])
        finally:
            agent.use_language = True
        H = agent._llava_hidden
        # Text half (the second H dims) must be exactly zero.
        text_half = pooled_nolang[:, H:]
        self.assertTrue(
            th.equal(text_half, th.zeros_like(text_half)),
            f"use_language=False didn't zero text_pool: max|·| = {text_half.abs().max().item():.3e}",
        )
        # Image half is non-trivial.
        img_half = pooled_nolang[:, :H]
        self.assertGreater(img_half.abs().max().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
