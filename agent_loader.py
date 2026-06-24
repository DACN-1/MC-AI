"""Checkpoint loading for VLA rollouts — shared by run_rollout.py (in the
MineRL Docker container) and inference_server.py (native macOS, MPS).

This module deliberately imports no `gym`/`minerl` so it can run natively on
Apple Silicon, where MineRL is unavailable but Metal (MPS) acceleration is. It
knows how to build two checkpoint flavours:

  * end-to-end `VLAAgent` checkpoints (carry a `llava_model` key), and
  * cached-head checkpoints from `train_cached_head` (carry a `cache_tag`);
    these only store the MLP head, so the frozen backbone is rebuilt and run
    live per frame via `_CachedHeadRolloutAgent`.

Both flavours expose the same call signature the rollout loop expects:
`agent(images, texts, past_actions) -> (B, chunk_size, NUM_OUTPUT_LOGITS)`.
"""

import torch

from VLAAgent import VLAAgent
from constants import NUM_OUTPUT_LOGITS, PAST_ACTION_DIM


class _CachedHeadRolloutAgent(torch.nn.Module):
    """Live counterpart of a `train_cached_head` checkpoint.

    Cached-head checkpoints only store the MLP head (`HeadOnlyAgent`), which was
    trained against pre-pooled features. To roll out we rebuild the matching
    frozen backbone and run its `encode()` per frame, then the head — giving the
    same `forward(images, texts, past_actions) -> (B, chunk, NUM_OUTPUT_LOGITS)`
    interface the rollout loop expects from `VLAAgent`.
    """

    def __init__(self, backbone_module: torch.nn.Module, head: torch.nn.Module):
        super().__init__()
        self.backbone = backbone_module  # used only for .encode()
        self.head = head

    def forward(self, images, texts, past_actions=None):
        features = self.backbone.encode(images, texts)
        return self.head(features, past_actions)


def _load_cached_head_agent(ckpt: dict, cfg: dict, device: str):
    """Build a backbone + HeadOnlyAgent from a `train_cached_head` checkpoint."""
    from feature_cache import HeadOnlyAgent

    cache_tag = ckpt.get("cache_tag", "")
    feature_dim = cfg["feature_dim"]
    past_action_dim = cfg.get("past_action_dim", 0)
    chunk_size = cfg.get("chunk_size", 1)
    hidden_dim = cfg.get("hidden_dim")
    if cfg.get("frame_history_k", 0) > 0:
        raise NotImplementedError(
            "This head was trained with frame_history_k="
            f"{cfg['frame_history_k']} (visual history windows); rollout "
            "serving needs a per-episode frame-feature buffer in the "
            "inference server, which is not implemented yet. Evaluate via "
            "training metrics, or implement the buffer first."
        )
    # cache tags are "<backbone>_<task>_<lang|nolang>[_strideN]" — the language
    # flag is a token, NOT a suffix (a `_stride4` suffix used to silently break
    # `endswith("nolang")`, loading nolang ckpts with use_language=True).
    # Match the token directly. Backbone is still the prefix; fall back on
    # feature_dim < 2048 to catch CLIP-only (=1536) without confusing LLaVA-pre-
    # Phase-C (=4096) and LLaVA-post-Phase-C (=8192) caches.
    is_clip = cache_tag.startswith("clip") or feature_dim < 2048
    tag_tokens = cache_tag.split("_")
    use_language = "nolang" not in tag_tokens
    # Spatial caches carry a `patch<G>` token (e.g. clip_..._stride4_patch4);
    # the rollout backbone must reproduce the same GxG patch-grid encode the
    # head was trained on.
    patch_grid = next(
        (int(t[5:]) for t in tag_tokens if t.startswith("patch") and t[5:].isdigit()),
        0,
    )

    if is_clip:
        from frozen_vision_baseline import FrozenVisionAgent

        backbone_module = FrozenVisionAgent(
            output_dim=NUM_OUTPUT_LOGITS,
            use_language=use_language,
            past_action_dim=0,
            chunk_size=1,
            patch_grid=patch_grid,
        )
    else:
        backbone_module = VLAAgent(
            output_dim=NUM_OUTPUT_LOGITS,
            backbone=ckpt.get("llava_model", "llava-hf/llava-1.5-7b-hf"),
            use_language=use_language,
            past_action_dim=0,
            chunk_size=1,
        )

    head = HeadOnlyAgent(
        feature_dim=feature_dim,
        output_dim=NUM_OUTPUT_LOGITS,
        past_action_dim=past_action_dim,
        chunk_size=chunk_size,
        hidden_dim=hidden_dim,
        learnable_bce_temp=cfg.get("learnable_bce_temp", False),
        # fix A (LayerNorm) and fix C (FiLM) are active inference modules — must be
        # rebuilt so the state_dict keys match. fix B (image_dropout) is train-only,
        # so it stays off here (eval() makes it a no-op regardless).
        feature_norm=cfg.get("feature_norm", False),
        film=cfg.get("film", False),
    )
    head.load_state_dict(ckpt["state_dict"])

    agent = _CachedHeadRolloutAgent(backbone_module, head).to(device)
    agent.eval()
    return agent, {
        "past_action_k": past_action_dim // PAST_ACTION_DIM,
        "chunk_size": chunk_size,
        "use_language": use_language,
    }


def load_agent(model_path: str, device: str) -> tuple[torch.nn.Module, dict]:
    """Load a VLA checkpoint produced by imitation_learning.train_vla or
    imitation_learning.train_cached_head.

    Returns (agent, config) — config carries past_action_k / chunk_size /
    use_language so the rollout loop can match training-time conditioning.
    End-to-end checkpoints carry a `llava_model` key and store a `VLAAgent`
    head; cached-head checkpoints carry a `cache_tag` and store a
    `HeadOnlyAgent`, requiring the backbone to be rebuilt for live encoding.
    """
    ckpt = torch.load(model_path, map_location="cpu")
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError(
            f"Checkpoint at {model_path} is not in the expected format "
            "(missing 'state_dict' key). Expected a dict produced by train_vla()."
        )
    cfg = ckpt.get("config", {})

    # Cached-head checkpoints have no end-to-end backbone id but do carry a
    # cache_tag (and a feature_dim in config). Route them to the live-encode path.
    if "llava_model" not in ckpt and ("cache_tag" in ckpt or "feature_dim" in cfg):
        return _load_cached_head_agent(ckpt, cfg, device)

    backbone = ckpt.get("llava_model", "llava-hf/llava-1.5-7b-hf")
    past_action_k = cfg.get("past_action_k", 0)
    chunk_size = cfg.get("chunk_size", 1)
    use_language = cfg.get("use_language", True)
    agent = VLAAgent(
        output_dim=NUM_OUTPUT_LOGITS,
        backbone=backbone,
        use_language=use_language,
        past_action_dim=past_action_k * PAST_ACTION_DIM,
        chunk_size=chunk_size,
    )
    agent.action_head.load_state_dict(ckpt["state_dict"])
    agent = agent.to(device)
    agent.eval()
    return agent, {
        "past_action_k": past_action_k,
        "chunk_size": chunk_size,
        "use_language": use_language,
    }
