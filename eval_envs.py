"""In-distribution evaluation environments for the BC ablation cells.

The training video is **real MineRL POV** (MineDreamer ``play`` output: a STEVE-1/VPT
agent playing the real engine, info.json telemetry confirms real gameplay) at 640×360,
saved through a cv2 BGR/RGB bug that swapped red/blue. It is NOT diffusion-generated —
the frames are crisp real-engine textures (verified 2026-06-04). The stock MineRL
rollout envs still mismatch it in a few ways this module fixes for closed-loop rollout:

  1. Two custom env ids — ``MineRLChopATree640-v0`` and
     ``MineRLCollectDirt640-v0`` — built on :class:`HumanControlEnvSpec` so they
     render at **640×360** (vs ``MineRLTreechop-v0``'s 64×64) with the **full
     near-human action space** the models were trained on (ESC / inventory /
     hotbar.1-9 / drop, identical to BASALT). NOT ``SimpleHumanEmbodimentEnvSpec``
     (Treechop's base), whose smaller action space would not match the model's 23
     canonical keys. The spawn biome is fixed via the world-generator JSON
     (``fixedBiome``), the agent starts **bare-handed** (matching the generation
     agent's empty inventory), time is pinned to midday, and a per-item reward is set.

  2. :class:`ReinhardColorWrapper` — the essential correction is the **R↔B channel
     swap** that undoes the save bug (verified: a/b chroma then match the training
     distribution within noise). A per-task Reinhard LAB transfer is available but
     **off by default** (brightness is handled at the env-config source instead).

Importing this module registers the env ids (after ``import minerl``).
"""

import json
import os
from typing import List

import cv2
import gym
import numpy as np

from minerl.herobraine.env_specs.human_controls import HumanControlEnvSpec
from minerl.herobraine.hero.handler import Handler
from minerl.herobraine.hero.mc import MS_PER_STEP
import minerl.herobraine.hero.handlers as handlers


# --------------------------------------------------------------------------- #
# Custom env specs                                                            #
# --------------------------------------------------------------------------- #

# Minecraft 1.16.5 numeric biome ids used in the world-generator JSON.
BIOME_FOREST = 4
BIOME_PLAINS = 1

# World-generator options copied from MineRL's treechop_specs.py (caves / lakes /
# structures disabled for a clean surface), with ``fixedBiome`` parameterised so
# each task spawns in the right biome.
_GEN_OPTS_TEMPLATE = (
    '{{"coordinateScale":684.412,"heightScale":684.412,"lowerLimitScale":512.0,'
    '"upperLimitScale":512.0,"depthNoiseScaleX":200.0,"depthNoiseScaleZ":200.0,'
    '"depthNoiseScaleExponent":0.5,"mainNoiseScaleX":80.0,"mainNoiseScaleY":160.0,'
    '"mainNoiseScaleZ":80.0,"baseSize":8.5,"stretchY":12.0,"biomeDepthWeight":1.0,'
    '"biomeDepthOffset":0.0,"biomeScaleWeight":1.0,"biomeScaleOffset":0.0,'
    '"seaLevel":1,"useCaves":false,"useDungeons":false,"dungeonChance":8,'
    '"useStrongholds":false,"useVillages":false,"useMineShafts":false,'
    '"useTemples":false,"useMonuments":false,"useMansions":false,"useRavines":false,'
    '"useWaterLakes":false,"waterLakeChance":4,"useLavaLakes":false,'
    '"lavaLakeChance":80,"useLavaOceans":false,"fixedBiome":{biome},"biomeSize":4,'
    '"riverSize":1,"dirtSize":33,"dirtCount":10,"dirtMinHeight":0,"dirtMaxHeight":256,'
    '"gravelSize":33,"gravelCount":8,"gravelMinHeight":0,"gravelMaxHeight":256,'
    '"graniteSize":33,"graniteCount":10,"graniteMinHeight":0,"graniteMaxHeight":80,'
    '"dioriteSize":33,"dioriteCount":10,"dioriteMinHeight":0,"dioriteMaxHeight":80,'
    '"andesiteSize":33,"andesiteCount":10,"andesiteMinHeight":0,"andesiteMaxHeight":80,'
    '"coalSize":17,"coalCount":20,"coalMinHeight":0,"coalMaxHeight":128,"ironSize":9,'
    '"ironCount":20,"ironMinHeight":0,"ironMaxHeight":64,"goldSize":9,"goldCount":2,'
    '"goldMinHeight":0,"goldMaxHeight":32,"redstoneSize":8,"redstoneCount":8,'
    '"redstoneMinHeight":0,"redstoneMaxHeight":16,"diamondSize":8,"diamondCount":1,'
    '"diamondMinHeight":0,"diamondMaxHeight":16,"lapisSize":7,"lapisCount":1,'
    '"lapisCenterHeight":16,"lapisSpread":16}}'
)

# Generous episode cap — run_rollout's --max-steps is the real limiter; we don't
# want the env to time out before that.
_MAX_EPISODE_STEPS = 8000


class _Combined640EnvSpec(HumanControlEnvSpec):
    """640×360, full human action space, fixed biome, single-item reward.

    Mirrors MineRL's ``Treechop`` task methods but on the ``HumanControlEnvSpec``
    base so the action/observation space matches the BASALT-style models.

    ``break_speed_multiplier`` is the Malmo ``BreakSpeedMultiplier`` knob (1.0 =
    vanilla Minecraft speed). Bare-handed log mining at speed 1.0 takes ~3 s of
    sustained attack on the same block (~60 ticks @ 20 tps), and the agent has
    to *then* walk over the dropped log for ``RewardForCollectingItems`` to fire.
    Across the typical 1000-step rollout, with stochastic head decoding +
    bare-handed slowness, that compound event rarely lands — every rollout this
    session returned ``total_reward = 0`` even when ``rollout_logs/exp2_thr/
    episode_004.mp4`` visibly shows the agent chopping a tree. Setting
    ``break_speed_multiplier=5.0`` collapses one break to ~12 ticks (~0.6 s), so
    the inventory-reward criterion has a fair shot of firing inside the rollout
    horizon. Frames are unchanged (the held-item slot stays empty); only the
    block-damage rate the engine applies per attack tick is scaled. The
    visual training distribution is unaffected.
    """

    def __init__(self, name, fixed_biome, inventory, reward_items,
                 break_speed_multiplier=1.0, start_pitch=None):
        self.fixed_biome = fixed_biome
        self.inventory = inventory
        self.reward_items = reward_items
        self.break_speed_multiplier = float(break_speed_multiplier)
        # If start_pitch is set (degrees, Malmo convention: positive = look down),
        # the agent spawns at a fixed XYZ with pitch override. Forces an aim
        # condition for evaluation where the trained head can't reliably point
        # downward by itself — the contractor data is heavily zero-pitch and
        # cam_weight isn't enough to overcome the prior. Yaw is left at 0.
        # XYZ is fixed at (0.5, 80, 0.5): above the procedural forest surface,
        # agent falls ~15 blocks onto terrain in <30 ticks. Procedural forest
        # generation around spawn near (0,0) reliably puts trees within view.
        self.start_pitch = None if start_pitch is None else float(start_pitch)
        super().__init__(
            name=name,
            max_episode_steps=_MAX_EPISODE_STEPS,
            reward_threshold=float("inf"),  # never auto-succeed; eval runs full length
        )

    def create_rewardables(self) -> List[Handler]:
        return [handlers.RewardForCollectingItems(self.reward_items)]

    def create_agent_start(self) -> List[Handler]:
        # Keep the gui/gamma/fov/cursor + low-level-input handlers from the base.
        # The MineDreamer generation agent ran bare-handed (info.json inventory=None;
        # the held item occupies bottom-center of every POV frame, so arming a tool
        # would be out-of-distribution). Only add a starting inventory if one is set.
        start = super().create_agent_start()
        if self.inventory:
            start.append(handlers.SimpleInventoryAgentStart(self.inventory))
        if self.break_speed_multiplier != 1.0:
            start.append(handlers.AgentStartBreakSpeedMultiplier(self.break_speed_multiplier))
        if self.start_pitch is not None:
            start.append(handlers.AgentStartPlacement(
                x=0.5, y=80, z=0.5, yaw=0.0, pitch=self.start_pitch,
            ))
        return start

    def create_agent_handlers(self) -> List[Handler]:
        # No quit-on-possession: episodes run to --max-steps for comparable rollouts.
        return []

    def create_server_world_generators(self) -> List[Handler]:
        return [
            handlers.DefaultWorldGenerator(
                force_reset="true",
                generator_options=_GEN_OPTS_TEMPLATE.format(biome=self.fixed_biome),
            )
        ]

    def create_server_quit_producers(self) -> List[Handler]:
        return [
            handlers.ServerQuitFromTimeUp(_MAX_EPISODE_STEPS * MS_PER_STEP),
            handlers.ServerQuitWhenAnyAgentFinishes(),
        ]

    def create_server_decorators(self) -> List[Handler]:
        return []

    def create_server_initial_conditions(self) -> List[Handler]:
        # Pin midday (start_time=6000) so frozen-time episodes stay bright and
        # match the training data's brightness (measured L≈81); the default start
        # time spawned at a low-sun evening (eval L≈57), a brightness gap better
        # closed here than by per-frame LAB normalization the training never had.
        return [
            handlers.TimeInitialCondition(allow_passage_of_time=False, start_time=6000),
            handlers.SpawningInitialCondition(allow_spawning=True),
        ]

    def determine_success_from_rewards(self, rewards: list) -> bool:
        return False

    def is_from_folder(self, folder: str) -> bool:
        return False

    def get_docstring(self):
        return self.name


_SPECS = [
    _Combined640EnvSpec(
        "MineRLChopATree640-v0",
        fixed_biome=BIOME_FOREST,
        inventory=[],  # bare-handed: matches the generation agent (info.json inventory=None)
        reward_items=[dict(type="log", amount=1, reward=1.0)],
    ),
    _Combined640EnvSpec(
        "MineRLCollectDirt640-v0",
        fixed_biome=BIOME_PLAINS,
        inventory=[],  # bare-handed: matches the generation agent (info.json inventory=None)
        reward_items=[dict(type="dirt", amount=1, reward=1.0)],
    ),
    # Fast variants: identical except for BreakSpeedMultiplier=5.0. Use these
    # for BC evaluation where total_reward needs to be a usable signal — the
    # default envs almost never see reward fire (bare-handed log break = ~3 s
    # sustained attack on the same block + walk over the dropped item, a
    # compound event that rarely lands in 1000-step rollouts). The Fast
    # variants drop log-break to ~0.6 s, so a competent agent's reward signal
    # actually fires.
    _Combined640EnvSpec(
        "MineRLChopATree640Fast-v0",
        fixed_biome=BIOME_FOREST,
        inventory=[],
        reward_items=[dict(type="log", amount=1, reward=1.0)],
        break_speed_multiplier=5.0,
    ),
    _Combined640EnvSpec(
        "MineRLCollectDirt640Fast-v0",
        fixed_biome=BIOME_PLAINS,
        inventory=[],
        reward_items=[dict(type="dirt", amount=1, reward=1.0)],
        break_speed_multiplier=5.0,
    ),
    # Force-aim variant: same as Fast but spawns at fixed XYZ looking down 20°
    # so the crosshair lands on a tree trunk if any is in the spawn vicinity.
    # Tests whether the current no_move_fix CLIP head can complete the chop
    # task given a fair starting aim — diagnoses pitch-policy collapse vs
    # other failure modes.
    _Combined640EnvSpec(
        "MineRLChopATree640FastAim-v0",
        fixed_biome=BIOME_FOREST,
        inventory=[],
        reward_items=[dict(type="log", amount=1, reward=1.0)],
        break_speed_multiplier=5.0,
        start_pitch=20.0,
    ),
]


def _register_specs() -> None:
    """Register the custom env ids (idempotent — safe to import twice)."""
    for spec in _SPECS:
        try:
            spec.register()
        except Exception:
            # Already registered in this process; ignore.
            pass


_register_specs()


# --------------------------------------------------------------------------- #
# Color matching                                                              #
# --------------------------------------------------------------------------- #

# Per-task Reinhard targets in OpenCV uint8 LAB space:
# [L_mean, L_std, a_mean, a_std, b_mean, b_std]. Precomputed from the generated
# training videos (see color_targets.json); the hardcoded fallback lets the
# Docker container run without trajectories_un mounted.
_LAB_TARGETS_FALLBACK = {
    "MineRLChopATree640-v0": [63.88, 40.62, 118.27, 10.60, 131.59, 11.33],
    "MineRLCollectDirt640-v0": [81.21, 32.54, 119.52, 10.67, 123.47, 12.13],
    # Fast variants share their parent task's training-distribution color
    # statistics — the BreakSpeedMultiplier only changes mining mechanics,
    # not rendering.
    "MineRLChopATree640Fast-v0": [63.88, 40.62, 118.27, 10.60, 131.59, 11.33],
    "MineRLCollectDirt640Fast-v0": [81.21, 32.54, 119.52, 10.67, 123.47, 12.13],
}


def _load_lab_targets() -> dict:
    path = os.path.join(os.path.dirname(__file__), "color_targets.json")
    try:
        with open(path) as fp:
            return json.load(fp)
    except (OSError, ValueError):
        return dict(_LAB_TARGETS_FALLBACK)


LAB_TARGETS = _load_lab_targets()


class ReinhardColorWrapper(gym.ObservationWrapper):
    """Put ``obs['pov']`` into the same color space the model trained on.

    The training videos are **real MineRL POV** (MineDreamer ``play`` output) saved
    through a cv2 BGR/RGB bug that **swapped red and blue** — confirmed visually
    (HUD hearts render blue; swapping back yields red hearts + blue water). The model
    therefore learned on R↔B-swapped frames, so the channel swap is the essential and
    *sufficient* color correction: with the swap, eval frames match the training
    distribution in hue (measured LAB a/b match within noise).

    The optional ``lab_transfer`` applies a per-task Reinhard LAB moment-match. It is
    **off by default**: it was originally added to bridge a presumed generated-vs-real
    gap that does not exist (both sides are the real engine), and per-frame LAB
    normalization is itself a transform the raw training frames never had. Brightness
    differences are better fixed at the env-config source (see ``create_server_initial_conditions``,
    which pins midday). Kept available for ablation.

    Mutates the pov in place so both the model input AND the recorded video reflect
    the correction. Only the ``pov`` key is touched; other obs keys pass through.
    """

    def __init__(self, env, target=None, swap_rb=True, lab_transfer=False):
        super().__init__(env)
        if lab_transfer:
            if target is None or len(target) != 6:
                raise ValueError(
                    f"lab_transfer needs target [L_mean,L_std,a_mean,a_std,b_mean,b_std], got {target!r}"
                )
            self.target = [float(x) for x in target]
        else:
            self.target = None
        self.swap_rb = swap_rb
        self.lab_transfer = lab_transfer

    def observation(self, obs):
        pov = obs.get("pov")
        if pov is None:
            return obs
        if self.swap_rb:
            pov = pov[:, :, ::-1]  # R<->B to match the training data's channel order
        if not self.lab_transfer:
            obs["pov"] = np.ascontiguousarray(pov)
            return obs
        lab = cv2.cvtColor(np.ascontiguousarray(pov), cv2.COLOR_RGB2LAB).astype(np.float32)
        for c in range(3):
            m = float(lab[:, :, c].mean())
            s = float(lab[:, :, c].std()) + 1e-6
            tm, ts = self.target[2 * c], self.target[2 * c + 1]
            lab[:, :, c] = (lab[:, :, c] - m) * (ts / s) + tm
        obs["pov"] = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
        return obs


def maybe_wrap_color(env, env_id, mode="auto"):
    """Wrap ``env`` with ReinhardColorWrapper per the --color-match mode.

    The essential correction is the R↔B channel swap (the training videos have
    swapped channels). LAB_TARGETS doubles as the registry of env ids whose frames
    need it. modes:
      * "auto" — swap-only iff env_id is a known custom env (default).
      * "on"   — swap-only; error if env_id is unknown.
      * "lab"  — swap + per-task Reinhard LAB transfer (opt-in; needs a target).
      * "off"  — never wrap.
    Returns (env, applied_bool).
    """
    if mode == "off":
        return env, False
    target = LAB_TARGETS.get(env_id)
    if target is None and mode in ("on", "lab"):
        raise ValueError(
            f"--color-match {mode}, but no color target registered for env {env_id!r}. "
            f"Known: {sorted(LAB_TARGETS)}"
        )
    if target is None:  # mode == "auto", unknown env
        return env, False
    return ReinhardColorWrapper(env, target, swap_rb=True, lab_transfer=(mode == "lab")), True
