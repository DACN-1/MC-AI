"""One-off probe: does the FlatInventoryObservation added to the 640 env specs
actually surface obs['inventory'] = {'log': n, 'dirt': m}? Run inside the MineRL
docker container (no model needed):

    docker compose run --rm minerl probe_inventory_obs.py
"""
import gym

import eval_envs  # noqa: F401  (import registers the custom env ids)


def probe(env_id: str) -> None:
    print(f"\n=== {env_id} ===", flush=True)
    env = gym.make(env_id)
    try:
        # Observation space first — proves the observable is registered.
        space = getattr(env, "observation_space", None)
        keys = list(space.spaces.keys()) if hasattr(space, "spaces") else None
        print("obs_space keys:", keys, flush=True)
        if keys is not None:
            print("inventory in obs_space:", "inventory" in keys, flush=True)

        obs = env.reset()
        ok = obs.get("inventory") if hasattr(obs, "get") else None
        print("reset obs keys:", list(obs.keys()) if hasattr(obs, "keys") else type(obs), flush=True)
        print("reset inventory:", ok, flush=True)

        for _ in range(10):
            obs, _r, done, _info = env.step(env.action_space.no_op())
            if done:
                break
        print("post-step inventory:", obs.get("inventory") if hasattr(obs, "get") else None, flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    for eid in ("MineRLCollectDirt640Fast-v0", "MineRLChopATree640Fast-v0"):
        try:
            probe(eid)
        except Exception as e:  # noqa: BLE001
            print(f"  PROBE ERROR on {eid}: {type(e).__name__}: {e}", flush=True)
    print("\nPROBE DONE", flush=True)
