import gym
import minerl

# MineRLBasaltFindCave-v0 is a v1.0 environment (640x360 obs, near-human action space)
env = gym.make("MineRLBasaltFindCave-v0")
obs = env.reset()
print("✅ Environment reset OK")
print("Obs keys:", list(obs.keys()))

for step in range(10):
    action = env.action_space.no_op()
    obs, reward, done, info = env.step(action)
    print(f"Step {step}: reward={reward:.3f}, done={done}")
    if done:
        obs = env.reset()

env.close()
print("✅ Random policy rollout complete")
