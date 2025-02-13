import gymnasium as gym
from dottore_gym.envs.dottore_gym_env import DottoreLinearGymEnv
# gym.envs.register(
#     id="DottoreGenius",
#     entry_point='dottore_gym.dottore_gym_env:DottoreLinearGymEnv',
#     max_episode_steps=1000,
# )
for k in gym.envs.registry.keys():
    print(k)
env = gym.make("DottoreGenius-v0")

obs = env.reset()
done = False

while not done:
    action = env.action_space.sample()  # Random action
    obs, reward, done, _ = env.step(action)
    env.render()

env.close()