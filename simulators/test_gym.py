import gymnasium as gym
from dottore_gym.envs.dottore_gym_env import DottoreLinearGymEnv

env = gym.make("DottoreGenius-v0")

obs = env.reset()
done = False

while not done:
    action = env.action_space.sample()  # Random action
    print(env.action_space)
    print(action)
    obs, reward, done, _ = env.step(action)
    env.render()

env.close()