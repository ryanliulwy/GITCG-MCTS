import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from gitcg_double_mini_gym_env import GITCGDoubleMiniGymEnv

from stable_baselines3.common.env_util import make_vec_env
from gymnasium.spaces.utils import flatten_space, flatten
import gymnasium as gym

class PettingZooSB3Wrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observations = flatten_space(env.observations[self.agents[0]])
        self.action_space = env.action_space[self.agents[0]]

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return flatten(self.env.observations[self.agents[0]], obs[self.agents[0]])

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return flatten(self.env.observations[self.agents[0]], obs[self.agents[0]]), reward, done, truncated, info
    
class MaskedPolicy(nn.Module):
    def __init__(self, observation_space, action_space, features_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(observation_space.shape[0], features_dim)
        self.fc2 = nn.Linear(features_dim, features_dim)
        self.action_head = nn.Linear(features_dim, action_space.n)
        self.value_head = nn.Linear(features_dim, 1)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        logits = self.action_head(x)
        
        action_mask = obs["action_mask"]  # Get action mask from observation

        if self.training:  # Apply action masking only during training
            logits = logits + (1 - action_mask) * -1e9  # Mask invalid actions

        return logits, self.value_head(x)

class MaskedPPO(PPO):
    def _predict(self, observation, deterministic=True):
        action_logits, _ = self.policy(observation)
        action_mask = observation["action_mask"]
        
        if self.training:  # Apply action mask only in training mode
            masked_logits = action_logits + (1 - action_mask) * -1e9
            action_probs = F.softmax(masked_logits, dim=-1)
        else:
            action_probs = F.softmax(action_logits, dim=-1)  # No mask in testing

        action = torch.argmax(action_probs, dim=-1) if deterministic else torch.multinomial(action_probs, 1)
        return action

env = GITCGDoubleMiniGymEnv()
env = PettingZooSB3Wrapper(env)
vec_env = make_vec_env(lambda: env, n_envs=4)

model = MaskedPPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=100000)