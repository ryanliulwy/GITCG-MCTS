import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from gitcg_double_mini_gym_env import GITCGDoubleMiniGymEnv

from stable_baselines3.common.env_util import make_vec_env
from gymnasium.spaces.utils import flatten_space, flatten
import gymnasium as gym

import numpy as np
from gymnasium import spaces

class PettingZooSB3Wrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
        # Get the size of the flattened observation
        self.observation_size = self._get_obs_size()
        
        # Define a Box space with the correct shape
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.observation_size,), dtype=np.float32
        )
        
        self.action_space = env.action_space()  # Ensure this works

    def _get_obs_size(self):
        """Calculate the size of the flattened observation space"""
        return 27  # 10 (character) + 1 (dice) + 5 (cards) + 1 (declared_end) + 8 (action_mask)

    def _flatten_observation(self, obs):
        """Flatten nested dictionary into a structured 1D numpy array."""
        flat_obs = []

        # Character attributes (10 values)
        char = obs["Kaeya"]
        flat_obs.append(char["max_hp"])
        flat_obs.append(char["hp"])
        flat_obs.append(char["max_energy"])
        flat_obs.append(char["energy"])
        flat_obs.append(char["atk_permanent"])
        flat_obs.append(char["atk_discount"])
        flat_obs.extend(char["actions"])  # Action IDs (3 values)
        flat_obs.append(int(bool(char["artifact"])))  # Encode artifact as binary
        flat_obs.append(int(bool(char["weapon"])))  # Encode weapon as binary
        flat_obs.append(char["full"])  # 1 if full, 0 otherwise

        # Dice count (1 value)
        flat_obs.append(obs["dice"])

        # Cards in hand (5 values, already integers)
        flat_obs.extend(obs["cards"])

        # Declared end flag (1 value)
        flat_obs.append(obs["declared_end"])

        # Action mask (8 values)
        flat_obs.extend(obs["action_mask"])

        return np.array(flat_obs, dtype=np.float32)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)[0]
        return self._flatten_observation(obs[self.env.agents[0]]), self.env.reset(**kwargs)[1]

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return self._flatten_observation(obs[self.env.agents[0]]), reward, done, truncated, info  

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

def make_env():
    return PettingZooSB3Wrapper(GITCGDoubleMiniGymEnv())

vec_env = make_vec_env(make_env, n_envs=4)

model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=100000)
