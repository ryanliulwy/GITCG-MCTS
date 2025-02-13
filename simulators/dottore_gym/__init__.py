from gymnasium.envs.registration import register

register(
    id="DottoreGenius-v0",
    entry_point='dottore_gym.envs.dottore_gym_env:DottoreLinearGymEnv',
)
