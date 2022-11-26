from gym.envs.registration import register

register(
    id='MagicMan-v0',
    entry_point='gym_MagicMan.envs:MagicManEnv',
)
