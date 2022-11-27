import gym
import gym_MagicMan
import torch
from stable_baselines3 import A2C
from stable_baselines3.common import env_checker

from stable_baselines3.common.logger import configure


tmp_path = "/tmp/sb3_log/"
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

env = gym.make("MagicMan-v0",current_round=2,verbose=0,verbose_obs=1)
env = gym.wrappers.FlattenObservation(env)

if __name__ == "__main__":

    
    model = A2C("MlpPolicy", env, verbose=1)
    model.set_logger(new_logger)
    model.learn(total_timesteps=10_000_000)
    

    #env_checker.check_env(env)