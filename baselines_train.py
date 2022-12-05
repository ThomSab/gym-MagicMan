import gym
import gym_MagicMan

import torch
import os
import time

from wandb.integration.sb3 import WandbCallback
import wandb

from stable_baselines3 import A2C,PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

os.chdir(r'C:\Users\jaspe\Documents\UNI\Sem_III\SEMINAR\gym-MagicMan')


config={"policy_type": "MlpPolicy",
        "learning_rate":1e-4,
       "batch_size":64,
       "total_timesteps":1_000_000,
       "env_name": "MagicMan-v0",
       "current_round": 2}


experiment_name = f"PPO_{int(time.time())}"
wandb.init(
        name=experiment_name,
        project="MagicManGym",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
)

def make_env():
    env = gym.make(config["env_name"],current_round=config["current_round"])#,current_round=2,verbose=0,verbose_obs=0)
    env = Monitor(env)
    env = gym.wrappers.FlattenObservation(env)
    return env


env = DummyVecEnv([make_env])
model = PPO(config["policy_type"], env, verbose=1 ,learning_rate=config["learning_rate"],tensorboard_log=f"runs/{experiment_name}")#,batch_size=config["batch_size"])
model.learn(total_timesteps=config["total_timesteps"],
            callback=WandbCallback(gradient_save_freq=10_000,
                                   verbose=2,
                                   model_save_freq=10_000,
                                   model_save_path=f"models/{experiment_name}",))


