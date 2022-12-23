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

from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

try:
    os.chdir(r'C:\Users\jasper\Documents\LINZ\Semester_III\SEMINAR\gym-MagicMan')
except FileNotFoundError:
    print("PATH NOT FOUND. - Ignore if run in Colab - ")

config={"policy_type": "MlpPolicy",
        "learning_rate":1e-4,
       "batch_size":64,
       "total_timesteps":10_000_000,
       "env_name": "MagicMan-v0",
       "current_round": 2}


experiment_name = f"CPU_Masked_PPO_{int(time.time())}"

wandb.init(
        name=experiment_name,
        project="MagicManGym",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
)

def make_env():
    env = gym.make(config["env_name"],current_round=config["current_round"],verbose=0)#,verbose_obs=0)
    env = Monitor(env)
    env = gym.wrappers.FlattenObservation(env)
    return env

def mask_fn(env: gym.Env) -> torch.Tensor:
    return env.action_mask

env = make_env()
env = ActionMasker(env, mask_fn)
model = MaskablePPO(config["policy_type"], env, verbose=1 ,learning_rate=config["learning_rate"],tensorboard_log=f"runs/{experiment_name}")#,batch_size=config["batch_size"])

"""
    params = model.get_parameters()
    for key,val in params["policy"].items():
        print(key)
        print(val.shape)
    input()
"""

model.learn(total_timesteps=config["total_timesteps"],
            callback=WandbCallback(gradient_save_freq=1_000_000,
                                   verbose=2,
                                   model_save_freq=1_000_000,
                                   model_save_path=f"models/{experiment_name}",))