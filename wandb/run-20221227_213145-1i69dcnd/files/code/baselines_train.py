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



    

def linear_schedule(initial_value):
    """
    Linear learning rate schedule.
    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress):
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress: (float)
        :return: (float)
        """
        return progress * initial_value

    return func
    
config={"policy_type": "MlpPolicy",
        "learning_rate":linear_schedule(1e-3),#3e-4
        "batch_size":128,
        "total_timesteps":1_000_000,
        "env_name": "MagicMan-v0",
        "current_round": 2,
        "adversaries":'jules',
        "clip_range":0.2,#0.2
        "ent_coef":0.2,#0.0
        "vf_coef":0.5,#0.5
        "normalize_advantage":False,
        "n_epochs":10,
        "gamma":0.99,#0.99
        "gae_lambda":0.95,#0.95
        "policy_kwargs":dict(activation_fn=torch.nn.ReLU,net_arch=[dict(pi=[64,64], vf=[64, 64])])}


experiment_name = f"CPU_MPPO_R{config['current_round']}_{int(time.time())}"
    
    
def make_env(config=config):
    env = gym.make(config["env_name"],
                   current_round=config["current_round"],
                   adversaries=config["adversaries"])
    env = Monitor(env)
    env = gym.wrappers.FlattenObservation(env)
    return env

def mask_fn(env: gym.Env) -> torch.Tensor:
    return env.action_mask


if __name__ == "__main__":
    experiment_name = f"CPU_MPPO_R{config['current_round']}_{int(time.time())}"
    wandb.init(
            name=experiment_name,
            project="MagicManGym",
            config=config,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
    )

    env = make_env()
    env = ActionMasker(env, mask_fn)
    model = MaskablePPO(config["policy_type"], env, verbose=1,
                        learning_rate=config["learning_rate"],
                        clip_range = config["clip_range"],
                        ent_coef = config["ent_coef"],
                        vf_coef = config["vf_coef"],
                        normalize_advantage = config["normalize_advantage"],
                        n_epochs = config["n_epochs"],
                        gamma = config["gamma"],
                        gae_lambda = config["gae_lambda"],
                        batch_size=config["batch_size"],
                        policy_kwargs = config["policy_kwargs"],
                        tensorboard_log=f"runs/{experiment_name}")

    """
        params = model.get_parameters()
        for key,val in params["policy"].items():
            print(key)
            print(val.shape)
    """
    model.learn(total_timesteps=config["total_timesteps"],
                callback=WandbCallback(gradient_save_freq=1_000_000,
                                       verbose=2,
                                       model_save_freq=1_000_000,
                                       model_save_path=f"models/{experiment_name}",))