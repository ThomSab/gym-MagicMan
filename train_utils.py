import gymnasium as gym
import gym_MagicMan

import torch
import os
import time
import sys

from stable_baselines3 import A2C,PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from gym_MagicMan.envs.utils.players.MagicManAdversary_Trained import TrainedAdversary
    
import wandb
from wandb.integration.sb3 import WandbCallback

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
        "current_round": 8,
        "adversaries":'naive',
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
                   
    if "seed" in config.keys():
        env.seed(config["seed"])
    env = Monitor(env)
    env = gym.wrappers.FlattenObservation(env)

    supported_action_spaces = (gym.spaces.discrete.Discrete,gym.spaces.multi_discrete.MultiDiscrete,gym.spaces.multi_binary.MultiBinary)
    if supported_action_spaces is not None:
        assert isinstance(env.action_space, supported_action_spaces), (
            f"The algorithm only supports {supported_action_spaces} as action spaces "
            f"but {self.action_space} was provided")
    
    return env

def mask_fn(env: gym.Env) -> torch.Tensor:
    return env.action_mask

def profile():
    cProfile.run("model.learn(total_timesteps=100000)")    


def train(config,resume_id=None,local=False,save_path=None):

    env = make_env(config)
    env = ActionMasker(env, mask_fn)


    if local:
        train_local_test(env=env,config=config)
    else:
        assert save_path, "When training online, save_path variable must be passed to training method."
        train_online(env=env,resume_id=resume_id,config=config,save_path=save_path)
  
  
def train_local_test(env,config):

    resume = False
    config["run_id"] = wandb.util.generate_id() + "_LOCALRUN"


    experiment_name = f"GPU_MPPO_R{config['current_round']}_{config['run_id']}"

    wandb.init(
            name=experiment_name,
            id=config['run_id'],
            project="MagicManGym",
            config=config,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            resume = resume
    )

    
    model = make_new_model(config,env)

    model.learn(total_timesteps=config["total_timesteps"],
                callback=WandbCallback(gradient_save_freq=0,
                                       verbose=2,
                                       model_save_freq=0,
                                       model_save_path=None))


def train_online(env,config,resume_id,save_path):

    if not resume_id:
        config['run_id'] = wandb.util.generate_id()
        model = online_init(config,env)
    else:
        model, config = online_resume(resume_id,save_path,env)

    model.learn(total_timesteps=config["total_timesteps"],
                callback=WandbCallback(gradient_save_freq=10_000,
                                       verbose=2,
                                       model_save_freq=10_000,
                                       model_save_path=f"{save_path}/{wandb.run.name}"))

def online_init(config,env):

    experiment_name = f"GPU_MPPO_R{config['current_round']}_{config['run_id']}"
    wandb.init(
                name=experiment_name,
                id=config['run_id'],
                project="MagicManGym",
                config=config,
                sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                monitor_gym=True,  # auto-upload the videos of agents playing the game
                resume = False
        )
    model = make_new_model(config,env)
    return model


def online_resume(resume_id,save_path,env):    

    wandb.init(project="MagicManGym",
               id=resume_id,
               sync_tensorboard=True,
               monitor_gym=True,
               resume="must")
    config = wandb.run.config        
    model = MaskablePPO.load(f"{save_path}/{wandb.run.name}"+r"/model",env=env)
    return model,config


def print_params(model):
    params = model.get_parameters()
    for key,val in params["policy"].items():
        print(key)
        print(val.shape)

def make_new_model(config,env):
    model = MaskablePPO(config["policy_type"], env, verbose=1,
                        seed=config["seed"],
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
    return model


if __name__ == "__main__":
    raise UserWarning("'train_utils.py' is not supposed to be called directly and contains only utility functions. Maybe you meant to call 'local_train_test.py'")



    

    
