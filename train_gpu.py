import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import shutil
from tqdm.notebook import tqdm
from time import strftime, time
from datetime import datetime
from collections import deque, namedtuple
import random
import pygame

import sys
sys.path.insert(0,'/content/gym-MagicMan')

import wandb
from wandb.integration.sb3 import WandbCallback


import gymnasium as gym
import gym_MagicMan

import torch
import os
import time

from stable_baselines3 import A2C,PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

import train_utils


device = 'cuda' if torch.cuda.is_available() else 'cpu'
Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
torch.cuda.set_per_process_memory_fraction(.9, 0)
print(device)

resume_id = None#"j4dhifyv"
save_path = "MagicManSavedModels"

config={"run_id":resume_id,
        "policy_type": "MlpPolicy",
        "learning_rate":1e-4,#linear_schedule(1e-3)
        "batch_size":2048,
        "n_steps":32768,
        "total_timesteps":100_000_000,
        "save_freq":10,
        "env_name": "MagicMan-v0",
        "seed":262144,
        "current_round": 4,
        "adversaries":"trained",
        "clip_range":0.2,#0.2
        "ent_coef":0.01,#0.0
        "vf_coef":0.01,#0.5
        "normalize_advantage":True,
        "n_epochs":10,
        "gamma":1,#0.99
        "gae_lambda":0.95,#0.95
        "policy_kwargs":dict(activation_fn=torch.nn.ReLU,net_arch=dict(pi=[2048,2048], vf=[2048,2048]))}



train_utils.train(config=config,
                  resume_id=resume_id,
                  save_path=save_path)
