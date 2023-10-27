import gym_MagicMan
import time
from sb3_contrib.common.wrappers import ActionMasker
from wandb.integration.sb3 import WandbCallback
from gym_MagicMan.envs.utils.players.MagicManAdversary_Trained import TrainedAdversary
import train_utils 
import os
import torch

if __name__ == "__main__":
    config={"run_id":None,
            "policy_type": "MlpPolicy",
            "learning_rate":6e-4,#linear_schedule(1e-3)
            "batch_size":256,
            "n_steps":32768,
            "total_timesteps":50_000_000,
            "save_freq":10,
            "env_name": "MagicMan-v0",
            "seed":262144,
            "current_round": 8,
            "adversaries":"trained",
            "clip_range":0.3,#0.2
            "ent_coef":0.0,#0.0
            "vf_coef":0.5,#0.5
            "normalize_advantage":True,
            "n_epochs":10,
            "gamma":1,#0.99
            "gae_lambda":0.8,#0.95
            "policy_kwargs":dict(activation_fn=torch.nn.ReLU,net_arch=dict(pi=[1024,1024], vf=[1024,1024]))
            }

    save_path = "MagicManSavedModels"
    train_utils.train(config=config,test=True,save_path=save_path)
