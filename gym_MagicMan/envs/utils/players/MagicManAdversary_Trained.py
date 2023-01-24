import numpy
import torch
import os
import random

import gym_MagicMan.envs.utils.MagicManDeck as deck
from gym_MagicMan.envs.utils.MagicManPlayer import AdversaryPlayer
from torch.distributions import Uniform, Categorical

from sb3_contrib.ppo_mask import MaskablePPO

class TrainedAdversary(AdversaryPlayer):

    def __init__(self,load_path):
        super().__init__()

        self.name = "TrainedAdversary"+str(random.randint(111111,999999))
        self.model = MaskablePPO.load(load_path)
             
        
        
    def bid (self,obs):
        bid = (1/4)
        return bid   

        
    def play (self,obs,action_mask):
        
        action, _states = self.model.predict(obs,action_masks=action_mask)
        return action

    def clean_hand(self):
        self.cards = []  
        
    
