import numpy
import torch
import os

import gym_MagicMan.envs.utils.MagicManDeck as deck
from gym_MagicMan.envs.utils.MagicManPlayer import AdversaryPlayer
from torch.distributions import Uniform, Categorical

from sb3_contrib.ppo_mask import MaskablePPO

class TrainedAdversary(AdversaryPlayer):

    def __init__(self,load_path):
        super().__init__()
        try:
            os.chdir(r"C:\Users\jasper\Documents\LINZ\Semester_III\SEMINAR\gym-MagicMan")
        except:
            print("chdir in trained player adversary module did not work, ignore if in colab")
        
        
        self.model = MaskablePPO.load(load_path)
             
        
        
    def bid (self,obs):
        return (torch.argmax(obs["n_cards"]).item()/4)      

        
    def play (self,obs,action_mask):
    
        action, _states = self.model.predict(obs)
        return action

    def clean_hand(self):
        self.cards = []  
        
    