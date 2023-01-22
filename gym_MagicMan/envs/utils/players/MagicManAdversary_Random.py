from gym_MagicMan.envs.utils.MagicManPlayer import AdversaryPlayer
import torch
from torch.distributions import Uniform, Categorical

class RandomAdversary(AdversaryPlayer):

    def __init__(self):
        super().__init__()     
        self.bid_distribution = Uniform(0,1)
  
    def play (self,obs,action_mask):
    
        action_distribution = torch.rand((60))+1e-5
        action_distribution = action_distribution*action_mask
        self.action_distribution = Categorical(action_distribution)
        
        return self.action_distribution.sample()

    
    def bid(self,obs):
        return self.bid_distribution.sample().item()

    def clean_hand(self):
        self.cards = []  
        
        
if __name__ == "__main__":
    rand_adv = RandomAdversary()