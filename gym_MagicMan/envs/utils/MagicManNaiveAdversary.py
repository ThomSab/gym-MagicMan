import numpy
import torch

from gym_MagicMan.envs.utils.MagicManPlayer import AdversaryPlayer

class NaiveAdversary(AdversaryPlayer):


"""
    player.round_obs = {_:
                            {
                                "norm_bids"                  : torch.zeros(self.n_players),
                                "all_bid_completion"         : torch.zeros(self.n_players),
                                "player_idx"                 : torch.zeros(self.n_players),
                                "player_self_bid_completion" : torch.zeros(1),
                                "n_cards"                    : torch.zeros(self.max_rounds),
                                "played_cards"               : torch.zeros((self.n_players,60)),
                                "legal_cards_tensor"         : torch.zeros(60),
                                "current_suit"               : torch.zeros(6),
                            } 
                        for _ in range(self.current_round)}
"""

    def __init__(self,play_network=None):
        super().__init__()
        
        self.mgm_factor     = 1.5
        self.tmp_n_factor   = 0.5
        self.tmp_val_factor = 0.1
        self.clr_n_factor   = -0.1
        self.clr_val_factor = 0.1
        self.foo_factor     = -0.5
       
    def play (self,obs,action_mask):
    
        action_distribution = torch.rand((60))+1e-5
        action_distribution = action_distribution*action_mask
        self.action_distribution = Categorical(action_distribution)
        
        return self.action_distribution.sample()
    
    def bid (self,obs):
        #bid as an integer and then divide by current round
        
        magic_man_bid = mgm_factor * sum(obs["legal_cards_tensor"][-4:])
        tmp_cards = obs["legal_cards_tensor"][::4][1:-1]
        trump_bid = tmp_n_factor * sum(clr_cards) + tmp_val_factor * sum([tmp_cards[_]*(_+1) for _ in range(len(tmp_cards))])
        
        clr_bid_dict = {"yellow_bid":0,"blue_bid":0,"green_bid":0}
        for clr_idx,clr_name in enumerate(clr_bid_dict.keys()):
            clr_cards = obs["legal_cards_tensor"][clr_idx+1][::4][1:-1]
            clr_bid_dict[clr_name] = clr_n_factor * sum(clr_cards) + clr_val_factor * sum([clr_cards[_]*(_+1) for _ in range(len(clr_cards))])
        
        fool_bid = foo_factor * sum(obs["legal_cards_tensor"][:4])
        
        bid = sum(magic_man_bid,trump_bid) + sum(clr_bid_dict.values())
        
        
        return bid / torch.argmax(obs["n_cards"])

    def clean_hand(self):
        self.cards = []  