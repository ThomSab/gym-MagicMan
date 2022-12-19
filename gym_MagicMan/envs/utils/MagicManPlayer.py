import numpy as np
import torch
import random



#______________________________________________________________________________
#Player Class constructor

class AdversaryPlayer:
    def __init__(self):
    
        self.name = "AdversaryPlayer"+str(random.randint(111111,999999))
    
        self.round_score = 0
        self.game_score  = 0
        self.cards_obj = []
        self.cards_tensor = torch.zeros(60) #one-hot encoded deck
        self.error_string = "empty"
        
        self.observation_shape ={_:{"norm_bids"                  : torch.zeros(self.n_players),
                                    "all_bid_completion"         : torch.zeros(self.n_players),
                                    "player_idx"                 : torch.zeros(self.n_players),
                                    "player_self_bid_completion" : torch.zeros(1),
                                    "n_cards"                    : torch.zeros(self.max_rounds),
                                    "played_cards"               : torch.zeros((self.n_players,60)),
                                    "legal_cards_tensor"         : torch.zeros(60),
                                    "current_suit"               : torch.zeros(6),
                                    } for _ in range(self.current_round)}
        
    def __repr__(self):
        return self.name
        
    def get_non_flat(self,flat_obs):
        assert len(flat_obs.shape)==1, f"Observation should be a 1d vector but is {flat_obs.shape}."
        
        #gym utils unflatten
            


class TrainPlayer:
    
    def __init__(self):
    
        self.name = "TrainPlayer"+str(random.randint(111111,999999))
    
        self.round_score = 0
        self.game_score  = 0
        self.cards_obj = []
        self.cards_tensor = torch.zeros(60) #one-hot encoded deck
        
    def play(self):
        raise UserWarning("Train Player input is external not internal --> DO NOT CALL trainplayer.play()") 
    
    def bid(self):
        raise UserWarning("Train Player input is external not internal --> DO NOT CALL trainplayer.bid()") 
    
    def clean_hand(self):
        self.cards = []  