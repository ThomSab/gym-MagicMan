import numpy
import torch

import gym_MagicMan.envs.utils.MagicManDeck as deck
from gym_MagicMan.envs.utils.MagicManPlayer import AdversaryPlayer
from torch.distributions import Uniform, Categorical


class JulesAdversary(AdversaryPlayer):
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

    def __init__(self):
        super().__init__()
        
        self.mgm_factor     = 1
        self.tmp_n_factor   = .3
        self.tmp_val_factor = .07
        self.clr_n_factor   = -.8
        self.clr_val_factor = .1
        self.foo_factor     = -1
             
        
        
        
    def bid (self,obs):
        #bid as an integer and then divide by current round
        
        magic_man_bid = self.mgm_factor * sum(obs["legal_cards_tensor"][-4:])
        tmp_cards = obs["legal_cards_tensor"][::4][1:-1]
        trump_bid = self.tmp_n_factor * sum(tmp_cards) + self.tmp_val_factor * sum([tmp_card*(_+1) for _,tmp_card in enumerate(tmp_cards)])
        
        clr_bid_dict = {"yellow_bid":0,"blue_bid":0,"green_bid":0}
        for clr_idx,clr_name in enumerate(clr_bid_dict.keys()):
            clr_cards = obs["legal_cards_tensor"][clr_idx+1:][::4][1:-1]
            clr_bid_dict[clr_name] = self.clr_n_factor * sum(clr_cards) + self.clr_val_factor * sum([clr_card*(_+1) for _,clr_card in enumerate(clr_cards)])
        
        fool_bid = self.foo_factor * sum(obs["legal_cards_tensor"][:4])
        
        bid = magic_man_bid + trump_bid + sum(clr_bid_dict.values())
        
        
        return (bid / 4).item()      

        
    def play (self,obs,action_mask):
    
        for round_obs in obs.values():
            if round_obs["round_active_flag"]:
                obs = round_obs #very simple agent doesnt look into the past
    
        if obs["player_self_bid_completion"]<0:
            self.mode="GET_SUITS"
        elif obs["player_self_bid_completion"]>=0:
            self.mode="STOP_SUITS"
        else:
            raise UserWarning (f"Adversary bid completion is faulty {obs['player_self_bid_completion']}")
        
        
        played_cards_obj = [deck.deck[torch.argmax(turn_played_cards).item()] for turn_played_cards in obs["played_cards"] if sum(turn_played_cards)>0]
        current_suit_idx = deck.legal(played_cards_obj,self.cards_obj)
        
        deck.turn_value(played_cards_obj,current_suit_idx)
        deck.hand_turn_value(torch.argmax(obs["player_idx"]),self.cards_obj,current_suit_idx)
        
        
        
        cards_value_dict = {card:card.turn_value for card in self.cards_obj if card.legal}
        
        max_val_played=0
        if played_cards_obj:
            max_val_played = max([card.turn_value for card in played_cards_obj])
            
        can_take_suit = max([card.turn_value for card in self.cards_obj]) > max_val_played
        
        
        
        if self.mode == "GET_SUITS":
            if can_take_suit:
                maxval_card = max(cards_value_dict, key=cards_value_dict.get)
                return deck.deck.index(maxval_card)
            else:
                minval_card = min(cards_value_dict, key=cards_value_dict.get)
                return deck.deck.index(minval_card)
            
        elif self.mode == "STOP_SUITS":
            if max_val_played>0:
                win_cards = [key for key,val in cards_value_dict.items() if val>max_val_played]
                if len(win_cards)<len(cards_value_dict.values()):
                    for key in win_cards:
                        del cards_value_dict[key]
                    maxval_nonwin_card = max(cards_value_dict, key=cards_value_dict.get)
                    return deck.deck.index(maxval_nonwin_card)    
            
            minval_card = min(cards_value_dict, key=cards_value_dict.get)
            return deck.deck.index(minval_card)
                
        raise UserWarning ("Playing method of JulesAdversary should have returned a card index but did not")
        
        # hoch fehl am anfang weil bedienen
        # niedrige truempfe am anfang zum trumpf ziehen
        # dont be a fool with the fool
        # play the wizards on the valuable piles
        
        
        # karten zaehlen
    


    def clean_hand(self):
        self.cards = []  
        
    