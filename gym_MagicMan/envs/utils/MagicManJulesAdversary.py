import numpy
import torch

import gym_MagicMan.envs.utils.MagicManDeck as deck
from gym_MagicMan.envs.utils.MagicManPlayer import AdversaryPlayer
from torch.distributions import Uniform, Categorical


class JulesAdversary(AdversaryPlayer):


    """
    Round 2
    AdversaryPlayer622424
    0.2178
    AdversaryPlayer825622
    0.2386
    AdversaryPlayer824997
    0.2222
    
    Round 3
    AdversaryPlayer854383
    0.0857
    AdversaryPlayer636705
    0.0665
    AdversaryPlayer251647
    0.0968
    TrainPlayer727031
    0.1223
    
    Round 6
    AdversaryPlayer699149
    -0.0891
    AdversaryPlayer342757
    -0.09
    AdversaryPlayer145428
    -0.0896
    TrainPlayer121391
    0.0071
    
    Round 10
    TrainPlayer177419
    -1.6445
    AdversaryPlayer637333
    -0.2177
    AdversaryPlayer343616
    -0.2162
    AdversaryPlayer992781
    -0.2162
    
    Round 15
    AdversaryPlayer274368
    -0.302
    AdversaryPlayer153045
    -0.3189
    TrainPlayer118756
    -1.7445
    AdversaryPlayer947529
    -0.3066
    
    
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
        trump_bid = self.tmp_n_factor * sum(tmp_cards) + self.tmp_val_factor * sum([tmp_cards[_]*(_+1) for _ in range(len(tmp_cards))])
        
        clr_bid_dict = {"yellow_bid":0,"blue_bid":0,"green_bid":0}
        for clr_idx,clr_name in enumerate(clr_bid_dict.keys()):
            clr_cards = obs["legal_cards_tensor"][clr_idx+1:][::4][1:-1]
            clr_bid_dict[clr_name] = self.clr_n_factor * sum(clr_cards) + self.clr_val_factor * sum([clr_cards[_]*(_+1) for _ in range(len(clr_cards))])
        
        fool_bid = self.foo_factor * sum(obs["legal_cards_tensor"][:4])
        
        bid = magic_man_bid + trump_bid + sum(clr_bid_dict.values())
        
        
        return (bid / torch.argmax(obs["n_cards"])).item()      

        
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
        
        
        played_cards_obj = [deck.deck[torch.where(obs["played_cards"][turn]==1)[0]] for turn in range(len(obs["norm_bids"])) if sum(obs["played_cards"][turn])>0]
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
        
    