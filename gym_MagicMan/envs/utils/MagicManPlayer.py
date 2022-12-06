import numpy as np
import torch
import random



#______________________________________________________________________________
#Player Class constructor

class AdversaryPlayer:


    def __init__(self,play_network=None):
        
        self.name = "AdversaryPlayer"+str(random.randint(111111,999999))
        
        self.play_network = None
        if play_network:
            self.play_network = play_network
        
        self.round_score = 0
        self.game_score  = 0
        self.cards_obj = []
        self.cards_tensor = torch.zeros(60) #one-hot encoded deck
        self.error_string = "empty"
       
       
    def play (self,obs):
        if self.play_network:
            action_distribution = self.play_network(obs)
        else:
            action_distribution = torch.rand((60))
        return action_distribution
    
    def bid (self,obs):
        activation = self.bid_network(obs)
        
        self.current_activation = activation[0]
            #multiply by the number of card in hand
            #then divide by the amount of players 
            #to have a better starting point for the bots

        
        return self.current_activation

    def clean_hand(self):
        self.cards = []  

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