import numpy as np
import torch
import random



#______________________________________________________________________________
#Player Class constructor

class AdversaryPlayer:
    def __init__(self):

        random.seed()
        self.random_id = random.randint(111111,999999)
        
        self.round_score = 0
        self.game_score  = 0
        self.cards_obj = []
        self.cards_tensor = torch.zeros(60) #one-hot encoded deck
        self.error_string = "empty"

            


class TrainPlayer:
    
    def __init__(self):

        random.seed()
        self.random_id = random.randint(111111,999999)

        self.name = "TrainPlayer"+str(self.random_id)
    
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
