import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.spaces import Dict,Box

import numpy as np
import torch
import random
from collections import deque

import gym_MagicMan.envs.utils.MagicNet as net
from gym_MagicMan.envs.utils.MagicManPlayer import AdversaryPlayer, TrainPlayer
import gym_MagicMan.envs.utils.MagicManDeck as deck


class MagicManEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,init_state=None,adversaries='random',verbose=False,verbose_obs=False,current_round=15):
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'        
        
        self.verbose = verbose
        self.verbose_obs= verbose_obs
    
        self.single_obs_space=334
        
        self.round_deck = []
        self.players = []
        
        if adversaries=='random':
            self.players = [AdversaryPlayer() for _ in range(3)]
        
        self.train_player = TrainPlayer()
        self.players.append(self.train_player)
        self.noorder_players = self.players #pls dont be a deep copy
        self.n_players = len(self.players)
        self.max_rounds = int(60/self.n_players)
        self.current_round = current_round
        self.bids = torch.zeros(self.n_players)#torch.full(tuple([self.n_players]),float(round(self.current_round/self.n_players)))
        self.trump = 0 #trump is red every time so the bots have a better time learning
        random.shuffle(self.players)
        self.players = deque(self.players) #pick from a list of players
        self.turnorder_idx = 0
        self.bid_idx = 0
        self.turn_cards = []
        self.current_suit_idx = 5
        self.current_suit = torch.zeros(6)
        self.all_bid_completion = torch.zeros(self.n_players)

        self.observation_space = Dict({_: Dict({"norm_bids":Box(low=np.full((self.n_players),-4),
                                                                high=np.full((self.n_players),4),
                                                               dtype=np.float32
                                                               ), #is done by "manual" normalization --> replace with torch function?
                                       "all_bid_completion":Box(low=np.full((self.n_players),-1),
                                                                high=np.full((self.n_players),1),
                                                            dtype=np.float32
                                                            ), #is done by tanh
                                       "player_idx":Box(low=np.full((self.n_players),0),
                                                        high=np.full((self.n_players),1),
                                                    dtype=np.float32
                                                    ), #sparse
                                       "player_self_bid_completion":Box(low=np.full((1),-1),
                                                                        high=np.full((1),1),
                                                                        dtype=np.float32
                                                                        ), #is done by tanh,
                                        #this information will be passed twice - once in an array of all players and once for self 
                                        #might be a problem because its biasing the decision
                                        #the agent will be told that there is another player with the exact same bid completion as him?
                                        #might also just not be a problem who knows
                                       "n_cards":Box(low=np.full((self.max_rounds),0),
                                                     high=np.full((self.max_rounds),1),
                                                 dtype=np.float32
                                                 ),#sparse
                                       "played_cards":Box(low=np.full((self.n_players,60),0),
                                                          high=np.full((self.n_players,60),1),
                                                      dtype=np.float32
                                                      ),#sparse
                                       "cards_tensor":Box(low=np.full((60),0),
                                                          high=np.full((60),1),
                                                      dtype=np.float32
                                                      ),#sparse
                                       "current_suit":Box(low=np.full((6),0),
                                                          high=np.full((6),1),
                                                      dtype=np.float32
                                                      )#sparse
                                      })
                                      for _ in range(self.current_round)})
        
        self.flat_obs_space = gym.spaces.utils.flatten_space(self.observation_space)
    
        self.action_space = Box(
                                     low=np.full(len(deck.deck),-1), 
                                     high=np.full(len(deck.deck),1),
                                     dtype=np.float32
                                     )
    
        
         
        #Observation Variables:
        
        self.bid_obs = None
        print("Bids are predetermined in this environment. --> see 'active bid' flag")
        print("Bid input is not ordered. Has to be implemented in the future.")
        self.r = 0
        self.info = {}
        self.done = False
        
        self.reset()
    
    def get_flat(self,obs_dict):
        return torch.from_numpy(gym.spaces.flatten(self.observation_space,obs_dict)).to(self.device)
    
    def reset(self):
        self.round_deck = []
        self.bids = torch.zeros(self.n_players) #torch.full(tuple([self.n_players]),float(round(self.current_round/self.n_players)))
        random.shuffle(self.players)
        self.players = deque(self.players)
        self.turnorder_idx = 0
        self.turn_cards = []
        self.current_suit_idx = 5
        self.current_suit = torch.zeros(6)
        self.all_bid_completion = torch.zeros(self.n_players)
        
        self.bid_obs = None
        self.r = 0
        self.info = {}
        self.done = False

        for player in self.noorder_players:
            player.round_suits = 0
            player.game_score = 0
            player.cards_obj = []
            player.round_obs = {_:{"norm_bids"                  : torch.zeros(self.n_players),
                                   "all_bid_completion"         : torch.zeros(self.n_players),
                                   "player_idx"                 : torch.zeros(self.n_players),
                                   "player_self_bid_completion" : torch.zeros(1),
                                   "n_cards"                    : torch.zeros(self.max_rounds),
                                   "played_cards"               : torch.zeros((self.n_players,60)),
                                   "cards_tensor"               : torch.zeros(60),
                                   "current_suit"               : torch.zeros(6),
                                   } for _ in range(self.current_round)}
       
        obs,r,done,info = self.init_bid_step()
        
        
        return obs


    def render(self, mode='human', close=False):
        raise NotImplementedError
        
        
    def starting_player(self,starting_player):
        self.players.rotate(  -self.players.index(starting_player) )
    
    def init_step(self):
        self.state = "TURN"
        self.turn_idx = 0
        self.current_suit_idx = 5
        self.current_suit = torch.zeros(6)
        flat_round_obs ,self.r, self.done, self.info = self.step(action = None)
        return flat_round_obs, self.r, self.done, self.info
            
    def step(self,action): #!!not round
        
        if action is not None:
            if type(action)==np.ndarray:
                action=torch.from_numpy(action)
            if self.verbose:
                print(f"Train Player action: {action}\nTrain Player hand: {self.train_player.cards_obj}")
            action_prob_distribution = torch.nn.functional.softmax(action,dim=-1)+1e-5
            valid_action_dist = action_prob_distribution*self.train_player.turn_obs["cards_tensor"]
            action = torch.argmax(valid_action_dist)
            played_card = deck.deck[action]
            self.turn_cards.append(played_card)
            self.train_player.cards_obj.remove(played_card)
            self.r,self.info = 0,{}
            self.turnorder_idx +=1
            
        if action is None:
            assert self.turnorder_idx==0, f"The turn index is {self.turnorder_idx} | Should be 0."
        
        norm_bids = self.bids/self.current_round

        while True: #returns when necessary
            if not self.turnorder_idx == (self.n_players):
                #____________________________________________________
                #collecting observation
                player = self.players[self.turnorder_idx] #pls do not be a deep copy
                
                player.turn_obs = {"norm_bids"                  : norm_bids,
                                   "all_bid_completion"         : self.all_bid_completion,
                                   "player_idx"                 : torch.zeros(self.n_players),
                                   "player_self_bid_completion" : torch.zeros(1),
                                   "n_cards"                    : torch.zeros(self.max_rounds),
                                   "played_cards"               : torch.zeros((self.n_players,60)),
                                   "cards_tensor"               : torch.zeros(60),
                                   "current_suit"               : torch.zeros(6),
                                   }
                
                player.turn_obs["player_idx"][self.turnorder_idx] = 1
                                
                player_self_bid_completion = torch.tanh(torch.tensor(player.round_suits-player.current_bid))
                player.turn_obs["player_self_bid_completion"]=torch.unsqueeze(player_self_bid_completion,0)
               
                
                player.turn_obs["n_cards"][len(player.cards_obj)-1] = 1 #how many cards there are in his hand

                self.current_suit = torch.zeros(6)
                current_suit_idx = deck.legal(self.turn_cards,player.cards_obj,self.trump)
                self.current_suit[current_suit_idx] = 1
                player.turn_obs["current_suit"]=self.current_suit
                
                for card in player.cards_obj:
                    if card.legal:
                        player.turn_obs["cards_tensor"][deck.deck.index(card)] = 1

                for card_idx in range(len(self.turn_cards)):
                    played_card = self.turn_cards[card_idx]
                    player.turn_obs["played_cards"][card_idx][deck.deck.index(played_card)] = 1
                
                assert self.turn_idx<self.current_round,f"Turn index is {self.turn_idx}, Round is {self.current_round}"
                player.round_obs[self.turn_idx] = player.turn_obs #incomplete information to be completed at the end of turn
                player_obs = player.round_obs
                
                """
                Each player might have a different stack of observations
                Emphasis is on order of players --> each player observes the order of the played cards differently
                the player to a players immediate left is a different player for every player
                """

                
                if isinstance(player,AdversaryPlayer):
                    if self.verbose_obs:
                        print(f"Adverse Player Observation: {player_obs}")
                    #action is input not output!!!
                    net_out = player.play(self.get_flat(player_obs))
                    action_prob_distribution = torch.nn.functional.softmax(net_out,dim=-1)+1e-5
                    card_activation = player.turn_obs["cards_tensor"]*action_prob_distribution
                    action_idx = torch.argmax(card_activation)
                    
                    played_card = deck.deck[action_idx]
                    if self.verbose:
                        print(f"Adv Intended Card: {played_card}\nAdv Player Hand: {player.cards_obj}")

                    player.cards_obj.remove(played_card)
                    self.turn_cards.append(played_card)
                    self.turnorder_idx +=1
                    
                elif isinstance(player,TrainPlayer):
                    if self.verbose_obs:
                        print(f"Train Player Observation: {player_obs}")
                    dict_round_obs = player_obs
                    return dict_round_obs, self.r, self.done, self.info
                    
                else:
                    raise UserWarning (f"Player is {type(player)} not instance of either AdversaryPlayer or TrainPlayer")            
                
            if self.turnorder_idx == (self.n_players):
                
                self.compute_final_turn_observations()
                
                deck.turn_value(self.turn_cards,self.trump,self.current_suit_idx) #turn value of the players cards    
                winner = self.players[[card.turn_value for card in self.turn_cards].index(max(card.turn_value for card in self.turn_cards))]        
                self.starting_player(winner)# --> rearanges the players of the player such that the winner is in the first position

                winner.round_suits += 1 #winner of the suit

                all_round_scores = torch.tensor([player.round_suits for player in self.noorder_players])
                self.all_bid_completion = torch.tanh(all_round_scores-self.bids)
                self.turn_cards = []
                self.turnorder_idx = 0

                self.turn_idx += 1
            
                if self.turn_idx == self.current_round:
                    return self.conclude_step()

        raise UserWarning (f"Turn Step should have returned an Observation but has not")

    def compute_final_turn_observations(self):
        
        norm_bids = self.bids/self.current_round
            
        for player_idx,player in enumerate(self.players):
                
            player.turn_obs = {"norm_bids"                  : norm_bids,
                               "all_bid_completion"         : self.all_bid_completion,
                               "player_idx"                 : torch.zeros(self.n_players),
                               "player_self_bid_completion" : torch.zeros(1),
                               "n_cards"                    : torch.zeros(self.max_rounds),
                               "played_cards"               : torch.zeros((self.n_players,60)),
                               "cards_tensor"               : torch.zeros(60),
                               "current_suit"               : torch.zeros(6),
                               }
                
            player.turn_obs["player_idx"][player_idx] = 1
                                
            player_self_bid_completion = torch.tanh(torch.tensor(player.round_suits-player.current_bid))
            player.turn_obs["player_self_bid_completion"]=torch.unsqueeze(player_self_bid_completion,0)
                

            player.turn_obs["n_cards"][len(player.cards_obj)-1] = 1 #how many cards there are in his hand

            self.current_suit = torch.zeros(6)
            current_suit_idx = deck.legal(self.turn_cards,player.cards_obj,self.trump)
            self.current_suit[current_suit_idx] = 1
            player.turn_obs["current_suit"]=self.current_suit
            
            for card in player.cards_obj:
                if card.legal:
                    player.turn_obs["cards_tensor"][deck.deck.index(card)] = 1
            

            for card_idx in range(len(self.turn_cards)):
                played_card = self.turn_cards[card_idx]
                player.turn_obs["played_cards"][card_idx][deck.deck.index(played_card)] = 1
            
            player.round_obs[self.turn_idx] = player.turn_obs 
            
            
            
    def init_bid_step(self,active_bid=False):

        self.state = "BID"
        
        self.round_deck = deck.deck.copy()
        random.shuffle(self.round_deck) 
        
        for _ in range(self.current_round):
            for player in self.noorder_players:
                player.cards_obj.append(self.round_deck.pop(-1))
                #pop not only removes the item at index but also returns it
                
        self.bids = torch.full(tuple([self.n_players]),float(round(self.current_round/self.n_players)))#bids that are not yet given become 0 (bids are normalized so 0 is the expected bid)

        self.bid_idx = 0
        self.done = False
        self.r,self.info = 0,{}

        if active_bid:
            self.bid_obs, self.r, self.done, self.info = self.bid_step(action=None,active_bid=active_bid)
            return self.bid_obs.numpy(), self.r, self.done, self.info
        else:
            obs, self.r, self.done, self.info = self.bid_step(action=None,active_bid=active_bid)
            return obs, self.r, self.done, self.info


    def bid_step(self,action,active_bid=False): # !!not turn
        
        if action is not None:
            self.bids[self.bid_idx] = action
            self.train_player.current_bid = action
            self.r,self.info = 0,{}
            self.bid_idx += 1
        
        while self.bid_idx <= (self.n_players-1): # order is relevant

            player = self.players[self.bid_idx]
            
            n_cards = torch.zeros(self.max_rounds)
            n_cards[len(player.cards_obj)-1] = 1 # how many cards there are in his hand
            player_idx = torch.zeros(self.n_players)
            player_idx[self.players.index(player,0,self.n_players)] = 1 # what place in the players the player has

            player.cards_tensor = torch.zeros(60)
            for card in player.cards_obj:
                player.cards_tensor[deck.deck.index(card)] = 1 # cards in hand
            last_player_bool = torch.zeros(1)
            #if self.players.index(player) ==  3
            #    last_player_bool[0] = 1
            
            norm_bids = self.bids/self.current_round

            player_obs = torch.cat((norm_bids,
                                    player_idx,
                                    n_cards,
                                    player.cards_tensor,
                                    torch.tensor([self.current_round])),dim=0)
            
            if isinstance(player,AdversaryPlayer):
                player.current_bid = round(self.current_round/self.n_players)
                self.bids[self.bid_idx] = player.current_bid
                self.bid_idx += 1

            elif isinstance(player,TrainPlayer):
                if active_bid:
                    self.bid_obs = player_obs
                    return self.bid_obs, self.r, self.done, self.info
                else:
                    player.current_bid = round(self.current_round/self.n_players)
                    self.bids[self.bid_idx] = player.current_bid
                    self.bid_idx += 1
            else:
                raise UserWarning (f"Player is {type(player)} not instance of either AdversaryPlayer or TrainPlayer")
                
        if self.bid_idx == (self.n_players):
            return self.init_step()


    def conclude_step(self):
        assert self.turn_idx == self.current_round, (f"Turn index is {self.turn_idx} and should be equal to Current Round [{self.current_round}]")       
        self.state="CONCLUDE"
        for player in self.players:

            if player.current_bid == player.round_suits:
                round_reward = player.current_bid + 2 #ten point for every turn won and 20 for guessing right
            else:
                round_reward =  -abs(player.current_bid-player.round_suits) #ten points for every falsly claimed suit
                
            player.game_score += round_reward
            if isinstance(player,TrainPlayer):
               self.r = round_reward
               self.done = True
                
        for player in self.players:
            player.clean_hand() #at this point all hands should be empty anyways
            self.all_bid_completion = torch.zeros(self.n_players)
            
        final_obs = self.train_player.round_obs

        return final_obs,self.r,self.done,self.info





if __name__ == "__main__":


    env = MagicManEnv(adversaries='random',current_round = 2)
    obs = env.reset()
    external_train_net = net.PlayNet(current_round=env.current_round, single_obs_space=env.single_obs_space, action_space=len(deck.deck))
    
    done = False
    
    
    while not done:
        player_action = external_train_net(obs)
        obs,r,done,info = env.step(player_action)
        print(r,done)
    


