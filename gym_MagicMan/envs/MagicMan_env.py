import gym
from gym import error, spaces, utils
from gym.spaces import Dict,Box,Discrete

import numpy as np
import torch
import random
from collections import deque

import pygame

from gym_MagicMan.envs.utils.MagicManPlayer import TrainPlayer,AdversaryPlayer
from gym_MagicMan.envs.utils.players.MagicManAdversary_Random import RandomAdversary
from gym_MagicMan.envs.utils.players.MagicManAdversary_Naive import NaiveAdversary
from gym_MagicMan.envs.utils.players.MagicManAdversary_Trained import TrainedAdversary

import gym_MagicMan.envs.utils.MagicManRender as mm_render
import gym_MagicMan.envs.utils.MagicManDeck as deck


class MagicManEnv(gym.Env):
    metadata = {'render.modes': ['human','human_interactive'], "render_fps": 30}

    def __init__(self,init_state=None,adversaries='random',render_mode=None,verbose=False,verbose_obs=False,current_round=15):
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'        
        
        self.verbose = verbose
        self.verbose_obs= verbose_obs
        self.render_mode = render_mode
        
        self.single_obs_space=395
        
        self.round_deck = []
        self.players = []

        self.current_round = current_round
        
        
        if adversaries=='random':
            self.flat_out = False
            self.players = [RandomAdversary() for _ in range(3)]
        elif adversaries=='naive':
            self.flat_out = False
            self.players = [NaiveAdversary() for _ in range(3)]
        elif adversaries == 'trained':
            self.flat_out = True
            self.players = [TrainedAdversary(fr"models\TrainedAdversary_R{self.current_round}\model") for _ in range(3)]
            
        
        self.train_player = TrainPlayer()
        self.players.append(self.train_player)
        self.n_players = len(self.players)
        self.max_rounds = int(60/self.n_players)
        self.bids = torch.zeros(self.n_players)#torch.full(tuple([self.n_players]),float(round(self.current_round/self.n_players)))
        self.trump = 0 #trump is red every time so the bots have a better time learning
        random.shuffle(self.players)
        self.noorder_players = self.players
        
        if self.render_mode in ["human","human_interactive"]:
            train_player_idx = self.players.index(self.train_player)
            for player_idx,player in enumerate(self.players):
                player.table_idx = ((player_idx+(4-train_player_idx))%4)
        
        self.players = deque(self.players) #pick from a list of players
        self.turnorder_idx = 0
        self.bid_idx = 0
        self.turn_cards = []
        self.last_turn_cards = []
        self.next_round_idx_list = []
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
                                       "legal_cards_tensor":Box(low=np.full((60),0),
                                                          high=np.full((60),1),
                                                      dtype=np.float32
                                                      ),#sparse
                                       "cards_tensor":Box(low=np.full((60),0),
                                                          high=np.full((60),1),
                                                      dtype=np.float32
                                                      ),#sparse
                                       "current_suit":Box(low=np.full((6),0),
                                                          high=np.full((6),1),
                                                      dtype=np.float32
                                                      ),#sparse
                                       "round_active_flag":Box(low=np.full((1),0),
                                                               high=np.full((1),1),
                                                      dtype=np.float32
                                                      )
                                      })
                                      for _ in range(self.current_round)})
        
        self.flat_obs_space = gym.spaces.utils.flatten_space(self.observation_space)
    
        self.action_space = Discrete(60)
        self.action_mask = torch.zeros(60)
    
        
         
        #Observation Variables:
        
        self.bid_obs = None
        print("Bids are predetermined in this environment. --> see 'active bid' flag")
        print("Bid input is not ordered. Has to be implemented in the future.")
        self.r = 0
        self.info = {player.name:0 for player in self.players}
        self.done = False
        
        self.window = None
        self.clock = None
        self.window_size = (1000,700)
        self.activate_cards_buttons = mm_render.activate_cards_buttons
    
    def seed(self, seed) -> None:
        random.seed(seed)
        np.random.seed(seed)
    
    def get_flat(self,obs_dict):
        return torch.from_numpy(gym.spaces.flatten(self.observation_space,obs_dict)).to(self.device)
    
    def reset(self):
        self.round_deck = []
        self.bids = torch.zeros(self.n_players) #torch.full(tuple([self.n_players]),float(round(self.current_round/self.n_players)))
        random.shuffle(self.players)
        
        if self.render_mode in ["human","human_interactive"]:
            train_player_idx = self.players.index(self.train_player)
            for player_idx,player in enumerate(self.players):
                player.table_idx = ((player_idx+(4-train_player_idx))%4)
        
        self.players = deque(self.players)
        self.turnorder_idx = 0
        self.turn_cards = []
        self.last_turn_cards = []
        self.next_round_idx_list = []
        self.current_suit_idx = 5
        self.current_suit = torch.zeros(6)
        self.all_bid_completion = torch.zeros(self.n_players)
        
        self.bid_obs = None
        self.r = 0
        self.info = {player.name:0 for player in self.players}
        self.done = False

        for player in self.noorder_players:
            player.round_suits = 0
            player.cards_obj = []
            player.round_obs = {_:{"norm_bids"                  : torch.zeros(self.n_players),
                                   "all_bid_completion"         : torch.zeros(self.n_players),
                                   "player_idx"                 : torch.zeros(self.n_players),
                                   "player_self_bid_completion" : torch.zeros(1),
                                   "n_cards"                    : torch.zeros(self.max_rounds),
                                   "played_cards"               : torch.zeros((self.n_players,60)),
                                   "legal_cards_tensor"         : torch.zeros(60),
                                   "cards_tensor"               : torch.zeros(60),
                                   "current_suit"               : torch.zeros(6),
                                   "round_active_flag"          : torch.zeros(1)
                                   } for _ in range(self.current_round)}
       
        if self.window is None and self.render_mode in ["human","human_interactive"]:
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption(f'Magic Man Round {self.current_round}')
            
            self.font = pygame.font.SysFont("monospace", 20)
            self.window = pygame.display.set_mode(self.window_size)
            
            
        if self.clock is None and self.render_mode in ["human","human_interactive"]:
            self.clock = pygame.time.Clock()

            self.canvas = pygame.Surface(self.window_size)
            self.canvas.fill((255, 255, 255))
            
            self.hand_pos_dict = mm_render.get_hand_pos_dict(*self.window_size)
            self.center_pos_dict = mm_render.get_center_pos_dict(*self.window_size)
            self.bid_pos_dict = mm_render.get_bid_pos_dict(*self.window_size)
           
       
        self.give_out_cards()
        obs,r,done,info = self.init_bid_step()
   
        return obs

    

    def render(self,placeholder_arg=None,last_step=False):
        #could just move the whole render function outside of the gym env
        #like mm_render.render(self.last_step)

        if self.render_mode in ["human","human_interactive"] :
            # The following line copies our drawings from `canvas` to the visible window
            
            self.window.blit(self.canvas, self.canvas.get_rect())
            pygame.event.pump()
            
            for player in self.players:
                mm_render.render_hand_cards(player,self.hand_pos_dict,self.card_sprite_dict,self.window)
                
                bid_loc = self.bid_pos_dict[player.table_idx]
                label = self.font.render(f"[{player.round_suits}/{player.current_bid}]", 1, (0,0,0))
                self.window.blit(label, bid_loc)
            
            for card_idx,card in enumerate(self.turn_cards):
                player_table_idx = self.players[card_idx].table_idx
                card_loc = self.center_pos_dict[player_table_idx]
                card_surface = self.card_sprite_dict[str(card)].surface
                self.window.blit(card_surface,dest=card_loc)         
            
            for player_table_idx,card in zip(self.next_round_idx_list,self.last_turn_cards):
                card_loc_x,card_loc_y = self.center_pos_dict[player_table_idx]
                card_loc = (card_loc_x+100,card_loc_y+100)
                card_surface = self.card_sprite_dict[str(card)].surface
                self.window.blit(card_surface,dest=card_loc)
                
            
            
            
            pygame.display.update()

            
            
            if self.render_mode=="human_interactive":
                interactive_action = self.activate_cards_buttons(self,deck,last_step)
                return interactive_action

            else:
                raise UserWarning("Do not attempt to render the environment when it is not in render mode!")
            
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()    
                       
    def starting_player(self,starting_player):
        self.players.rotate(  -self.players.index(starting_player) )
 
    def give_out_cards(self):
        self.round_deck = deck.deck.copy()
        random.shuffle(self.round_deck) 
   
        if self.render_mode in ["human","human_interactive"]:
            self.card_sprite_dict = {str(card):mm_render.CardSprite(card) for card in deck.deck}
       
                
        for _ in range(self.current_round):
            for player in self.players:
                card = self.round_deck.pop(-1)
                player.cards_obj.append(card)
                
                if self.render_mode in ["human","human_interactive"]:
                    self.card_sprite_dict[str(card)].owned_by(player.table_idx)
                    
 
    def init_bid_step(self,active_bid=False):

        self.state = "BID"     

        self.bids = torch.full(tuple([self.n_players]),float(round(self.current_round/self.n_players)))#bids that are not yet given become 0 (bids are normalized so 0 is the expected bid)

        self.bid_idx = 0
        self.done = False
        self.r,self.info = 0,{player.name:0 for player in self.players}

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
            self.r,self.info = 0,{player.name:0 for player in self.players}
            self.bid_idx += 1
        
        while self.bid_idx <= (self.n_players-1): # order is relevant

            norm_bids = self.bids/self.current_round
            
            player = self.players[self.bid_idx]
            player.bid_obs = {"norm_bids"                   : norm_bids,
                               "all_bid_completion"         : self.all_bid_completion,
                               "player_idx"                 : torch.zeros(self.n_players),
                               "player_self_bid_completion" : torch.zeros(1),
                               "n_cards"                    : torch.zeros(self.max_rounds),
                               "played_cards"               : torch.zeros((self.n_players,60)),
                               "legal_cards_tensor"         : torch.zeros(60),
                               "cards_tensor"               : torch.zeros(60),
                               "current_suit"               : torch.zeros(6),
                               }

            player.bid_obs["player_idx"][self.bid_idx] = 1
                                       
            player.bid_obs["n_cards"][len(player.cards_obj)-1] = 1 #how many cards there are in his hand
            
            for card in player.cards_obj:
                card_idx = deck.deck.index(card)
                player.bid_obs["cards_tensor"][card_idx] = 1
                player.bid_obs["legal_cards_tensor"][card_idx] = 1

            if isinstance(player,AdversaryPlayer):
                
                player.current_bid = round(self.current_round*player.bid(player.bid_obs))
                self.bids[self.bid_idx] = player.current_bid
                self.bid_idx += 1

            elif isinstance(player,TrainPlayer):
                if active_bid:
                    raise UserWarning (f"Active bidding is not intended for Train Player --> active_bid flag is {active_bid}")
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
                print(f"Train Player action: {deck.deck[action]}")
            played_card = deck.deck[action]
            self.turn_cards.append(played_card)
            self.train_player.cards_obj.remove(played_card)
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
                                   "legal_cards_tensor"         : torch.zeros(60),
                                   "cards_tensor"               : torch.zeros(60),
                                   "current_suit"               : torch.zeros(6),
                                   "round_active_flag"          : torch.ones(1)
                                   }
                player.turn_obs["norm_bids"] = self.bids/self.current_round
                player.turn_obs["player_idx"][self.turnorder_idx] = 1
                                
                player_self_bid_completion = torch.tanh(torch.tensor(player.round_suits-player.current_bid))
                player.turn_obs["player_self_bid_completion"]=torch.unsqueeze(player_self_bid_completion,0)
               
                
                player.turn_obs["n_cards"][len(player.cards_obj)-1] = 1 #how many cards there are in his hand

                self.current_suit = torch.zeros(6)
                self.current_suit_idx = deck.legal(self.turn_cards,player.cards_obj,self.trump)
                self.current_suit[self.current_suit_idx] = 1
                player.turn_obs["current_suit"]=self.current_suit
                
                for card in player.cards_obj:
                    card_idx = deck.deck.index(card)
                    player.turn_obs["cards_tensor"][card_idx] = 1
                    if card.legal:
                        player.turn_obs["legal_cards_tensor"][card_idx] = 1
                        
                        
                self.action_mask=player.turn_obs["legal_cards_tensor"]
                assert sum(self.action_mask)>0 or not player.cards_obj, f"{player} has no valid moves: {player.cards_obj}"


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
                    if self.flat_out:
                        flat_round_obs = self.get_flat(player_obs)
                        player_obs = flat_round_obs
                    net_out = player.play(player_obs,self.action_mask)
                    action_idx = net_out
                    
                    played_card = deck.deck[action_idx]
                    if self.verbose:
                        print(f"{player} Intended Card: {played_card}")

                    player.cards_obj.remove(played_card)
                    self.turn_cards.append(played_card)
                    self.turnorder_idx +=1
                    #--> go round the circle again
                    
                elif isinstance(player,TrainPlayer):
                    if self.verbose_obs:
                        print(f"Train Player Observation: {player_obs}")
                        print(f"Train Player action mask: {self.action_mask}")
                    dict_round_obs = player_obs

                    return dict_round_obs, self.r, self.done, self.info
                    
                else:
                    raise UserWarning (f"Player is {type(player)} not instance of either AdversaryPlayer or TrainPlayer")            
                
            if self.turnorder_idx == (self.n_players):
                
                self.compute_final_turn_observations()
                
                deck.turn_value(self.turn_cards,self.current_suit_idx) #turn value of the players cards    
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


    def compute_final_turn_observations(self):
        
        norm_bids = self.bids/self.current_round
            
        for player_idx,player in enumerate(self.players):
                
            player.turn_obs = {"norm_bids"                  : norm_bids,
                               "all_bid_completion"         : self.all_bid_completion,
                               "player_idx"                 : torch.zeros(self.n_players),
                               "player_self_bid_completion" : torch.zeros(1),
                               "n_cards"                    : torch.zeros(self.max_rounds),
                               "played_cards"               : torch.zeros((self.n_players,60)),
                               "legal_cards_tensor"         : torch.zeros(60),
                               "cards_tensor"               : torch.zeros(60),
                               "current_suit"               : torch.zeros(6),
                               "round_active_flag"          : torch.zeros(1)
                               }
                
            player.turn_obs["player_idx"][player_idx] = 1
                                
            player_self_bid_completion = torch.tanh(torch.tensor(player.round_suits-player.current_bid))
            player.turn_obs["player_self_bid_completion"] = torch.unsqueeze(player_self_bid_completion,0)
                

            player.turn_obs["n_cards"][len(player.cards_obj)-1] = 1 #how many cards there are in his hand

            self.current_suit = torch.zeros(6)
            self.current_suit_idx = deck.legal(self.turn_cards,player.cards_obj,self.trump)
            self.current_suit[self.current_suit_idx] = 1
            player.turn_obs["current_suit"]=self.current_suit
            
            # no need to refresh the legal cards
            # all players have played a card so legal is irrelevant

            for card_idx in range(len(self.turn_cards)):
                played_card = self.turn_cards[card_idx]
                player.turn_obs["played_cards"][card_idx][deck.deck.index(played_card)] = 1
            if self.render_mode in ["human","human_interactive"]:
                self.last_turn_cards = self.turn_cards
                self.next_round_idx_list = [self.players[card_idx].table_idx for card_idx,card in enumerate(self.turn_cards)]

            
            player.round_obs[self.turn_idx] = player.turn_obs 

    def conclude_step(self):
        assert self.turn_idx == self.current_round, (f"Turn index is {self.turn_idx} and should be equal to Current Round [{self.current_round}]")       
        self.state="CONCLUDE"
        for player in self.players:

            if player.current_bid == player.round_suits:
                round_reward = player.current_bid + 2 #ten point for every turn won and 20 for guessing right
            else:
                round_reward =  -abs(player.current_bid-player.round_suits) #ten points for every falsly claimed suit
            
            player.round_r = round_reward
            
            if isinstance(player,TrainPlayer):
                self.r = round_reward
                self.done = True
                self.info[player.name] = player.current_bid-player.round_suits
                self.info[player.name] = player.current_bid-player.round_suits
                
            elif isinstance(player,AdversaryPlayer):
                self.info[player.name] = player.current_bid-player.round_suits
                
        for player in self.players:
            player.clean_hand() #at this point all hands should be empty anyways
            self.all_bid_completion = torch.zeros(self.n_players)
            
        final_obs = self.train_player.round_obs

        return final_obs,self.r,self.done,self.info





if __name__ == "__main__":
    current_round=8

    env = gym.make("MagicMan-v0",adversaries='trained',current_round=current_round,render_mode='human_interactive',verbose=False)#,current_round=2,verbose=0,verbose_obs=0)
    env.seed(None)

    r_list = []
    info_mean = None
    
    for player in env.players:
        player.total_score = 0
        print(f"{player.name} total score is {player.total_score} points.\n")
    
    for _ in range(100):
        done = False
        obs = env.reset()
        round_idx=0
        while not done:
            action = env.render()
            obs, r, done, info = env.step(action)
            round_idx+=1
            
        env.render(last_step=True)
        print(f"Round {_+1}")
        for player in env.players:
            player.total_score +=player.round_r*10
            print(f"{player.name} recieves {player.round_r*10} points.")
            print(f"{player.name} total score is {player.total_score} points.")
            print(f"{player.name} average score is {round(player.total_score/(_+1),2)} points.\n")
        print("\n")    

    env.close()

