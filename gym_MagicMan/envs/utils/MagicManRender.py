import pygame
import os
from gym_MagicMan.envs.utils.MagicManDeck import Card
from gym_MagicMan.envs.utils.MagicManPlayer import TrainPlayer,AdversaryPlayer


class CardSprite(pygame.sprite.Sprite):
    def __init__(self,card):
        super(CardSprite, self).__init__()
        self.card_img = pygame.image.load(f"imgs\{str(card)}.png")
        self.surface = self.card_img.convert()
        self.back_img = pygame.image.load("imgs\Backside.png")
        self.backside = self.back_img.convert()
        self.rect = self.surface.get_rect()
        
    def owned_by(self,player_idx):
        angle = player_idx*-90
        self.surface = pygame.transform.rotate(self.card_img, angle)
        self.backside = pygame.transform.rotate(self.back_img, angle)

def reset_cards(sprite_deck):
    for card_sprite in sprite_deck:
        card_sprite.surface = card_sprite.card_img.convert()
        

def get_hand_pos_dict(canvas_size_x,canvas_size_y=None):
    if not canvas_size_y:
        canvas_size_y = canvas_size_x
    
    
    player_0_loc={_:(canvas_size_x*.1+_*canvas_size_x/20,canvas_size_y-80                   )   for _ in range(15)}
    player_1_loc={_:(0                                  ,canvas_size_y*.2+_*canvas_size_y/20)   for _ in range(15)}
    player_2_loc={_:(canvas_size_x*.2+_*canvas_size_x/20,0                                  )   for _ in range(15)}
    player_3_loc={_:(canvas_size_x-80                   ,canvas_size_y*.2+_*canvas_size_y/20)   for _ in range(15)}

    hand_positions_dict = {0:player_0_loc,
                           1:player_1_loc,
                           2:player_2_loc,
                           3:player_3_loc
                           }
    return hand_positions_dict
    
def get_bid_pos_dict(canvas_size_x,canvas_size_y=None):

    player_0_loc=(canvas_size_x*.1-canvas_size_x/15,canvas_size_y-80                 )
    player_1_loc=(0                                ,canvas_size_y*.2-canvas_size_y/15)
    player_2_loc=(canvas_size_x*.2-canvas_size_x/15,0                                )
    player_3_loc=(canvas_size_x-80                 ,canvas_size_y*.2-canvas_size_y/15)

    bid_positions_dict = {0:player_0_loc,
                          1:player_1_loc,
                          2:player_2_loc,
                          3:player_3_loc
                          }
    
    return bid_positions_dict

    
def get_center_pos_dict(canvas_size_x,canvas_size_y=None):
    if not canvas_size_y:
        canvas_size_y = canvas_size_x
        
    center_pos_dict = {0:(canvas_size_x/2,canvas_size_y/2+25),
                       1:(canvas_size_x/2-45,canvas_size_y/2),
                       2:(canvas_size_x/2,canvas_size_y/2-45),
                       3:(canvas_size_x/2+25,canvas_size_y/2)
                       }
                           
    return center_pos_dict
        

def render_hand_cards(player,hand_positions_dict,card_sprite_dict,window):
    
    for card_idx,card in enumerate(player.cards_obj):
        card_sprite = card_sprite_dict[str(card)] 
        
        pos = hand_positions_dict[player.table_idx][card_idx]
        if isinstance(player,TrainPlayer):
            card_sprite.rect = card_sprite.surface.get_rect().move(*pos)
            window.blit(card_sprite.surface,dest=pos)
            
        else:
            card_sprite.rect = card_sprite.backside.get_rect().move(*pos)
            window.blit(card_sprite.backside,dest=pos)

def activate_cards_buttons(self,deck,last_step):
    if last_step:
        return None
    while True:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                for card in self.train_player.cards_obj:
                    card_sprite = self.card_sprite_dict[str(card)]
                    if card_sprite.rect.collidepoint(pygame.mouse.get_pos()):
                        intended_card_idx = deck.deck.index(card)
                        self.action_mask

                        if self.action_mask[intended_card_idx]:
                            return intended_card_idx
                        else:
                            print(f"{str(card)} is not legal")
                            #I'm working on my excessive denting sins.
                            #I swear.
                          
    self.clock.tick(self.metadata["render_fps"])