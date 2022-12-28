import pygame
import os
from gym_MagicMan.envs.utils.MagicManDeck import Card

try:
    os.chdir(r"C:\Users\jasper\Documents\LINZ\Semester_III\SEMINAR\gym-MagicMan\gym_MagicMan\envs\imgs")
except:
    raise UserWarning("chdir did not work")

class CardSprite(pygame.sprite.Sprite):
    def __init__(self,card):
        super(CardSprite, self).__init__()
        self.card_img = pygame.image.load(f"{str(card)}.png")
        self.surface = self.card_img.convert()
        self.backside = pygame.image.load("Backside.png").convert()
        self.rect = self.surface.get_rect()
        
    def owned_by(self,player_idx):
        angle = player_idx*-90
        self.surface = pygame.transform.rotate(self.card_img, angle)

def reset_cards(sprite_deck):
    for card_sprite in sprite_deck:
        card_sprite.surface = card_sprite.card_img.convert()
        

def get_hand_positions_dict(canvas_size_x,canvas_size_y=None):
    if not canvas_size_y:
        canvas_size_y = canvas_size_x
    
    
    player_0_loc={_:(canvas_size_x*.1+_*canvas_size_x/15,canvas_size_y-80                   )   for _ in range(15)}
    player_1_loc={_:(0                                  ,canvas_size_y*.2+_*canvas_size_y/15)   for _ in range(15)}
    player_2_loc={_:(canvas_size_x*.2+_*canvas_size_x/15,0                                  )   for _ in range(15)}
    player_3_loc={_:(canvas_size_x-80                   ,canvas_size_y*.2+_*canvas_size_y/15)   for _ in range(15)}

    hand_positions_dict = {0:player_0_loc,
                           1:player_1_loc,
                           2:player_2_loc,
                           3:player_3_loc
                           }
    return hand_positions_dict
    
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
        card_surface = card_sprite_dict[str(card)].surface
        window.blit(card_surface,dest=hand_positions_dict[player.table_idx][card_idx])
