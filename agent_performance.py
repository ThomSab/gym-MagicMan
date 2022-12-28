import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

from gym_MagicMan.envs.MagicMan_env import *

if __name__ == "__main__":

    for round_ in range(8,10):
        current_round=round_

        env = gym.make("MagicMan-v0",adversaries='jules',current_round=current_round)#,current_round=2,verbose=0,verbose_obs=0)
        #env = gym.wrappers.FlattenObservation(env)

        r_list = []
        info_mean = None
        for _ in range(1000):
            print(_)
            done = False
            obs = env.reset()
            round_idx=0
            while not done:
                
                assert sum(obs[round_idx]["legal_cards_tensor"])>0,f"legal cards tensor empty: {obs[turn_idx]['legal_cards_tensor']}"
                legal_cards = torch.where(obs[round_idx]["legal_cards_tensor"]==1)[0]
                action = random.choice(legal_cards)
                obs, r, done, info = env.step(action)
                round_idx+=1
                
            if not info_mean:
                info_mean = info
                for key,val in info.items():
                    info_mean[key] = [val]
            else:
                for key,val in info.items():
                    info_mean[key].append(val)
        
        print(round_)
        for player,scores in info_mean.items(): 
            print(player)
            print(np.mean(scores))
            #plt.hist(scores,bins=list(range(min(scores),max(scores))),align='mid')
            #plt.title("player.current_bid-player.round_suits")
            #plt.xlabel("<-- bid too low   |   bid too high --->")
            #plt.show()
        
