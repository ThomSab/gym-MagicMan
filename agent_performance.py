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
        for _ in range(100):
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
        
    all_rounds_score = [
        [0.2013 ,0.1967 ,0.1845 ],
        [0.0227 ,0.0278 ,0.0416 ],
        [-0.0845,-0.0754,-0.0685],
        [-0.1629,-0.1598,-0.1541],
        [-0.2318,-0.2518,-0.2263],
        [-0.3021,-0.3213,-0.3149],
        [-0.3739,-0.3957,-0.3929],
        [-0.4433,-0.4497,-0.4739],
        [-0.5326,-0.5326,-0.5202],
        [-0.6161,-0.5968,-0.5753],
        [-0.6845,-0.67  ,-0.6798],
        [-0.7375,-0.729,-0.7282 ],
        [-0.8088,-0.8082,-0.8107]]
    mean_score = [np.mean(score) for score in all_rounds_score]

    plt.plot(mean_score)
    plt.show()