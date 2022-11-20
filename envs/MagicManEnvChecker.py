import stable_baselines3
from stable_baselines3.common import env_checker
from MagicMan_env import MagicManEnv
import MagicNet as net
import MagicManDeck  as deck
from MagicManPlayer import AdversaryPlayer, TrainPlayer


if __name__ == "__main__":

    demo_train_player = TrainPlayer()

    adversary_players = [AdversaryPlayer(net.PlayNet(),net.BidNet()) for _ in range(3)]
    env = MagicManEnv(demo_train_player, adversary_players)
    
    """
    Players need to go inside the environment.
    
    
    """
    
    env.current_round = 2
    
    env_checker.check_env(env)