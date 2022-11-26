import stable_baselines3
from stable_baselines3.common import env_checker
from MagicMan_env import MagicManEnv
import MagicManDeck  as deck
from MagicManPlayer import AdversaryPlayer, TrainPlayer


if __name__ == "__main__":

    env = MagicManEnv(adversaries='random',current_round=2)
    env_checker.check_env(env)