# gym-MagicMan

Magic Man is a Deep Reinforcement Learning Project that I have been working on and off on since maybe as early as 2020. Since then the approach has changed a number of times. I have for example applied a genetic algorithm ( Neuro-Evolution of Augmenting Topologies aka. NEAT) to the problem. 

The approach is straight forward. I want to implement an agent for [Wizard](https://en.wikipedia.org/wiki/Wizard_(card_game)) the card game. If you have never heard of the game before: It is - in a nutshell- a trick-taking card game where you have to predict the amount of tricks that you win each round. Players are rewarded points when they manage to predict the exact amount of tricks but lose points if the prediction is wrong. The goal of the game is therefore not to gain as many tricks as possible as is the case for the majority of trick-taking card games like for example Doppelkopf, but instead to assess the outcome of the game corectly.

I have implemented the game in python as a [gymnasium](https://gymnasium.farama.org/) environment as well as a number of algorithms that I thought would be interesting to try. The current approach is to run the [Stable Baselines 3](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) implementation of a [Proximal Policy Optimization Algorithm](https://arxiv.org/abs/1707.06347) and have it play first against randomly acting adversaries and then have it play against other versions of itself. Increase in the quality of the agents play slows down over time and is not likely to reach the niveau of skill that is comparable to humans.

The image below show the average score over the duration of training. Human average score is around 20, whereas these agents will likely never go beyond 2.

<div>
<img src="score curves.png" alt="training progress" width="1000"/>
</div>

Simply using PPO without any form of pre-training appears to be incapable of achieving better-that-human performance in Wizards.
There is however, a number of things that bear promise in my opinion:
 - Pre-training the agent on **human data**. Its possible to play Wizards on online platforms. Using this data to pre-train an agent would set it up better that starting from a randomly initiated agent. Agents like AlphaGo have been developed in a similar fashion.
 - Train an encoder for the environment. Learning an **embedding for the environment** would provide a better foundation for further training. This can be done in similar ways that are similar to the methods used to train BERT encoders i.e. Masked State Prediction, Next-State Prediction or Contrastive Learning.
 - **Structured literature research** - I have not systematically read through the literature on this topic, there is an abundance of precedence on this topic. The project itself has been to me more of a trying-it-by-myself approach with an emphasis on tinkering rather that actual research and development.


## Requirements

- python 3.8 or later
- Windows 10 / 11
- MacOs or Linux-based OS havent been tested

## Required Modules
- numpy
- torch
- pygame
- gymnasium
- stable-baselines3
- sb3-contrib

## Installation

1. Clone the repo
   ```sh
   git clone https://github.com/ThomSab/gym-MagicMan.git
   ```
2. Install modules
   ```sh
   py -m pip install module_name
   ```
3. Inside gym-MagicMan directory open shell and execute
   ```sh
   py -m pip install -e . 
   ```

   The editable (-e) flag is only necessary if you want to edit the code after installation.

If you are interested in the code or the project and have trouble installing it I would be happy to help!  Its a good way to make the install guide more inclusive.
 
## Run a Playable version of the Game
It possible to play Wizards against the agents in a very minimal interface. Much of the learned behavior of the agents is similar to natural strategies often employed by human players, like taking a trick with a wizard card when a high card of a non-trump suit is played.

To run the interface, go to ```..\gym-MagicMan\gym_MagicMan\envs\``` and execute 

```sh
py MagicMan_env.py
```


 

