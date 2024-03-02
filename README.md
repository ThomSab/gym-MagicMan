# gym-MagicMan

## Hello

I am ThomSab and I have been working on and off on a Deep Reinforcement Learning Project for the last five years now.

The idea is very simple. I wanted to code an agent for [Wizard](https://en.wikipedia.org/wiki/Wizard_(card_game)) the card game. If you have never heard of the game before: It is - in a nutshell- a trick-taking card game where you have to announce the amount of tricks that you win each round and gain points if you get this exact amount of tricks but lose points otherwise.

I have implemented the game in python as a [gymnasium](https://gymnasium.farama.org/) environment as well as a number of algorithms that I thought would be interesting to try. The current approach is to run the [Stable Baselines 3](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) implementation of a [Proximal Policy Optimization Algorithm](https://arxiv.org/abs/1707.06347) and have it play first against randomly acting adversaries and then have it play against other versions of itself. In theory, training would go on until the trained agent surpasses human level of play.

It's possible that an agent trained with PPO as I have been doing it, is not capable of achieving better-that-human performance in Wizards.
But there is a number of things that I have thought of that could still bring some hope:
 - Pre-Training the Agent on human data. Possible but I haven't looked into where I could acquire data like this.
 - There might be a better way to pass information from the environment to the agent. This might be a bit harder to explain so I'll elaborate when I write a more detailed post.
 - Actual literature research - I have not seriously looked into machine learning literature on trick-taking card games so there might be some helpful publications on this topic. 


## Requirements

- python 3.8 or later
- Windows 10 / 11
- Not sure about MacOs or Linux-based OS

## Required Modules
- numpy
- torch
- pygame
- gymnasium
- stable-baselines3
- sb3-contrib

## Installation
Inside gym-MagicMan directory open shell and execute:

 " py -m pip install -e . "
 
(-editable flag is only necessary if you want to edit the code after install)

If you are interested in the code or the project and have trouble installing it I would be happy to help!  Its a good way to make the install guide more inclusive.
 
## Run
Execute "MagicMan_env.py" inside
 
..\gym-MagicMan\gym_MagicMan\envs\
