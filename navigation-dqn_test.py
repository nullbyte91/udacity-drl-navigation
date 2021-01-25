from unityagents import UnityEnvironment
import numpy as np
import gym
from dqn_agent import Agent
import matplotlib.pyplot as plt
import random
import torch
import numpy as np
from collections import deque

env = UnityEnvironment(file_name="/home/nullbyte/Desktop/mygit/udacity_rl/Banana_Linux/Banana.x86_64")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space 
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)

agent = Agent(state_size=state_size, action_size=action_size, seed=0)

# Load the saved model
state_dict = torch.load('checkpoint.pth')
agent.qnetwork_local.load_state_dict(state_dict)
agent.qnetwork_local.eval()

n_episodes = 5
max_t = 300

# Train the Agent
scores = []       # list containing scores from each episode
scores_mean = []  # list the mean of the window scores
for i_episode in range(1, n_episodes+1):
    state = env.reset(train_mode=False)[brain_name].vector_observations[0]
    score = 0
    for t in range(max_t):
        action = agent.act(state)
        env_info = env.step(action)[brain_name]        # send the action to the environment
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # see if episode has finished
        state = next_state
        score += reward
        print('\rEpisode {}\tScore: {:.2f}'.format(i_episode, score), end="")
        if done:
            break 

    scores.append(score)                                # save most recent score
    scores_mean.append(np.mean(scores))

    print('\rEpisode {}\tScore: {:.2f}'.format(i_episode, scores[-1]))

# Close the environment
env.close()