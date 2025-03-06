
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from monte_carlo_tree_search.policy_agent import *
from monte_carlo_tree_search.qtable import QTable
from monte_carlo_tree_search.conversation_env import *
from scipy import stats
import numpy as np
import torch
import os.path
import pandas as pd
import random
import tiktoken

# with mcts
def offline_train_q_function(conversation_starters, human, llm_agent, pretrained_q_function_name, timeout=100, search_depth=5):
    qfunction = DeepQFunction()
    for idx, conversation_starter in enumerate(conversation_starters):
        print("training index: ", idx)
        conversation_env = conversation_environment(human, llm_agent, conversation_starter, max_depth=search_depth)
        mcts = SingleAgentMCTS(conversation_env, qfunction, UpperConfidenceBounds())
        mcts.mcts(timeout=timeout)
        qfunction = mcts.qfunction
        if idx % 10 == 0:
            print("saving model...")
            torch.save(qfunction, pretrained_q_function_name + str(len(conversation_starters))) # save after each training
    return qfunction

# with just static convo
def offline_train_q_function_static_conversation(conversations, pretrained_q_function_name, reward_function, num, cycle=1, terminating_steps=10):
    qfunction = DeepQFunction()
    
    # how many times to iterate through dataset
    for num_cycle in range(cycle):
        
        # learn for each convo in conversations
        for idx, convo in enumerate(conversations[:num]):
            cumulative_reward = 0
            if len(convo) > 2:
                cumulative_reward = reward_function(convo[2:])
            state = conversation_state(convo[0], convo[0])
            state.depth = 1
            qfunction.update(state, convo[1], 0, 1, cumulative_reward)
            
            if idx % 10 == 0:
                print("saving model...")
                torch.save(qfunction, pretrained_q_function_name) # save after each a few training loop