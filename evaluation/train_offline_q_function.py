import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from monte_carlo_tree_search.policy_agent import *
from monte_carlo_tree_search.qtable import QTable
from monte_carlo_tree_search.conversation_env import *
from evaluation.train_q_function_helper import *
from scipy import stats
import numpy as np
import torch
import os.path
import pandas as pd
import random

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# Parse command line arguments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--data", help="data-set name. i.e name of the parquet file")
parser.add_argument("--number",  help="number of conversation starters")
args = vars(parser.parse_args())

data_path = args["data"]
n = int(args["number"])

# mcts
if data_path == "daily_dialogue":
    # create the llm and human simulator
    human, llm_agent = create_human_and_llm()

    # check if Q-function pretrained) exists, if not, train one offline with some conversation starters,
    pretrained_q_function_name = "trained_q_function_" + str(data_path)
    conversation_data = pd.read_parquet('daily_dialogue.parquet', engine='auto')
    conversation_starters = [x[0] for x in list(conversation_data['dialog'])]
    conversation_starters = random.choices(conversation_starters, k=n)

    pretraining_mcts_timeout = 500 # how long to run simulation
    pretraining_depth = 8 # how deep to run mcts

    q_function_offline_learnt = offline_train_q_function(conversation_starters, human, llm_agent, pretrained_q_function_name, timeout=pretraining_mcts_timeout, search_depth=pretraining_depth)
    torch.save(q_function_offline_learnt, pretrained_q_function_name + str(len(conversation_starters)))
    
if data_path == "daily_dialogue_static":
    
    pretrained_q_function_name = "trained_q_function_STATIC_" + str(data_path)
    conversation_data = pd.read_parquet('daily_dialogue.parquet', engine='auto')
    conversation_starters = [x for x in list(conversation_data['dialog'])]
    
    q_function_offline_learnt = offline_train_q_function_static_conversation(conversation_starters, pretrained_q_function_name, length_convo, n)
    
    torch.save(q_function_offline_learnt, pretrained_q_function_name)