import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from monte_carlo_tree_search.policy_agent import *
from monte_carlo_tree_search.qtable import QTable, DeepQSemanticFunction, ReplayBufferDeepQFunction
from monte_carlo_tree_search.conversation_env import *
from agent.Model import create_human_and_llm
from transformers import AutoTokenizer, BertModel, AutoModel
from transition_models.transition_model import TransitionModel, TransitionModelMOE, TransitionModelMDN
from transition_models.embedding_model import embedding_model_mistral, embedding_model_nomic, embedding_model_llama
from reward.Embedding_Length_Reward import Embedding_Length_Reward
from reward.Human_Length_Reward import Human_Length_Reward
from reward.Llama_2_Guard_Reward import Llama_2_Guard_Reward

import torch
from scipy import stats
import numpy as np
import torch
import os.path
import time
import random
import pandas as pd

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from datasets import load_from_disk
from itertools import repeat

import logging
[logging.getLogger(name).setLevel(logging.ERROR) for name in logging.root.manager.loggerDict if "transformers" in logging.getLogger(name).name.lower()]

# Parse command line arguments
parser = ArgumentParser()
parser.add_argument("--evaluation_data", help="evaluation_data", default="evaluation_starters_simple.txt")
parser.add_argument("--evaluation_start", help="start index for evaluation data", type=int, default=0)
parser.add_argument("--evaluation_end", help="end index for evaluation data", type=int, default=100)
parser.add_argument("--evaluation_depth",  help="number of sequential actions to evaluate", default=5)
parser.add_argument("--mcts_search_depth", help="mcts search depth; only applies to mcts approaches", default=8)
parser.add_argument("--mcts_time",  help="mcts search time budget", default=100)
parser.add_argument("--pretrained_q_function",  help="pre-learnt q function for heuristic or initialization", default="model_pretrained_qfn")
parser.add_argument("--result_file",  help="result_file_name", default="evaluation_results")
parser.add_argument("--agent",  help="agent type")
parser.add_argument("--embedding", default="llama")
parser.add_argument("--cuda_for_llm_reward",  help="cuda", default="0")
parser.add_argument("--cuda_for_q_embedding_transition",  help="cuda", default="0")
parser.add_argument("--transition_model_dir",  help="directory containing transition models", default="models/deterministic/")
parser.add_argument("--reward_decay",  default=0.9)
parser.add_argument("--trials",  help="trials", default=5)
parser.add_argument("--reward_func", help="reward", default="harmful")
parser.add_argument("--lr", default=0.0001)
parser.add_argument("--use_icl", default=False, action="store_true")
parser.add_argument("--use_descriptions", default=False, action="store_true")
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--config", default="agent/llm_config.yaml")
args = vars(parser.parse_args())
print("command-line args: ", args)

assert not(args["use_icl"] and args["use_descriptions"]), "Cannot use both icl and descriptions"

seed = args["seed"]

trials = int(args["trials"])
reward_func = args["reward_func"]
evaluation_output = args["result_file"]
evaluation_data = args["evaluation_data"]
evaluation_action_depth = int(args["evaluation_depth"])
runtime_mcts_search_depth = int(args["mcts_search_depth"])
runtime_mcts_timeout = int(args["mcts_time"])
agent_ = args["agent"]
embedding_type = args["embedding"]
transition_model_dir = args["transition_model_dir"]
if "mdn" in transition_model_dir.lower():
    TransitionModel_type = TransitionModelMDN
else:
    TransitionModel_type = TransitionModelMOE

# Cuda devices
convert = lambda s: int(s) if s.isdigit() else s
cuda_q_embedding = convert(args["cuda_for_q_embedding_transition"])
cuda_transition = convert(args["cuda_for_q_embedding_transition"])
cuda_llm = convert(args["cuda_for_llm_reward"])
cuda_reward = convert(args["cuda_for_llm_reward"])

reward_decay = float(args["reward_decay"])
lr = float(args["lr"])

start_index = args["evaluation_start"]
end_index = args["evaluation_end"]

# get the convo starters for evaluation
file_name = 'evaluation/' + str(evaluation_data)
human_prompts = repeat(None)
human_descriptions = repeat(None)
if os.path.isfile(file_name):
    with open(file_name) as f:
        evaluation_starters = f.readlines()
    end_index = min(end_index, len(evaluation_starters)) if end_index > 0 else len(evaluation_starters)
    evaluation_starters = evaluation_starters[start_index: end_index]
elif os.path.isdir(evaluation_data):
    tmp = load_from_disk(evaluation_data)
    evaluation_starters = [i[0].strip() for i in tmp["conversations"]]
    end_index = min(end_index, len(evaluation_starters)) if end_index > 0 else len(evaluation_starters)
    evaluation_starters = evaluation_starters[start_index: end_index]
    if args["use_icl"]:
        human_prompts = [[j.strip() for j in i[::2]] for i in tmp["conversations"]]
        human_prompts = human_prompts[start_index: end_index]
    if args["use_descriptions"]:
        human_descriptions = [[i] for i in pd.read_pickle(evaluation_data + "/descriptions.pkl")["filtered_description"]]
        human_descriptions = human_descriptions[start_index: end_index]

# create the llm and human simulator
cg = args["config"]
human_sim, human_eval, llm_agent = create_human_and_llm(config=cg, cuda=cuda_llm)

if reward_func == "length_human":
    reward_function = Embedding_Length_Reward(add_llm_length=False, device_map=cuda_reward)
if reward_func == "length_both":
    reward_function = Embedding_Length_Reward(add_llm_length=True, device_map=cuda_reward)
if reward_func == "harmful":
    reward_function = Llama_2_Guard_Reward(device_map=cuda_reward)

agents = []
agent_type = []

if agent_ == "zero_step_greedy":
    greedy_agent = GreedyAgent(zero_step_greedy_reward_generator(human_sim, reward_function.get_reward), llm_agent) # infer human's next response and choose best one
    agent_type.append(agent_)
    agents.append(greedy_agent)

if agent_ == "one_step_greedy":
    greedy_agent = GreedyAgent(one_step_greedy_reward_generator(human_sim, reward_function.get_reward), llm_agent) # infer human's next response and choose best one
    agent_type.append(agent_)
    agents.append(greedy_agent)
    
if agent_ == "random":
    random_agent = RandomAgent(llm_agent)
    agent_type.append(agent_)
    agents.append(random_agent)

if agent_ == "pure_offline":
    model = torch.load(args["pretrained_q_function"], map_location=torch.device(cuda_q_embedding))
    pure_offline_agent = OfflineAgent(model, llm_agent) # use pretrained q functon, don't do any mcts
    agent_type.append(agent_)
    agents.append(pure_offline_agent)
    
if agent_ == "pure_online":
    pure_online_mcts_agent = OnlineAgent(ReplayBufferDeepQFunction(alpha=lr, steps_update=50, cuda=torch.device(cuda_q_embedding)), runtime_mcts_search_depth, runtime_mcts_timeout, llm_agent, human_sim, reward_function, search_space="response_space", reward_decay=reward_decay) # use a brand new q function and do mcts during runtime
    agent_type.append(agent_)
    agents.append(pure_online_mcts_agent)

if agent_ == "offline_online_mixed":
    model = torch.load(args["pretrained_q_function"], map_location=torch.device(cuda_q_embedding))
    pretrained_offline_online_mcts_agent = OnlineAgent(model, runtime_mcts_search_depth, runtime_mcts_timeout, llm_agent, human_sim, reward_function) # use pretrained q function and perform mcts
    agent_type.append(agent_)
    agents.append(pretrained_offline_online_mcts_agent)

if agent_ == "semantic_online":
    embed_model=None
    dim = None

    if embedding_type == "llama":
        if reward_func == "harmful":
            model = reward_function.model
        else:
            model = None
        embed_model = embedding_model_llama(model=model, cuda=torch.device(cuda_q_embedding))
        dim = embed_model.output_dim

        if reward_func == "length_human":
            reward_function = Embedding_Length_Reward(add_llm_length=False)
        elif reward_func == "length_both":
            reward_function = Embedding_Length_Reward(add_llm_length=True)
        transition_model = TransitionModel_type(noise=0.005, cuda=cuda_transition, transition_model_dir=args["transition_model_dir"]) # need to convert to cuda. Now using CPU (does it matter?).
        semanticqfunction = DeepQSemanticFunction(dim=dim, alpha=lr, cuda=torch.device(cuda_q_embedding), steps_update=50) # more sophisticated Q function?
        pure_online_agent_semantic_agent = OnlineAgent(semanticqfunction, runtime_mcts_search_depth, runtime_mcts_timeout, llm_agent, human_sim, reward_function, search_space="semantic_space", transition_model=transition_model, embedding_model=embed_model) # online SEMANTIC space agent

    agent_type.append(agent_)
    agents.append(pure_online_agent_semantic_agent)

if agent_ == "semantic_exhaustive":
    if reward_func == "harmful":
        model = reward_function.model
    else:
        model = None
    embed_model = embedding_model_llama(model=model, cuda=torch.device(cuda_q_embedding))
    transition_model = TransitionModel_type(noise=0.005, cuda=cuda_transition, transition_model_dir=args["transition_model_dir"]) # need to convert to cuda. Now using CPU (does it matter?).
    pure_online_agent_semantic_agent = ExhastiveOnlineAgent(runtime_mcts_search_depth, runtime_mcts_timeout, llm_agent, human_sim, reward_function, search_space="semantic_space", reward_decay=reward_decay, transition_model=transition_model, embedding_model=embed_model) # online SEMANTIC space agent
    agent_type.append(agent_)
    agents.append(pure_online_agent_semantic_agent)

np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

# create the mdp environment for evaluation
evaluation_conversation_env = conversation_environment(human_eval, llm_agent, "", max_depth=evaluation_action_depth*2, reward_function=reward_function)
all_results = []
all_results.append(evaluation_starters)
all_results_dict = []
for agent,type in zip(agents, agent_type):
    results = {}
    start = time.time()
    result_row, convo_generated = run_evaluations(agent, type, evaluation_conversation_env, evaluation_starters, evaluation_action_depth, trials, context_list = human_prompts, human_descriptions = human_descriptions, results=results, index = list(range(start_index, end_index)), seed = seed, output_file=evaluation_output)
    all_results.append(result_row)
    time_taken = time.time()-start
    results["time_taken_for_agent_type"] = time_taken
    print("time taken for all trials:", time_taken)
    for starters in convo_generated:
        print("input conversation starter: ", starters, "\n")
        print("conversation generated: ", convo_generated[starters])
    all_results_dict.append(results)

import lz4.frame

with lz4.frame.open(evaluation_output+'_dump.pkl', 'wb') as f:
    pickle.dump(all_results_dict, f)

all_results = [list(i) for i in zip(*all_results)] # transpose
import csv

with open(evaluation_output+'.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(all_results)

import pickle 
with open(evaluation_output+'.pkl', 'wb') as f:
    pickle.dump(convo_generated, f)