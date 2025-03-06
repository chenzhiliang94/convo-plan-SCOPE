from monte_carlo_tree_search.qtable import QTable, DeepQFunction
from monte_carlo_tree_search.single_agent_mcts import SingleAgentMCTS
from monte_carlo_tree_search.conversation_env import conversation_environment, conversation_state
from monte_carlo_tree_search.semantic_conversation_env import semantic_conversation_environment, conversation_semantic_state
from monte_carlo_tree_search.ucb import UpperConfidenceBounds

from agent.Conversation import Conversation
from agent.Model import Model
from reward.Base_Reward import Base_Reward
from reward.Llama_2_Guard_Reward import Llama_2_Guard_Reward
from reward.Embedding_Length_Reward import Embedding_Length_Reward

from transition_models.transition_model import TransitionModelMOE
import random
import copy
import numpy as np
from scipy import stats
import torch
from typing import List

from abc import abstractmethod

from time import time

from tqdm import tqdm
import itertools

class LearntAgent():
    def __init__(self) -> None:
        pass
    @abstractmethod
    def generate_action(self, state : conversation_state, results = {}, seed=None, **kwargs):
        pass
    def seed(self, seed):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

# an agent that just greedily returns the best action during runtime. Infer next response by human and choose greedily.
class RandomAgent(LearntAgent):
    
    def __init__(self, action_generator : Model) -> None:
        self.action_generator = action_generator

    def generate_action(self, state : conversation_state, results = {}, seed=None, **kwargs):
        self.seed(seed)
        possible_actions = self.action_generator.sample_actions(state.conversation)
        print("possible actions random agent proposed: ", possible_actions)

        start_time = time()
        self.seed(seed)
        best_action_index = random.randint(0, len(possible_actions)-1)
        best_action = possible_actions[best_action_index]
        
        results["possible_actions"] = possible_actions
        results["selected_action_index"] = best_action_index
        print(f"action selected by random agent: {best_action}\ttime taken: {time()-start_time}")
        return best_action
    
# an agent that just greedily returns the best action during runtime. Infer next response by human and choose greedily.
class GreedyAgent(LearntAgent):
    
    def __init__(self, reward_calculator, action_generator : Model) -> None:
        self.reward_calculator = reward_calculator
        self.action_generator = action_generator

    def generate_action(self, state : conversation_state, results = {}, seed=None, **kwargs):
        self.seed(seed)
        possible_actions = self.action_generator.sample_actions(state.conversation) # maybe add an argument to choose number of actions
        start_time = time()
        best_action = self.reward_calculator.select(state, possible_actions, results = results)
        print(f"action selected by greedy agent: {best_action}\ttime taken: {time()-start_time}")

        return best_action
# greedy reward functions to be used in GreedyAgent
def len_reward_function(human_response):
    return len(human_response)

class zero_step_greedy_reward_generator():
    def __init__(self, human_agent : Model, reward_function) -> None:
        self.human = human_agent
        self.reward_function = reward_function
    
    # greedy reward: infer multiple human responses. take average reward from them.
    def select(self, state : conversation_state, possible_actions, results = {}):
        print("selecting greedy action...")
        convo = state.conversation
        print("current state: ", convo)
        action_reward = []
        for action in possible_actions:
            greedy_reward = self.reward_function(convo, action, None)
            print("one step greedy lookahead reward: ", greedy_reward)
            action_reward.append(greedy_reward)
        best_action_idx = np.argmax(action_reward)
        best_action = possible_actions[best_action_idx]
        # print("selected greedy reward: ", best_action)

        results["possible_actions"] = possible_actions
        results["possible_actions_reward"] = action_reward
        results["selected_action_index"] = best_action_idx
        return best_action
    
class one_step_greedy_reward_generator():
    def __init__(self, human_agent : Model, reward_function) -> None:
        self.human = human_agent
        self.reward_function = reward_function
    
    # greedy reward: infer multiple human responses. take average reward from them.
    def select(self, state : conversation_state, possible_actions, results = {}):
        print("selecting greedy action...")
        convo = state.conversation
        print("current state: ", convo)
        action_reward = []
        for action in possible_actions:
            print("candidate action: ", action)
            human_responses = self.human.sample_actions(convo + action)
            reward_to_be_averaged = []
            for response in human_responses:
                print("one step greedy lookahead reward: ", self.reward_function(convo, action, response))
                reward_to_be_averaged.append(self.reward_function(convo, action, response))
            print("mean greedy one step reward: ", np.mean(reward_to_be_averaged))
            action_reward.append(np.mean(reward_to_be_averaged))
        best_action_idx = np.argmax(action_reward)
        best_action = possible_actions[best_action_idx]
        print("selected greedy reward: ", best_action)
        results["possible_actions"] = possible_actions
        results["possible_actions_reward"] = action_reward
        results["selected_action_index"] = best_action_idx
        return best_action
            
# An agent with a pretrained Q function used to find best action during runtime. No searching is done.
class OfflineAgent(LearntAgent):
    
    def __init__(self, qfunction : DeepQFunction, llm_agent : Model) -> None:
        self.qfunction = qfunction
        self.llm_agent = llm_agent

    def generate_action(self, state : conversation_state, results = {}, seed=None, **kwargs):
        self.seed(seed)
        possible_actions = self.llm_agent.sample_actions(state.conversation) # maybe add an argument to choose number of actions
        start_time = time()
        qs = self.qfunction.get_qs(state, possible_actions)
        best_action_index = np.argmax(qs)
        best_action = possible_actions[best_action_index]
        
        results["possible_actions"] = possible_actions
        results["possible_actions_reward"] = qs
        results["selected_action_index"] = best_action_index
        print(f"action selected by offline agent: {best_action}\ttime taken: {time()-start_time}")
        return best_action

# An agent which performs MCTS during runtime. Takes in a Q functon during initialization (possibly pretrained)
class OnlineAgent(LearntAgent):
    
    def __init__(self, qfunction : DeepQFunction, search_depth, mcts_time_limit, llm_agent : Model, human_simulator, reward_function_for_mcts, search_space="response_space", reward_decay=1.0, terminating_heuristic_q_function="get_last_action", transition_model=None, embedding_model=None) -> None:
        self.search_depth = search_depth
        self.mcts_time_limit = mcts_time_limit
        self.llm_agent = llm_agent
        self.human_simulator = human_simulator
        self.qfunction = qfunction
        self.original_qfunction = copy.deepcopy(qfunction)
        self.terminating_heuristic_q_function = terminating_heuristic_q_function
        self.reward_function_for_mcts = reward_function_for_mcts
        self.search_space = search_space
        self.reward_decay = reward_decay
        self.transition_model = transition_model
        self.embedding_model = embedding_model
        if isinstance(self.reward_function_for_mcts, Llama_2_Guard_Reward):
            self.reward_from_embedding = self.reward_function_for_mcts.get_safe_prob_from_embedding
        elif isinstance(self.reward_function_for_mcts, Embedding_Length_Reward):
            self.reward_from_embedding = lambda x: self.reward_function_for_mcts.model(x).detach().cpu()
    
    def generate_action(self, state : conversation_state, results = {}, seed=None, **kwargs):
        evaluation_conversation_env = conversation_environment(self.human_simulator, self.llm_agent, state.conversation, max_depth=self.search_depth, reward_function=self.reward_function_for_mcts)
        print("generating action in realtime...")
        if self.search_space=="response_space":
            conversation_env = conversation_environment(self.human_simulator, self.llm_agent, state.conversation, max_depth=self.search_depth, reward_function=self.reward_function_for_mcts)
        elif self.search_space=="semantic_space":
            conversation_env = semantic_conversation_environment(embedding_model=self.embedding_model, transition_model=self.transition_model, initial_state=state.conversation, max_depth=self.search_depth, reward_function=self.reward_function_for_mcts)
            conversation_env.initial_actions_asked = True
            # get initial action, change to semantics, and store it!
            self.seed(seed)
            possible_starting_actions = evaluation_conversation_env.get_actions(state) # generate initial actions
            start_time = time()
            starting_convo_semantics = self.embedding_model.embed(state.conversation).cpu().detach().numpy()
            action_semantics = []
            action_rewards = []
            for action in possible_starting_actions:
                concatenated_convo = state.conversation + action # conversation of initial state + action, in string form
                output = self.embedding_model.embed(concatenated_convo) # embedding form
                reward = self.reward_from_embedding(output)
                action_rewards.append(reward)
                action_semantic = tuple(output.cpu().detach().numpy()) # convert to tuple
                action_semantic = tuple([x1-x2 for x1,x2 in zip(list(action_semantic),list(starting_convo_semantics))]) # get the difference (action in semantic form)
                action_semantics.append(action_semantic)
    
            results["greedy_rewards"] = action_rewards
            results["greedy_action_index"] = np.argmax(action_rewards)
            

            conversation_env.state_to_action_map[tuple(starting_convo_semantics)] = action_semantics # store the initial action, so later we can use it.
                

        print("performing MCTS search...")
        self.seed(seed)
        mcts = SingleAgentMCTS(conversation_env, self.qfunction, UpperConfidenceBounds(), terminating_heuristic_q_function=self.terminating_heuristic_q_function)
        mcts.mcts(timeout=self.mcts_time_limit, seed=seed)
        self.qfunction = mcts.qfunction # qfunction learnt after performing mcts

        # get best action from learnt q function after mcts
        if self.search_space=="response_space":
            print("getting best action from Q function...")
            print("current state: ", state)
            self.seed(seed)
            possible_actions = mcts.initial_actions 
            start_time = time()
            print("proposed actions: \n", possible_actions)
            qs = self.qfunction.get_qs(state, possible_actions)
            best_action_index = np.argmax(qs)
            best_action = possible_actions[best_action_index]
        
        # if semantic space used, some semantic projection is needed
        elif self.search_space=="semantic_space":
            print("getting best action from Q function...")
            print("current state: ", state)
            # get conversation semantics
            truncated_state = state.conversation # actual convo
  
            output = self.embedding_model.embed(truncated_state) # embedding
            
            conversation_semantics = tuple(output.cpu().detach().numpy())
            semantic_state = copy.deepcopy(state)
            semantic_state.conversation = conversation_semantics
            
            # get action semantics
            action_semantics = []
            # get real actions
            print("getting actions during evaluation..")
            self.seed(seed)
            possible_actions = evaluation_conversation_env.get_actions(state)
            print("possible actions generated: ", possible_actions)
            action_rewards = []
            for action in possible_actions:
                concatenated_convo = truncated_state + action # conversation
                output = self.embedding_model.embed(concatenated_convo) # embedding
                # output is the semantics after combining action with state.
                # we deduct from the output the state semantics to obtain a directional vector which
                # represents the action semantics
                action_semantic = tuple(output.cpu().detach().numpy())
                action_semantic = tuple([x1-x2 for x1,x2 in zip(list(action_semantic),list(conversation_semantics))])
                action_semantics.append(action_semantic)

                action_rewards.append(self.reward_from_embedding(output))
                
            # best_action_index = np.argmax(action_rewards) # greedy
            
            #filter off the worst 2 actions if there are more than 2
            # if len(action_rewards) > 2:
            #     smallest_indices = sorted(range(len(action_rewards)), key=lambda i: action_rewards[i])[:(2)]
            #     for index in sorted(smallest_indices, reverse=True):
            #         print("deleting the following index: ", index)
            #         del action_semantics[index]
            #         del action_rewards[index]
            
            # use Q function to get Q value
            qs = self.qfunction.get_qs(semantic_state, action_semantics)
            best_action_index = np.argmax(qs)
            
            best_action = possible_actions[best_action_index]
            if results["greedy_action_index"] != best_action_index:
                print("different greedy action selected")
                # print(f"Different action selected. greedy selected: {results['greedy_action_index']} (q={qs[results['greedy_action_index']]:.3f}), actual selected: {best_action_index} (q={qs[best_action_index]:.3f})",
                #       possible_actions[results['greedy_action_index']], possible_actions[best_action_index]
                #       )
                
        results["possible_actions"] = possible_actions
        results["possible_actions_reward"] = qs
        results["selected_action_index"] = best_action_index
        print(f"action selected by online agent: {best_action}\ttime taken: {time()-start_time}")
        return best_action
    
    # util function for resetting q function
    def reset(self):
        self.qfunction = copy.deepcopy(self.original_qfunction)
    
class ExhastiveOnlineAgent(LearntAgent):
    def __init__(self, search_depth, mcts_time_limit, llm_agent : Model, human_simulator : Model, reward_function_for_mcts : Base_Reward, search_space="response_space", reward_decay=1.0, transition_model : TransitionModelMOE =None, embedding_model=None, **kwargs) -> None:
        self.search_depth = search_depth
        self.mcts_time_limit = mcts_time_limit
        self.llm_agent = llm_agent
        self.reward_function_for_mcts = reward_function_for_mcts
        self.search_space = search_space
        self.reward_decay = reward_decay
        self.transition_model = transition_model
        self.embedding_model = embedding_model
        if isinstance(self.reward_function_for_mcts, Llama_2_Guard_Reward):
            self.reward_from_embedding = self.reward_function_for_mcts.get_safe_prob_from_embedding
        elif isinstance(self.reward_function_for_mcts, Embedding_Length_Reward):
            self.reward_from_embedding = lambda x: self.reward_function_for_mcts.model(x).detach().cpu()
    
    def generate_action(self, state : conversation_state, results = {}, seed=None, **kwargs):
        assert self.search_space=="semantic_space", "Only semantic space is supported for exhastive search"
        print("performing exhastive search...")

        self.seed(seed)
        possible_actions = self.llm_agent.sample_actions(state.conversation)
        start_time = time()

        llm_actions = torch.stack([self.embedding_model.embed(state.conversation + i) for i in possible_actions])
        rewards = [self.reward_from_embedding(llm_actions)]

        for depth in range(1, 1+self.search_depth):
            if depth % 2:
                start_time = time()
                human_actions = self.transition_model.batch_sample_human(llm_actions)
                human_actions_time = time() - start_time
                start_time = time()
                rewards.append(self.reward_from_embedding(human_actions))
                reward_time = time() - start_time
                print(f"Current depth: {depth} considering {torch.prod(torch.tensor(human_actions.shape[:-1]))} actions. Human action time: {human_actions_time:.3f}, reward time: {reward_time:.3f}" )
            else:
                start_time = time()
                llm_actions = self.transition_model.batch_sample_human(human_actions)
                llm_actions_time = time() - start_time
                start_time = time()
                rewards.append(self.reward_from_embedding(llm_actions) - rewards[-1])
                reward_time = time() - start_time
                print(f"Current depth: {depth} considering {torch.prod(torch.tensor(llm_actions.shape[:-1]))} actions. llm action time: {llm_actions_time:.3f}, reward time: {reward_time:.3f}")

        reward = rewards[-1]
        for i in range(len(rewards)-2, -1, -1):
            if i % 2:
                reward = reward.max(dim=0).values    # max over llm responses
            else:
                reward = reward.mean(dim=0) * self.reward_decay # mean over human responses
            reward = reward + rewards[i]
            # reward = (reward * self.reward_decay + rewards[i]).max(dim=0).values.mean(dim=0)
        # reward = reward.squeeze()

        best_action_index = reward.argmax()
        best_action, best_reward = possible_actions[best_action_index], reward[best_action_index]
        results["possible_actions"] = possible_actions
        results["possible_actions_reward"] = reward
        results["selected_action_index"] = best_action_index
        results["greedy_rewards"] = rewards[0]
        results["greedy_action_index"] = rewards[0].argmax()

        if results["greedy_action_index"] != best_action_index:
            print(f'Different action selected. greedy selected: {results["greedy_action_index"]} (q={rewards[0][results["greedy_action_index"]].squeeze():.3f}), actual selected: {best_action_index} (q={reward[best_action_index].squeeze():.3f})\n{possible_actions[best_action_index]}\n\n{possible_actions[results["greedy_action_index"]]}'
                )

        print(f"action selected by exhaustive agent: \"{best_action}\" with cummulative reward {best_reward}\ttime taken: {time()-start_time}")
        return best_action

def evaluate_agent(agent : LearntAgent, env : conversation_environment, starting_state : conversation_state, number_replies, results = {}, seed=None, **kwargs):
    
    cumulative_reward = 0.0
    time_taken_to_generate_action = []
    time_taken_in_simulation = []
    reward_for_one_step = []
    results["convo_replies"] = []
    for r in range(number_replies):
        
        # get best action based on starting_state
        if hasattr(agent, 'qfunction'):
            agent.qfunction.reset()

        start_time = time()
        curr_result = {}
        
        curr_seed = hash((r, seed)) % (2**32)
        action = agent.generate_action(starting_state, results = curr_result, seed=curr_seed, **kwargs)
        time_taken = time()-start_time
        time_taken_to_generate_action.append(time_taken)
        print("Time taken by agent to generate action", time_taken)
        start_time = time()
        # go to next state
        next_state, reward = env.execute_in_simulation(starting_state, action, results = curr_result, seed=curr_seed, **kwargs)
        time_taken = time()-start_time
        time_taken_in_simulation.append(time_taken)
        print("Time taken in simulation", time_taken)
        print("eval human response: ", next_state.response)
        print("reward for one step of evaluation: ", reward)
        starting_state = next_state
        cumulative_reward += reward

        reward_for_one_step.append(reward)
        results["convo_replies"].append(curr_result)
    
    # do one more action generation and do not need to generate human response
    # curr_seed = hash((-1, seed)) % (2**32)
    # action = agent.generate_action(starting_state, results = curr_result, seed=curr_seed)
    # last_step_reward = env.get_reward(starting_state.conversation, action, None) # None for human response. The reward function will handle this case to only get reward for action.
    # final_convo_including_last_actions = starting_state.conversation + action
    # starting_state = conversation_state(action, final_convo_including_last_actions)
    # starting_state.depth = starting_state.depth + 1
    # print("reward for last action step: ", last_step_reward)
    # cumulative_reward += last_step_reward
    # reward_for_one_step.append(reward)
        
    
    print("entire evaluation convo: ", starting_state.conversation)
    results["time_taken_to_generate_action"] = time_taken_to_generate_action
    results["time_taken_in_simulation"] = time_taken_in_simulation
    results["reward_for_one_step"] = reward_for_one_step
    results["entire_convo"] = starting_state.conversation.full_convo
    return cumulative_reward, starting_state

import lz4.frame
import pickle
# evaluate an agent with the mdp.
def run_evaluations(agent, type, env : conversation_environment, evaluation_starters : List[str], number_replies : int, trials : int, context_list = itertools.repeat(None), human_descriptions = itertools.repeat(None), results = {}, index = itertools.count(), seed = 42, output_file="tmp"):
    result_row = []
    convo_generated = {}
    results["time_taken_for_trial"] = {}
    results["all_rewards_from_trials"] = {}
    results["reward_mean"] = {}
    results["reward_std"] = {}
    results["trial_results"] = {}
    for i, evaluation_starter, context, human_description in zip(index, tqdm(evaluation_starters), context_list, human_descriptions):

        initial_state = conversation_state((evaluation_starter), Conversation(evaluation_starter))
        initial_state.depth = 1
        
        # repeated trials
        rewards = []
        convo_generated_from_trials = []
        time_taken_for_trial = []
        trial_results = []
        for x in range(trials):

            if hasattr(agent, 'qfunction'):
                agent.qfunction.reset()
            print("trial: ", x, " of evaluation for agent of type:  ", type)
            start_time = time()
            curr_trial_result = {}

            curr_seed = hash((seed, evaluation_starter, x)) % (2**32)

            cumulative_reward, entire_convo = evaluate_agent(agent, env, initial_state, number_replies, results = curr_trial_result, context = context, human_description = human_description, seed = curr_seed)
            time_taken = time()-start_time
            print("Time taken for current trial", time_taken)
            convo_generated_from_trials.append(entire_convo.conversation)
            print("cumulative reward for this trial: ", cumulative_reward)

            time_taken_for_trial.append(time_taken)
            rewards.append(cumulative_reward)
            trial_results.append(curr_trial_result)

        reward_mean = np.mean(rewards)
        reward_std = stats.sem(rewards)
        results["time_taken_for_trial"][i] = time_taken_for_trial
        results["all_rewards_from_trials"][i] = rewards
        results["reward_mean"][i] = reward_mean
        results["reward_std"][i] = reward_std
        results["trial_results"][i] = trial_results
        print(evaluation_starter)
        print("all rewards from trials: ", rewards)
        print("mean: ", reward_mean)
        print("std error: ", reward_std)
        convo_generated[evaluation_starter] = convo_generated_from_trials
        result_row.append(((np.mean(rewards)), (stats.sem(rewards))))

        if i % 20 == 19:
            with lz4.frame.open(output_file+'_tmp.pkl', 'wb') as f:
                pickle.dump(results, f)
    
    return result_row, convo_generated

def run_evaluations_singular(agent : LearntAgent, type, env : conversation_environment, evaluation_starter : str, context_list = itertools.repeat(None), human_descriptions = itertools.repeat(None), results = {}, index = itertools.count(), seed = 42, output_file="tmp"):
    initial_state = conversation_state((evaluation_starter), Conversation(evaluation_starter))
    initial_state.depth = 1
    best_response, possible_actions, rewards = evaluate_agent_singular(agent, env, initial_state)
    return best_response, possible_actions, rewards
    
        
    
def evaluate_agent_singular(agent : LearntAgent, env : conversation_environment, starting_state : conversation_state, seed=None, **kwargs):
        
    # get best action based on starting_state
    if hasattr(agent, 'qfunction'):
        agent.qfunction.reset()

    start_time = time()
    curr_result = {}
    
    curr_seed = hash((seed, seed)) % (2**32)
    action = agent.generate_action(starting_state, results = curr_result, seed=curr_seed, **kwargs)

    return action, curr_result["possible_actions"], curr_result["possible_actions_reward"]