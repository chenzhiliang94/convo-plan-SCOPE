import random
from reward.rewards_import import *
from agent.Conversation import Conversation
from agent.Model import Model

class conversation_state():
    depth = 0
    def __init__(self, response, conversation : Conversation) -> None:
        self.response = response
        self.conversation = conversation
        self.depth = 2
    
    def __str__(self):
        return "Depth: {}, Response: {}, Conversation: {}".format(self.depth, self.response, self.conversation)
        
class conversation_environment():
    
    def __init__(self, human : Model, llm : Model, initial_state, max_depth=10, reward_function : Base_Reward = Human_Length_Reward(), reward_decay=0.9) -> None:
        self.state_to_action_map = {}
        self.state_action_to_response_map = {}
        self.max_depth = max_depth
        self.human_env = human
        self.llm_agent = llm
        self.initial_state = initial_state
        self.reward_function = reward_function.get_reward
        self.reward_decay = reward_decay
    
    def get_initial_state(self):
        initial_state = conversation_state(str(self.initial_state), self.initial_state)
        initial_state.depth = 1
        return initial_state
        
    def get_actions(self, state : conversation_state):
        historical_context = state.conversation
        if historical_context in self.state_to_action_map:
            actions = self.state_to_action_map[historical_context]
            return actions
        else:
            actions = self.llm_agent.sample_actions(historical_context)
            self.state_to_action_map[historical_context] = actions
            return actions
        
    def is_terminal(self, state):
        if state.depth >= self.max_depth or state.response == "EXIT":
            return True
        return False
    
    def get_reward(self, prev_state : Conversation, action : str, human_response : str | None):
        return self.reward_function(prev_state, action, human_response)

    # get action in simulation stage. So no storing of actions here
    def get_actions_in_simulation(self, state : conversation_state):
        historical_context = state.conversation
        possible_responses = self.llm_agent.sample_actions(historical_context)
        return possible_responses
    
    # randomly get a result state (this is only in simulation)
    def execute_in_simulation(self, state : conversation_state, action, results = {}, seed=None, **kwargs):
        historical_context = state.conversation

        possible_responses = self.human_env.sample_actions(historical_context + action, **kwargs)
        print("possible human responses: ", possible_responses)
        if seed is not None:
            random.seed(seed)
        rand_index = random.randint(0, len(possible_responses)-1)
        result_human_response = possible_responses[rand_index]
        new_historical_context = historical_context  + action
        new_historical_context = new_historical_context + result_human_response
        selected_state = conversation_state(result_human_response, new_historical_context)
        selected_state.depth = state.depth + 2

        results["possible_human_response"] = possible_responses
        results["selected_human_index"] = rand_index
        
        return selected_state, self.get_reward(state.conversation, action, result_human_response)
        
    # during selection, we already have defined action to possible response mapping. So the transition probability is already approximated
    def execute_in_selection(self, state : conversation_state, action):
        historical_context = state.conversation
        
        possible_responses = self.state_action_to_response_map[(historical_context + action)]
        
        # choose a random state to happen. TODO: use a transition probability
        result_human_response = random.choice(list(possible_responses))
        
        # generate a state
        new_historical_context = historical_context  + action
        new_historical_context = new_historical_context + result_human_response
        selected_state = conversation_state(result_human_response, new_historical_context)
        selected_state.depth = state.depth + 2
        
        # get reward; calculate reward value of result_response using some metric
        reward = 0.0 # actually, dependent only on action
        
        return selected_state, self.get_reward(state.conversation, action, result_human_response)
    
    # during expansion, we are trying out an action that is definitely not used before at this state
    def execute_in_expansion(self, state : conversation_state, action):
        historical_context = state.conversation
        immediate_response = state.response
        
        # given a state, and action, how will a human respond? We shall find out using a simulator and store the possible responses in our dictionary
        input_to_human_env = historical_context + action
        possible_responses = self.human_env.sample_actions(input_to_human_env)
        if (input_to_human_env) in self.state_action_to_response_map:
            print("something wrong! when expanding somehow the resulting state was already seen before, but its ok we will re-add.")
        self.state_action_to_response_map[(input_to_human_env)] = possible_responses

        # choose a random state to happen. TODO: use a transition probability
        result_human_response = random.choice(list(possible_responses))
        
        # generate a state
        new_historical_context = input_to_human_env + result_human_response
        expanded_state = conversation_state(result_human_response, new_historical_context)
        expanded_state.depth = state.depth + 2
        
        return expanded_state, self.get_reward(state.conversation, action, result_human_response)
        
    def get_discount_factor(self):
        return self.reward_decay