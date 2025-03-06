import random
import torch
from reward.rewards_import import *
from transition_models.transition_model import TransitionModel

class conversation_semantic_state():
    depth = 0
    def __init__(self, conversation : tuple) -> None:
        self.conversation = conversation
        self.response = conversation
        self.depth = 2
    
    def __str__(self):
        return "Depth: {}, Conversation: {}".format(self.depth, self.conversation)
        
class semantic_conversation_environment():

    def __init__(self, embedding_model, transition_model : TransitionModel, initial_state = "Tell me about a fact about Singapore.", max_depth=10, reward_function : Base_Reward = Human_Length_Reward()) -> None:
        self.embedding_model = embedding_model
        self.transition_model = transition_model
        self.state_to_action_map = {}
        self.state_action_to_response_map = {}
        self.max_depth = max_depth
        self.initial_state = initial_state
        self.reward_function = reward_function.get_reward
    
    def get_initial_state(self):
        print("getting initial state...")
        with torch.no_grad():
            initial_embedding = self.embedding_model.embed(self.initial_state)
            initial_embedding = initial_embedding.cpu().numpy()
            
            conversation_semantics = tuple(initial_embedding)
            initial_state = conversation_semantic_state(conversation_semantics)
            initial_state.depth = 1
            return initial_state
        
    def get_actions(self, state : conversation_semantic_state) -> tuple:
        historical_context = state.conversation
        if historical_context in self.state_to_action_map:
            print("state already in state_to_action_map dict, use the actions!")
            actions = self.state_to_action_map[historical_context]
            return actions
        else:
            actions = self.transition_model.sample_actions(historical_context)
            self.state_to_action_map[historical_context] = actions
            return actions
        
    def is_terminal(self, state):
        if state.depth >= self.max_depth:
            return True
        return False
    # H_current, L, H_current + L + H_next
    def get_reward(self, prev_state : tuple, action : tuple, new_state : tuple | None):
        return self.reward_function(prev_state, action, new_state)
    
    # get action in simulation stage. So no storing of actions here
    def get_actions_in_simulation(self, state : conversation_semantic_state):
        historical_context = state.conversation
        possible_actions = self.transition_model.sample_actions(historical_context)
        return possible_actions
    
    # randomly get a result state (this is only in simulation)
    def execute_in_simulation(self, state : conversation_semantic_state, action, seed=None, **kwargs):
        
        # old way: get responses explicitly from model 
        historical_context = state.conversation
        possible_results = self.transition_model.transit(historical_context, action)
        if seed is not None:
            random.seed(seed)
        result_state_after_human_response = random.choice(possible_results) 
        
        new_state = conversation_semantic_state(result_state_after_human_response)
        new_state.depth = state.depth + 2
        
        # new way: immediately jump from s_old,a -> s_new with transition model
        # new_state = self.transition_model.transit(state, action)
        
        return new_state, self.get_reward(state.conversation, action, new_state.conversation)
        
    # during selection, we already have defined action to possible response mapping. So the transition probability is already approximated
    def execute_in_selection(self, state : conversation_semantic_state, action):
        historical_context = state.conversation
        possible_responses = self.state_action_to_response_map[(historical_context, action)]
        
        # choose a random state to happen. TODO: use a transition probability
        result_human_response = random.choice(list(possible_responses))
        
        # generate a state
        selected_state = conversation_semantic_state(result_human_response)
        selected_state.depth = state.depth + 2
        
        return selected_state, self.get_reward(state.conversation, action, selected_state.conversation)
    
    # during expansion, we are trying out an action that is definitely not used before at this state
    def execute_in_expansion(self, state : conversation_semantic_state, action):
        historical_context = state.conversation
        
        # given a state, and action, how will a human respond? We shall find out using a simulator and store the possible responses in our dictionary
        possible_responses = self.transition_model.transit(historical_context, action)
        assert not (historical_context, action) in self.state_action_to_response_map
        self.state_action_to_response_map[(historical_context, action)] = possible_responses
        
        # choose a random state to happen. TODO: use a transition probability
        result_state_after_human_response = random.choice(list(possible_responses))
        
        # generate a state
        expanded_state = conversation_semantic_state(result_state_after_human_response)
        expanded_state.depth = state.depth + 2
        
        return expanded_state, self.get_reward(state.conversation, action, expanded_state.conversation)
        
    def get_discount_factor(self):
        return 1.0