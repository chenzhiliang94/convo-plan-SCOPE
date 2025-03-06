
def length_convo(convo):
    '''
    assume convo is a list of sentence strings, more than size 2
    '''
    
    cumulative_reward = 0.0  
    for idx, sentence in enumerate(convo):
        if idx % 2 == 0:
            cumulative_reward += len(sentence)
    return cumulative_reward

'''
Each reward function here receives a (convo_state, action, human_response)
input and returns the IMMEDIATE reward/cost for performing a certain action and observing the human response.
The cumulative sum should be derived by the function user.
'''

from abc import ABC, abstractmethod
from agent.Conversation import Conversation

class Base_Reward(ABC):
    @abstractmethod
    def get_reward(prev_state : Conversation, action : str, human_response : str) -> float:
        pass