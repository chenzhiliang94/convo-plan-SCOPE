
from reward.Base_Reward import Base_Reward
from agent.Conversation import Conversation

# Reward function that returns the length of the human response
class Human_Length_Reward(Base_Reward):
    def get_reward(self, prev_state : Conversation, action : str, human_response : str) -> float:
        return 0.01*len(human_response)