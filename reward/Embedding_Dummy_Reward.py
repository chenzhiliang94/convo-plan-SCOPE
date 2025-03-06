
from reward.Base_Reward import Base_Reward
import random

# Reward function that returns random reward
class Embedding_Dummy_Reward(Base_Reward):
    def get_reward(self, prev_state, action, human_response) -> float:
        return random.uniform(0.99,1)