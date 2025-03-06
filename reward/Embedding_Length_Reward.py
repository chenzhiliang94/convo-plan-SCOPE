
from reward.Base_Reward import Base_Reward
import torch
import torch.nn as nn
from token_count import TokenCount

from agent.Conversation import Conversation
class MLPRegression(nn.Module):
    def __init__(self):
        super(MLPRegression, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = (self.fc5(x))
        return x

# Reward function that returns random reward
class Embedding_Length_Reward(Base_Reward):
    
    def __init__(self, add_llm_length : bool, path_to_model="reward/embedding_length_reward", device_map=0) -> None:
        super().__init__()
        print(f"Loading embedding length model on device {device_map}...")
        self.model = MLPRegression()
        self.model.load_state_dict(torch.load(path_to_model, map_location=torch.device(device_map)))
        self.add_llm_length = add_llm_length
        print("length model initialized with add_llm_length: ", self.add_llm_length)
        
    def get_reward(self, prev_state : tuple | str | Conversation, action : tuple | str, human_response : tuple | str | None) -> float:
        if isinstance(prev_state, Conversation):
            prev_state = str(prev_state)
        # for last step evaluation
        if human_response is None:
            if isinstance(action, str):
                return self.get_tokens_from_str(action)
            else:
                with torch.no_grad():
                    reward = self.model(torch.FloatTensor(prev_state) + torch.FloatTensor(action)) - self.model(torch.FloatTensor(prev_state))
                print("reward from embedding length: ", reward)
                return reward * 10
        
        # if instance is string. its during evaluation and just response length
        if isinstance(human_response, str):
            if self.add_llm_length:
                return self.get_tokens_from_str(human_response) + self.get_tokens_from_str(action)
            else:
                return self.get_tokens_from_str(human_response)
        
        # if not string, human response length is in semantic space. So take difference.
        with torch.no_grad():
            if self.add_llm_length:
                reward = self.model(torch.FloatTensor(human_response)) - self.model(torch.FloatTensor(prev_state))
            else:
                reward = self.model(torch.FloatTensor(human_response)) - self.model(torch.FloatTensor(prev_state) + torch.FloatTensor(action))
        print("reward from embedding length: ", reward)
        return reward * 10
    
    def get_tokens_from_str(self, convo : str) -> float:
        tc = TokenCount(model_name="gpt-3.5-turbo")
        return tc.num_tokens_from_string(convo)/100