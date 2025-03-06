from collections import defaultdict
from monte_carlo_tree_search.qfunction import QFunction
import time
import numpy as np

def combine_encoded_inputs(input1, input2):
    new_encoding = {}
    for k in input1.keys():
        # padding first 
        i1_size = input1[k].shape[1]
        i2_size = input2[k].shape[1]
        i1 = input1[k]
        i2 = input2[k]
        if i2_size > i1_size:
            i1 = nn.functional.pad(input1[k], (0, i2_size-i1_size), 'constant', 0)
        elif i2_size < i1_size:
            i2 = nn.functional.pad(input2[k], (0, i1_size-i2_size), 'constant', 0)
        new_encoding[k] = torch.cat((i1,i2), 0)
    return new_encoding


class QTable(QFunction):
    def __init__(self, default=0.0):
        self.qtable = defaultdict(lambda: default)

    def update(self, state, action, delta, visits, reward):
        self.qtable[(state, action)] = self.qtable[(state, action)] + delta

    def get_q_value(self, state, action):
        return self.qtable[(state, action)]
    
import torch
import torch.nn as nn
from monte_carlo_tree_search.qfunction import QFunction
from torch.optim import Adam
from monte_carlo_tree_search.deep_agent import DeepAgent
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class DeepQFunction(QFunction, DeepAgent):
    """ A neural network to represent the Q-function.
        This class uses PyTorch for the neural network framework (https://pytorch.org/).
    """

    def __init__(
        self, alpha=0.001, steps_update=100, cuda = torch.device('cuda:2')
    ) -> None:
        raise NotImplementedError("This class should not be used. Are you sure you are using the right QFunction?.")
        self.alpha = alpha
        self.steps_update = steps_update
        self.tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        self.q_network = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-uncased", 
                                                           num_labels = 1).to(cuda)
        self.optimiser = Adam(self.q_network.parameters(), lr=self.alpha)
        self.cuda = cuda

    def merge(self, state, action):
        # merge conversation, and LLM response together.
        return state.conversation + action
    
    def update(self, state, action, delta, visits, reward):
        optimiser = Adam(self.q_network.parameters(), lr=0.01 * (1/visits)**2)
        optimiser.zero_grad()  # Reset gradients to zero
        merged_convo = self.merge(state, action)
        merged_convo = str(merged_convo)
        if len(merged_convo) > 1000:
            merged_convo = merged_convo[-999:]
        encoded_input = self.tokenizer(merged_convo, return_tensors='pt')
        if len(encoded_input) > 512:
            encoded_input = encoded_input[:512]
        # if len(merged_convo) > 1000:
        #     merged_convo = merged_convo[-999:]
        encoded_input = self.tokenizer(merged_convo, truncation=True, max_length=512,  return_tensors='pt').to(self.cuda)

        #print("output of network before update: ", output.logits)
        for x in range(self.steps_update):
            optimiser.zero_grad()  # Reset gradients to zero
            output = self.q_network(**encoded_input, labels = torch.tensor(reward, dtype=torch.float).to(self.cuda))
            output.loss.backward()
            optimiser.step()  # Do a gradient descent step with the optimiser
        #print("output of network after update: ", output.logits)
        
    def get_q_value(self, state, action):
        
        merged_convo = self.merge(state, action)
        merged_convo = str(merged_convo)
        if len(merged_convo) > 1000:
            merged_convo = merged_convo[-999:]
        # if len(merged_convo) > 1000:
        #     merged_convo = merged_convo[-999:]
        # Convert the state into a tensor
        encoded_input = self.tokenizer(merged_convo, truncation=True, max_length=512,  return_tensors='pt').to(self.cuda)
        with torch.no_grad():
            output = self.q_network(**encoded_input)
        return output.logits[0][0]

    def get_qs(self, state, actions):
        qs = []
        for action in actions:
            merged_convo = self.merge(state, action)
            # if len(merged_convo) > 1000:
            #     merged_convo = merged_convo[-999:]
            encoded_input = self.tokenizer(merged_convo, truncation=True, max_length=512, return_tensors='pt').to(self.cuda)
            with torch.no_grad():
                reward_estimate = self.q_network(**encoded_input).logits[0][0].cpu()
            qs.append(reward_estimate)
        return qs

    def get_max_q(self, state, actions):
        qs = self.get_qs(state, actions)
        arg_max_q = np.argmax(qs)
        best_action = actions[arg_max_q]
        best_reward = qs[arg_max_q]
        return (best_action, best_reward)
            
# class DeepQSemanticFunction(QFunction, DeepAgent):
#     """ A neural network to represent the Q-function for semantic space
#         This class uses PyTorch for the neural network framework (https://pytorch.org/).
#     """

#     def __init__(
#         self, dim, alpha=0.001
#     ) -> None:
#         self.alpha = alpha
#         self.dim = dim
#         self.q_network = nn.Sequential(
#             nn.Linear(dim * 2, 128),
#             nn.ReLU(),
#             nn.Linear(128, 24),
#             nn.ReLU(),
#             nn.Linear(24, 12),
#             nn.ReLU(),
#             nn.Linear(12, 1)
#         )
#         self.optimiser = Adam(self.q_network.parameters(), lr=self.alpha)

#     def merge(self, state, action):
#         # merge conversation, and LLM response together.
#         merged_convo = list(state.conversation) + list(action)
#         return torch.Tensor([merged_convo])
    
#     def update(self, state, action, delta, visits, reward):
#         self.optimiser.lr=0.0005 * (1/visits)**2
#         merged_convo = self.merge(state, action)
#         for x in range(30):
#             self.optimiser.zero_grad()  # Reset gradients to zero
#             loss_fn = nn.MSELoss()
#             y_pred = self.q_network(merged_convo)
#             loss = loss_fn(y_pred, torch.tensor([reward],requires_grad=True))
#             loss.backward()
#             self.optimiser.step()
        
#     def get_q_value(self, state, action):
#         merged_convo = self.merge(state, action)
#         output = self.q_network(merged_convo)
#         return output[0][0]

#     def get_max_q(self, state, actions):
            
#         best_action = None
#         best_reward = float("-inf")
#         for action in actions:
#             merged_convo = self.merge(state, action)
#             reward_estimate = self.q_network(merged_convo)[0][0]
#             if reward_estimate > best_reward:
#                 best_action = action
#                 best_reward = reward_estimate
#         return (best_action, best_reward)
    
    
class DeepQSemanticFunction(QFunction, DeepAgent):
    """ A neural network to represent the Q-function for semantic space
        This class uses PyTorch for the neural network framework (https://pytorch.org/).
    """

    def __init__(
        self, dim, cuda, steps_update, alpha=0.001
    ) -> None:
        self.alpha = alpha
        self.dim = dim
        self.update_steps = steps_update
        self.cuda = cuda
        self.q_network = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, int(dim/4)),
            nn.ReLU(),
            nn.Linear(int(dim/4), 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(cuda)
        print(f"Using {cuda} for DeepQSemanticFunction")
        self.reset()
    
    def reset(self):
        for layer in self.q_network:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        self.optimiser = Adam(self.q_network.parameters(), lr=self.alpha)
        self.replay_buffer = None
        self.past_rewards = None

    def merge(self, state, action):
        # merge conversation, and LLM response together.
        merged_convo = list(state.conversation) + list(action)
        return torch.Tensor([merged_convo])
    
    def update_buffer(self, input, reward):
        if self.past_rewards is None:
            self.past_rewards = reward
        else:
            self.past_rewards = torch.cat((self.past_rewards, reward), 0)
        if self.replay_buffer is None:
            self.replay_buffer = input
        else:
            self.replay_buffer = torch.cat((self.replay_buffer, input), 0)
            
    def update(self, state, action, delta, visits, reward):
        loss_fn = nn.MSELoss()
        self.optimiser = Adam(self.q_network.parameters(), lr=self.alpha * (1/visits)**2)
        merged_convo = self.merge(state, action).to(self.cuda)
        reward = torch.tensor(reward,dtype=torch.float).to(self.cuda).unsqueeze(0)
        losses = []
        for x in range(self.update_steps):
            self.optimiser.zero_grad()  # Reset gradients to zero
            y_pred = self.q_network(merged_convo)
            loss = loss_fn(y_pred.squeeze(1), reward)
            losses.append(loss.item())
            loss.backward()
            self.optimiser.step()
        self.update_buffer(merged_convo, reward)
        print(f"loss for regular q update {losses[-1:0:-10][::-1]}")

        if self.replay_buffer is None:
            return
        
        # print("past reward: ", self.past_rewards)
        # print("y ", y_pred)
        losses = []
        for x in range(self.update_steps):
            self.optimiser.zero_grad()  # Reset gradients to zero
            y_pred = self.q_network(self.replay_buffer)
            loss = loss_fn(y_pred.squeeze(), self.past_rewards.squeeze())
            losses.append(loss.item())
            loss.backward()
            self.optimiser.step()
        
        print(f"loss for replay buffer q update {losses[-1:0:-10][::-1]}")
        
        # for x in range(self.steps_update):
        #     optimiser.zero_grad()  # Reset gradients to zero
        #     output = self.q_network(**encoded_input, labels = torch.tensor(reward, dtype=torch.float).to(self.cuda))
        #     if output.loss == torch.tensor(float('nan')): # if loss becomes nan, reduce LR
        #         optimiser = Adam(self.q_network.parameters(), lr= 0.1 * self.alpha * (1/visits)**2)
        #         continue
        #     output.loss.backward()
        #     optimiser.step()  # Do a gradient descent step with the optimiser
        #     print("loss in standard update: ", output.loss)
            
    def get_q_value(self, state, action):
        merged_convo = self.merge(state, action).to(self.cuda)
        with torch.no_grad():
            output = self.q_network(merged_convo)
        return output[0][0]
    
    def get_qs(self, state, actions):
        qs = []
        for action in actions:
            merged_convo = self.merge(state, action).to(self.cuda)
            with torch.no_grad():
                reward_estimate = self.q_network(merged_convo)[0][0].cpu()
            qs.append(reward_estimate)
        print("q values estimate for actions are: ", qs)
        return qs

    def get_max_q(self, state, actions):
        qs = self.get_qs(state, actions)
        arg_max_q = np.argmax(qs)
        best_action = actions[arg_max_q]
        best_reward = qs[arg_max_q]
        return (best_action, best_reward)
    
    
class ReplayBufferDeepQFunction(QFunction, DeepAgent):
    """ A neural network to represent the Q-function.
        This class uses PyTorch for the neural network framework (https://pytorch.org/).
    """

    def __init__(
        self, alpha=0.1, steps_update=100, cuda = torch.device('cuda:2')
    ) -> None:
        self.alpha = alpha
        self.steps_update = steps_update
        self.model_name = "google-bert/bert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.cuda = cuda
        self.reset()

    def reset(self):
        self.q_network = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels = 1).to(self.cuda)
        self.optimiser = Adam(self.q_network.parameters(), lr=self.alpha)
        self.replay_buffer = None
        self.past_rewards = None
    
    def merge(self, state, action):
        # merge conversation, and LLM response together.
        return state.conversation + action
    
    def update_buffer(self, input, reward):
        if self.past_rewards is None:
            self.past_rewards = reward
        else:
            self.past_rewards = torch.cat((self.past_rewards, reward), 0)
        if self.replay_buffer is None:
            self.replay_buffer = input
        else:
            self.replay_buffer = combine_encoded_inputs(self.replay_buffer, input)

    def update(self, state, action, delta, visits, reward):
        merged_convo = self.merge(state, action)
        merged_convo = str(merged_convo)
        
        # update replay buffer
        encoded_input = self.tokenizer(merged_convo, truncation=True, max_length=512,  padding=True, return_tensors='pt').to(self.cuda)
        reward = torch.tensor(reward,dtype=torch.float).to(self.cuda).unsqueeze(0)
        self.update_buffer(encoded_input, reward)
        
        self.q_network.train()
        # update based on this specific experience
        start_time = time.time()
        self.optimiser.param_groups[0]['lr'] *= self.alpha * (1/visits)**2
        for x in range(self.steps_update):
            self.optimiser.zero_grad()  # Reset gradients to zero
            output = self.q_network(**encoded_input, labels = reward)
            if torch.isnan(output.loss): # if loss becomes nan, reduce LR
                self.optimiser.param_groups[0]['lr'] *= 0.1
                continue
            output.loss.backward()
            self.optimiser.step()  # Do a gradient descent step with the optimiser
            print("loss in standard update: ", output.loss)
        print("time taken for update Q", time.time()-start_time)
        start_time = time.time()
        
        # update based on replay buffer
        losses = []
        self.optimiser.param_groups[0]['lr'] = 0.3* self.alpha * (1/visits)**2
        for x in range(self.steps_update):
            self.optimiser.zero_grad()  # Reset gradients to zero
            output = self.q_network(**self.replay_buffer, labels = self.past_rewards)

            if torch.isnan(output.loss): # if loss becomes nan, reduce LR
                self.optimiser.param_groups[0]['lr'] *= 0.1
                continue
            output.loss.backward()
            losses.append(output.loss.item())
            self.optimiser.step()  # Do a gradient descent step with the optimiser
        print(f"loss for replay buffer q update {losses[-1:0:-10][::-1]}")
        print("time taken for update Q with replay buffer: ", time.time()-start_time)
    
    # def update_with_replay_buffer(self):
    #     optimiser = Adam(self.q_network.parameters(), lr=self.alpha)
        
    #     # update based on replay buffer
    #     for x in range(self.steps_update):
    #         optimiser.zero_grad()  # Reset gradients to zero
    #         output = self.q_network(**self.replay_buffer, labels = torch.tensor(self.past_rewards, dtype=torch.float).to(self.cuda))
    #         output.loss.backward()
    #         print(output.loss)
    #         optimiser.step()  # Do a gradient descent step with the optimiser
        
    def get_q_value(self, state, action):
        print("getting q value of merged convo:")
        
        merged_convo = self.merge(state, action)
        merged_convo = str(merged_convo)
        print(merged_convo)
        encoded_input = self.tokenizer(merged_convo, truncation=True, max_length=512,  return_tensors='pt').to(self.cuda)
        #print(encoded_input)
        self.q_network.eval()
        with torch.no_grad():
            output = self.q_network(**encoded_input)
            q_value = output.logits[0][0].cpu()
            print(f"Q value is: {q_value:.8f}")
        return q_value
    
    def get_qs(self, state, actions):
        qs = []
        for action in actions:
            reward_estimate = self.get_q_value(state, action)
            qs.append(reward_estimate)
        return qs

    def get_max_q(self, state, actions):
        qs = self.get_qs(state, actions)
        arg_max_q = np.argmax(qs)
        best_action = actions[arg_max_q]
        best_reward = qs[arg_max_q]
        return (best_action, best_reward)