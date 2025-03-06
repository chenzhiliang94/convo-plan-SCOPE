import random
import numpy as np
import torch
from transition_models.regression_wrapper import RegressionWrapper
from mixture_of_experts import HeirarchicalMoE
from mdn.src.blocks import MixtureDensityNetwork
import os
import traceback

'''
input and output data are all in tuple format (due to the need for dict hashing in MCTS procedure).
'''
class TransitionModel:
    def __init__(self, samples=5) -> None:
        self.samples = samples
        pass
    
    # given a state (n-dim embedding), acton (n-dim directional vector), account for human stochastic response and return a new state
    def transit(self, state, action):
        
        # convert tuple to np array
        state = np.array(list(state))
        action = np.array(list(action))
        
        intermediate_state = state + action # action is a directional vector, so we can add them directly
        
        # mimic transition for now randomly
        # convert back to tuple format
        new_states = [tuple(intermediate_state * random.gauss(0, 1)) for x in range(self.samples)]
        
        return new_states
    
    # given a state (n-dim embedding), return a LLM action (n-dimensional vector)
    def sample_actions(self, state):
        
        # convert tuple to np array
        state = np.array(list(state))
        
        # mimic action for now randomly
        dim = state.shape[0]
        return [tuple(np.random.normal(0, 1, dim)) for x in range(self.samples)]
    
class TransitionModelMOE:
    def __init__(self, samples=4, noise=0.05, cuda=torch.device("cpu"), max_batch_size = 2048, transition_model_dir="models/deterministic/") -> None:
        self.samples = samples
        self.std = noise
        self.llm_models = [] # used to generate actions
        self.human_models = [] # used to generate transition to next state
        self.cuda = cuda
        self.max_batch_size = max_batch_size
        print(f"Loading transition models on device {cuda}...")
        for i, dir in enumerate(os.listdir(transition_model_dir)):
            try:
                models_dir = f"{dir}/human_llm"
                self.llm_models.append(RegressionWrapper(HeirarchicalMoE(1024)).float().to(self.cuda))
                self.llm_models[i].load_state_dict(torch.load(f"{models_dir}/model_min_train.pth")["model_state_dict"])
                self.llm_models[i].eval()
            except:
                pass
            try:
                models_dir = f"{dir}/llm_human"
                self.human_models.append(RegressionWrapper(HeirarchicalMoE(1024)).float().to(self.cuda))
                self.human_models[i].load_state_dict(torch.load(f"{models_dir}/model_min_train.pth")["model_state_dict"])
                self.human_models[i].eval()
            except:
                pass
        assert len(self.llm_models) >= 1 and len(self.human_models) >= 1, f"No transition models loaded! Are you sure the directory {transition_model_dir} contains the models?"
        print(f"Loaded {len(self.llm_models)} LLM models and {len(self.human_models)} human models on device {cuda}, taking up {np.sum([print_model_memory_usage(i) for i in self.llm_models]) + np.sum([print_model_memory_usage(i) for i in self.human_models]):.0f}MB.")
    
    def forward(self, input, models):   # input should be (batch x dim)
        next_states = []
        for model in models:
            with torch.no_grad():
                tmp = []
                for i in range(0, len(input), self.max_batch_size):
                    tmp.append(model.forward(input[i:i+self.max_batch_size].to(self.cuda)))
                next_states.append(torch.cat(tmp, dim=0).cpu())

        next_states = torch.stack(next_states)

        if len(next_states) == 1:
            noise = torch.randn(self.samples, *next_states.shape) * self.std
            perturbed_state = next_states.repeat(self.samples + 1, *([1] * len(next_states.shape)))
            perturbed_state[1:] += noise
        else:
            perturbed_state = next_states
        return perturbed_state  # (samples x batch x dim)

    # given a state (n-dim embedding), acton (n-dim directional vector), account for human stochastic response and return a new state
    def transit(self, state, action):
        
        # convert to torch tensor
        state = torch.tensor(state)
        action = torch.tensor(action)

        intermediate_state = state + action # action is a directional vector, so we can add them directly

        perturbed_state = self.forward(intermediate_state.unsqueeze(0), self.human_models)[:,0,:]
        perturbed_state = [tuple(i.numpy()) for i in perturbed_state]

        return perturbed_state

    
    # given a state (n-dim embedding), return a LLM action (n-dimensional vector)
    def sample_actions(self, state):
        
        # convert to torch tensor
        state = torch.tensor(state)

        perturbed_state = self.forward(state.unsqueeze(0), self.llm_models)[:,0,:]
        action_vector = perturbed_state - state
        action_vector = [tuple(i.numpy()) for i in action_vector]

        return action_vector
    
    def batch_sample_human(self, input):            # input should be (... x dim)
        flattened = input.view(-1, input.shape[-1])
        output = self.forward(flattened, self.human_models)
        return output.view(-1, *input.shape)        # output should be (samples x ... x dim)

    def batch_sample_llm(self, input):              # input should be (... x batch x dim)
        flattened = input.view(-1, input.shape[-1])
        output = self.forward(flattened, self.llm_models)
        return output.view(-1, *input.shape)        # output should be (samples x ... x dim)
    
class TransitionModelMDN:
    def __init__(self, samples=4, noise=0.05, cuda=torch.device("cpu"), max_batch_size = 2048, transition_model_dir="models/deterministic/") -> None:
        self.samples = samples
        self.std = noise
        self.cuda = cuda
        self.max_batch_size = max_batch_size
        print(f"Loading transition models on device {cuda}...")
        try:
            models_dir = f"{transition_model_dir}/human_llm"
            self.llm_model = RegressionWrapper(MixtureDensityNetwork(1024,1024,256,512)).float().to(self.cuda)
            self.llm_model.load_state_dict(torch.load(f"{models_dir}/model_min_val.pth")["model_state_dict"])
            self.llm_model.eval()
        except:
            traceback.print_exc()
            print(f"No transition models loaded! Are you sure the directory {models_dir} contains the models?")
            exit()
        try:
            models_dir = f"{transition_model_dir}/llm_human"
            self.human_model = RegressionWrapper(MixtureDensityNetwork(1024,1024,256,512)).float().to(self.cuda)
            self.human_model.load_state_dict(torch.load(f"{models_dir}/model_min_val.pth")["model_state_dict"])
            self.human_model.eval()
        except:
            traceback.print_exc()
            print(f"No transition models loaded! Are you sure the directory {models_dir} contains the models?")
            exit()
        print(f"Loaded LLM models and human models on device {cuda}, taking up {print_model_memory_usage(self.llm_model) + print_model_memory_usage(self.human_model):.0f}MB.")
    
    def forward(self, input, model):   # input should be (batch x dim)
        with torch.no_grad():
            tmp = []
            for i in range(0, len(input), self.max_batch_size):
                tmp.append(model.sample(input[i:i+self.max_batch_size].to(self.cuda), samples_per_input=self.samples).cpu())
        next_states = torch.cat(tmp, dim=1)
        perturbed_state = next_states
        return perturbed_state  # (samples x batch x dim)

    # given a state (n-dim embedding), acton (n-dim directional vector), account for human stochastic response and return a new state
    def transit(self, state, action):
        
        # convert to torch tensor
        state = torch.tensor(state)
        action = torch.tensor(action)

        intermediate_state = state + action # action is a directional vector, so we can add them directly

        perturbed_state = self.forward(intermediate_state.unsqueeze(0), self.human_model)[:,0,:]
        perturbed_state = [tuple(i.numpy()) for i in perturbed_state]

        return perturbed_state

    
    # given a state (n-dim embedding), return a LLM action (n-dimensional vector)
    def sample_actions(self, state):
        
        # convert to torch tensor
        state = torch.tensor(state)

        perturbed_state = self.forward(state.unsqueeze(0), self.llm_model)[:,0,:]
        action_vector = perturbed_state - state
        action_vector = [tuple(i.numpy()) for i in action_vector]

        return action_vector
    
    def batch_sample_human(self, input):            # input should be (... x dim)
        flattened = input.view(-1, input.shape[-1])
        output = self.forward(flattened, self.human_model)
        return output.view(-1, *input.shape)        # output should be (samples x ... x dim)

    def batch_sample_llm(self, input):              # input should be (... x batch x dim)
        flattened = input.view(-1, input.shape[-1])
        output = self.forward(flattened, self.llm_model)
        return output.view(-1, *input.shape)        # output should be (samples x ... x dim)

def model_memory_usage(model):
    def tensor_memory(tensor):
        if tensor is None:
            return 0
        num_elements = tensor.numel()
        element_size = tensor.element_size()
        return num_elements * element_size

    total_memory = 0

    for param in model.parameters():
        total_memory += tensor_memory(param)
        if param.grad is not None:
            total_memory += tensor_memory(param.grad)

    return total_memory

def print_model_memory_usage(model):
    memory_in_bytes = model_memory_usage(model)
    memory_in_megabytes = memory_in_bytes / (1024 ** 2)  # Convert bytes to megabytes

    return memory_in_megabytes