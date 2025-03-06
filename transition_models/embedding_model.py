import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from reward.Llama_2_Guard_Reward import Llama_2_Guard_Reward
from typing import List
from agent.Conversation import Conversation

class embedding_model_mistral():
    def __init__(self, tokenizer, model, to_normalize = False, cuda = torch.device('cuda:5')) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.output_dim = 4096
        self.cuda = cuda
        self.to_normalize = to_normalize
    
    def last_token_pool(self, last_hidden_states: Tensor,
        attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    
    # input is a single text string
    def embed(self, text):
        
        with torch.no_grad():
            batch_dict = self.tokenizer([str(text)], max_length=self.output_dim, padding=True, truncation=True, return_tensors="pt").to(self.cuda)
            outputs = self.model(**batch_dict)
            embeddings = self.last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            
            if self.to_normalize:
                return F.normalize(embeddings, p=2, dim=1)[0]
            return embeddings[0]
        
class embedding_model_nomic():
    def __init__(self, tokenizer, model, to_normalize=False, cuda = torch.device('cuda:5')) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.output_dim = 768
        self.cuda = cuda
        self.to_normalize = to_normalize
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # input is a single text string
    def embed(self, text):
        # tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', model_max_length=8192)
        # model = AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v1', trust_remote_code=True, rotary_scaling_factor=2)
        self.model.eval()

        encoded_input = self.tokenizer([str(text)], padding=True, truncation=True, return_tensors='pt').to(self.cuda)

        with torch.no_grad():
            model_output = self.model(**encoded_input)

        embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        
        if self.to_normalize:
            return F.normalize(embeddings, p=2, dim=1)[0]
        return embeddings[0]
        
        #embeddings = F.normalize(embeddings, p=2, dim=1)

class embedding_model_llama(Llama_2_Guard_Reward):
    def __init__(self, tokenizer = None, model = None, output_dim = 1024, to_normalize=None, cuda = torch.device('cuda:5'), seed = 42) -> None:
        super().__init__(
            model = model, device_map = cuda, random_projection = output_dim, random_proj_seed = seed
            )
        self.output_dim = output_dim