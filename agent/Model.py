
import yaml

from typing import List, Tuple
from tqdm import tqdm
from agent.Local_LLM import Local_LLM
from agent.Online_LLM import Online_LLM
from agent.Conversation import Conversation, HUMAN_SIM, HUMAN_EVAL, LLM, get_role

DEBUG = False

class Model:
    """Abstract base class for large language models."""

    def __init__(self, role, config, needs_confirmation=False, disable_tqdm=True, 
                 model=None, tokenizer=None):
        """Initializes the model."""
        self.role = role
        self.config = config
        self.needs_confirmation = needs_confirmation
        self.disable_tqdm = disable_tqdm

        # Initialise human model
        if self.config["type"] == "local":
            LLM_class = Local_LLM
        else:
            LLM_class = Online_LLM
        self.model = LLM_class(
            self.config,
            model = model,
            tokenizer = tokenizer,
            )
        tqdm.write(f'Initialized {get_role(role)} as {self.config["type"]} model: {self.config["model_config"]["pretrained_model_name_or_path"]}.')

    def sample_actions(self, prompt : Conversation, **kwargs) -> List[str]:
        # convo = Conversation.from_delimited_string(prompt)
        convo = prompt
        return self.generate_text(convo, **kwargs)

    def generate_text(self, convos : Conversation | List[Conversation], batch = False, **kwargs) -> List[str] | List[List[str]]:
        """Generates text from the model.
        Parameters:
            convos: The prompt to use. List of Conversation.
        Returns:
            A list of list of strings.
        """
        convos_is_list = isinstance(convos, list)
        if not convos_is_list:
            convos = [convos]

        chats : List[List[dict]] = []
        # Create prompts from converstation histories
        for convo in convos:
            chat = self.model.apply_chat_format(convo, **kwargs)
            chats.append(chat)
        if DEBUG:
            print("generated prompts")
            print(chats)

        generated_text = []

        if batch:
            raise NotImplementedError
        else:
            if not self.disable_tqdm:
                chats = tqdm(chats)
            for chat in chats:
                output = self.model.generate(chat)
                generated_text.append(output)
        if not convos_is_list:
            generated_text = generated_text[0]
        return generated_text

# Create human_sim, human_eval, and llm_model
def create_human_and_llm(config="agent/llm_config.yaml",human_sim_to_use="human_sim", human_eval_to_use="human_eval", llm_model_to_use ="llm_model", cuda = 0,**kwargs) -> List[Model]:
    with open(config, "r") as f:
        llm_config = yaml.full_load(f)
    llm_config[llm_model_to_use]["model_config"]["device_map"] = cuda
    llm_config[human_sim_to_use]["model_config"]["device_map"] = cuda
    llm_config[human_eval_to_use]["model_config"]["device_map"] = cuda
    models = []
    models_to_use = [human_sim_to_use, human_eval_to_use, llm_model_to_use]
    for model, model_type in zip(models_to_use, [HUMAN_SIM, HUMAN_EVAL, LLM]):
        m = None
        for j, prev_model in enumerate(models): # Reuse the same model if possible
            if not isinstance(prev_model.model, Local_LLM):
                continue
            if llm_config[model]["model_config"]["pretrained_model_name_or_path"] == llm_config[models_to_use[j]]["model_config"]["pretrained_model_name_or_path"]:
                m = prev_model.model.model
                break
        models.append(Model(model_type, llm_config[model], model=m, **kwargs))
    human_sim, human_eval, llm_model = models
    return human_sim, human_eval, llm_model