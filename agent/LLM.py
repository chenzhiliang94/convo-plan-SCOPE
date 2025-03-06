"""Contains classes for querying large language models."""

from abc import ABC, abstractmethod
from agent.Conversation import Conversation
from typing import List

class LLM(ABC):
    system_prompt = ""

    @abstractmethod
    def generate(self, chat : List[dict]) -> List[str]:
        pass

    # Create chat and add add system prompt
    def apply_chat_format(self, convo : Conversation, **kwargs) -> List[dict]:
        chat = convo.create_chat()

        if isinstance(self.system_prompt, ICL_prompt):
            if "context" in kwargs and kwargs["context"] is not None:
                self.system_prompt.set_context(kwargs["context"])
            elif "human_description" in kwargs and kwargs["human_description"] is not None:
                self.system_prompt.set_context(kwargs["human_description"])

        if len(self.system_prompt) > 0:
            if chat[0]["role"] == "assistant":
                chat[0]["content"] = self.system_prompt + "\n\n" + chat[0]["content"]
                chat = chat[1:]
            elif self.tokenizer_has_system_prompt:
                    chat.insert(0,{"role": "system", "content":str(self.system_prompt)})
            # else:
            #     chat.insert(0,{"role": "assistant", "content":self.system_prompt})

        if isinstance(self.system_prompt, ICL_prompt):
            self.system_prompt.clear_context()

        return chat

class ICL_prompt:
    def __init__(self, model_config):
        self.use_icl = "sys_prompt_pre" in model_config and "sys_prompt_context" in model_config and "sys_prompt_post" in model_config
        if not self.use_icl:
            self.sys_prompt = model_config["sys_prompt"]
        else:
            self.sys_prompt_pre = model_config["sys_prompt_pre"]
            self.sys_prompt_context = model_config["sys_prompt_context"]
            self.sys_prompt_post = model_config["sys_prompt_post"]
        self.context = ""

    def set_context(self, context):
        if not self.use_icl:
            return
        context = "\n".join(context)
        self.context = self.sys_prompt_context + "\n\n" + context + "\n\n"

    def clear_context(self):
        self.context = ""
    
    def __str__(self):
        if not self.use_icl:
            return self.sys_prompt
        if self.context == "":
            return self.sys_prompt_pre + self.sys_prompt_post
        return self.sys_prompt_pre + self.context + self.sys_prompt_post
    
    def __len__(self):
        return len(str(self))