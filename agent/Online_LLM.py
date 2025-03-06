"""Contains classes for querying OpenAI large language models."""

from typing import List
from agent.LLM import LLM
from openai import OpenAI

class Online_LLM(LLM):
    def __init__(self, model_config, **kwargs):
        self.model_name = model_config["name"]
        self.tokenizer_has_system_prompt = True

        self.generation_config = model_config["generation_config"]
        self.system_prompt = model_config["sys_prompt"]

        self.client = OpenAI()

    def generate(self, chat : List[dict], **kwargs) -> List[str]:
        print("generating responses in chatgpt。。。")
        print(chat)
        output = self.client.chat.completions.create(
            model=self.model_name,
            messages=chat,
            **self.generation_config,
            **kwargs
        )
        output = [i.message.content for i in output.choices]
        return output