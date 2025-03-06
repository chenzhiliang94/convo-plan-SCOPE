
from copy import deepcopy
from typing import List, Self
import re

HUMAN_SIM = 0
HUMAN_EVAL = 1
LLM = 2

def get_role(role):
    if role == HUMAN_SIM:
        return "Human"
    elif role == HUMAN_EVAL:
        return "Human eval"
    elif role == LLM:
        return "LLM"

class Conversation:
    def __init__(self, starting_convo : Self | List[dict] | str | None = None, start_with_human : bool = True): #, tokenizer : PreTrainedTokenizer
        self.human_responses : List[str] = []
        self.llm_responses : List[str] = []
        self.full_convo : List[str] = []
        self.order : List[int] = []
        self.start_with_human = start_with_human
        if isinstance(starting_convo, Conversation):
            self.human_responses = starting_convo.human_responses
            self.llm_responses = starting_convo.llm_responses
            self.full_convo = starting_convo.full_convo
            self.order = starting_convo.order
            self.start_with_human = starting_convo.start_with_human
        elif isinstance(starting_convo, list):
            if len(starting_convo) > 0:
                if isinstance(starting_convo[0], dict):
                    if starting_convo[0]["role"] == "system":
                        starting_convo = starting_convo[1:]
                    if starting_convo[0]["role"] == "assistant":
                        self.start_with_human = False
                    for i in starting_convo:
                        self.add_response(i["content"], copy = False)
                else:
                    for i in starting_convo:
                        self.add_response(i, copy = False)
        elif isinstance(starting_convo, str):
            self.add_response(starting_convo, copy = False)

        self.roles = ["user", "assistant"]
        if not self.start_with_human:
            self.roles = self.roles[::-1]

    @classmethod
    def from_delimited_string(cls, string : str, delimiters : List[str] = ["[YOU]: ", "[THEM]: "]) -> Self:
        convo = Conversation()
        regex_pattern = '|'.join(map(re.escape, delimiters))
        for i in re.split(regex_pattern, string):
            convo = convo.add_response(i, copy = False)
        return convo

    def last_is_human(self) -> bool:
        if len(self.order) == 0:
            return not self.start_with_human
        return self.order[-1] != LLM

    def last_is_llm(self) -> bool:
        if len(self.order) == 0:
            return self.start_with_human
        return self.order[-1] == LLM

    def add_human_response(self, response : str, copy : bool = True) -> Self:
        if len(self.order) > 0:
            assert self.order[-1] == LLM, f"Cannot add human response as last response was {get_role(self.order[-1])}."

        if copy:
            obj = deepcopy(self)
        else:
            obj = self

        obj.human_responses.append(response)
        obj.full_convo.append(response)
        obj.order.append(HUMAN_SIM)
        return obj

    def add_llm_response(self, response : str, copy : bool = True) -> Self:
        if len(self.order) > 0:
            assert self.order[-1] != LLM, f"Cannot add llm response as last response was {get_role(self.order[-1])}."

        if copy:
            obj = deepcopy(self)
        else:
            obj = self

        obj.llm_responses.append(response)
        obj.full_convo.append(response)
        obj.order.append(LLM)
        return obj

    def add_response(self, response : str, copy : bool = True) -> Self:
        assert isinstance(response, str)
        if self.last_is_llm() | (len(self.order) == 0 and self.start_with_human):
            return self.add_human_response(response, copy)
        else:
            return self.add_llm_response(response, copy)

    def create_chat(self) -> List[dict]:
        # Generate a list of alternating user/assistant chat

        assert len(self.order) > 0, "No convo yet, cannot generate prompt."

        chat : List[dict] = []
        tmp_convo = self.full_convo
        if len(tmp_convo) % 2 == 0:
            tmp_convo.insert(0, "")
        for i, response in enumerate(tmp_convo):
            chat.append({"role": self.roles[i%2], "content":response})

        return chat
    
    def __repr__(self):
        return str(self)
    
    def __str__(self):
        output = []
        for role, convo in zip(self.order, self.full_convo):
            output.append(f"{get_role(role)}\t: \"{convo}\"")
        output = "\n".join(output)
        return output

    def __add__(self, other):
        if isinstance(other, str):
            other = other.strip()
            # if len(other) == 0:
            #     return self
            return self.add_response(other)
        else:
            raise ValueError("str value is required")
        
    def __eq__(self, other):
        if not isinstance(other, Conversation):
            return False

        return (self.start_with_human == other.start_with_human and
                self.full_convo == other.full_convo)

    def __hash__(self):
        return hash("".join(self.full_convo))
    
    # def __len__(self):
    #     output = ""
    #     for role, convo in zip(self.order, self.full_convo):
    #         output += f"{get_role(role)}\t: \"{convo}\"\n"
    #     return len(output)