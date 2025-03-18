# Broaden your SCOPE! Efficient Multi-turn Conversation Planning for LLMs with Semantic Space

This is the official repo for ICLR 2025 Spotlight paper "Broaden your SCOPE! Efficient Multi-turn Conversation Planning for LLMs with Semantic Space".

To cite this works, please use the following Bibtex: 
```
@inproceedings{
chen2025scope,
title={Broaden your {SCOPE}! Efficient Multi-turn Conversation Planning for {LLM}s with Semantic Space},
author={Zhiliang Chen and Xinyuan Niu and Chuan-Sheng Foo and Bryan Kian Hsiang Low},
booktitle={Proc. ICLR},
year={2025},
url={https://openreview.net/forum?id=3cgMU3TyyE}
}
```

# Overview
SCOPE consists of two phase. The learning phase has already been done and we have uploaded the models in this repository. Hence, users can simply use SCOPE during runtime to find the best response in a conversation.<br>

![SCOPE overview image](https://github.com/user-attachments/assets/a37909ce-7b30-4321-bbc0-2e24eba6c129)

# SETUP (DO THIS BEFORE ANYTHING ELSE)
0. From the repo home, run `mkdir transition_models/deterministic` to create an empty folder.
1. Download the files from https://drive.google.com/drive/folders/1NLK8f8aV476frtIuMwC8IgwTVPxbOB6S and move them into `transition_models/deterministic/`. There should be four folders (seed_0_batch_2048, seed_1_batch_2048, seed_2_batch_2048, seed_3_batch_2048) added.
2. `pip3 install -r requirements.txt`

# Given a conversation starter, get the best LLM response.
A simple use case is that given a conversation starter, we want to use SCOPE to simply

0. Go to `evaluation/conversation_starter.txt`, you should place one question that you want to ask the LLM with here. (Currently, we do not support multiple questions for this section, but later sections do).
1. Run `python3 -u evaluation/run_evaluation_singular.py --reward_func=length_human --cuda_for_llm_reward=0 --cuda_for_q_embedding_transition=1 --lr=0.0001 --evaluation_depth=4 --mcts_time=5 --agent=pure_online --result_file=camera --trials=1 --evaluation_data=conversation_starter.txt 2>&1 | tee output.out` The reward function and parameters can be adjusted. We provide more details what they mean later.
2. You should see the following output. We see the LLM response options (we can adjust the number of proposed responses, see later sections), and their associated learnt Q values. The higher Q value indicates better cumulative reward that we think a certain response has (based on our MCTS forward simulation in semantic space)
```
conversation starter:  Can you tell me something about Singapore, the place where ICLR 2025 is held?
possible actions:
0: Singapore is a great destination! It's a modern and efficient city-state with a rich cultural heritage. The city is known for its cleanliness, food, and Gardens by the Bay. ICLR 2025 will likely take place in the Marina Bay Sands Expo and Convention Centre, which is a popular venue for conferences and events.
1: ICLR 2025 is indeed being held in Singapore! It's a great city-state with a mix of Asian and Western cultures. You can expect to enjoy the vibrant food scene, beautiful gardens, and world-class infrastructure.
2: Yes, Singapore is a popular destination for conferences and events! ICLR 2025 will likely take place in the city-state's vibrant financial district, surrounded by iconic landmarks like the Marina Bay Sands and Gardens by the Bay.
3: Singapore is a modern and efficient city-state with a blend of Asian and Western cultures. It's known for its cleanliness, food, and Gardens by the Bay. The ICLR 2025 conference will likely take place in the city's central business district, which is easily accessible by public transportation.
4: Singapore is a modern and vibrant city-state with a rich cultural heritage. It's known for its cleanliness, safety, and efficiency. The city has a blend of Asian and Western influences, with a mix of traditional and modern architecture. ICLR 2025 will likely take place in one of the many convention centers or hotels in the city.
Learnt Q value rewards:  [tensor(1.4333), tensor(1.4436), tensor(1.3992), tensor(1.4457), tensor(1.4493)]
```

# Given a conversation starter, perform multi-step evaluation in a real conversation and produce the cumulative rewards.
0. Certainly, we might want to verify if SCOPE really did choose LLM responses that really lead to higher cumulative rewards in an actual conversation. You can certainly wrap `run_evaluation_singular.py` with an iterative loop to evaluate this. For convenience, we have introduced a wrapper to help evaluate the cumulative rewards actually produced by responses selected by SCOPE in a multi-turn conversation.
1. Run `python3 -u evaluation/run_evaluation.py --reward_func=length_human --cuda_for_llm_reward=0 --cuda_for_q_embedding_transition=1 --lr=0.0001 --evaluation_depth=4 --mcts_time=2 --agent=random --result_file=output --trials=1 --evaluation_data=camera_ready.txt 2>&1 | tee output.out`
2. We can observe the cumulative rewards and the actual conversation generated from our LLM responses.
```
all rewards from trials:  [1.1600000000000001]
mean:  1.1600000000000001
std error:  nan
  0%|          | 0/1 [00:18<?, ?it/s]
time taken for all trials: 18.15906810760498
input conversation starter:  Can you tell me about how conference locations are selected? 

conversation generated:  [Human : "Can you tell me about how conference locations are selected?"
LLM     : "Typically, conference organizers consider factors such as accessibility, amenities, and capacity when selecting a location. They may also look at the location's reputation, safety record, and potential for networking opportunities."
Human   : ""And what about the local culture and attractions? I think that can also play a big role in choosing a conference location, don't you?""
LLM     : "Absolutely! The local culture and attractions can be a major draw for attendees, and organizers often try to choose a location that offers a unique experience that complements the conference's theme or topic."
Human   : ""That's a great point! Have you ever attended a conference where the location was particularly memorable or unique? I'd love to hear about it.""
LLM     : "Actually, I attended a conference in Tokyo once, and the location was a historic temple that was converted into a conference center. It was really unique and added to the overall experience of the conference."
Human   : ""That sounds amazing! What was it like to have a conference in such a historic setting? Did it add any challenges or benefits to the event?""
LLM     : "It was fascinating! The temple's architecture and atmosphere added a sense of serenity and tranquility to the conference, which was a nice contrast to the usual conference settings. However, it did present some logistical challenges, such as limited amenities and restricted access to certain areas."
```
# Parameter information
Here, we highlight the important parameters that you can adjust.
`python3 -u evaluation/run_evaluation.py --reward_func=length_human --cuda_for_llm_reward=0 --cuda_for_q_embedding_transition=1 --lr=0.0001 --evaluation_depth=4 --mcts_time=5 --agent=pure_online --result_file=output --trials=1 --evaluation_data=conversation_starter.txt`
- **reward_func**
  - `length_human`: Maximize the cumulative token length in the human replies (which we have no direct control of; we can only use the LLM response to implicitly control the length of human replies) in a conversation.
  - `length_both`: Maximize the cumulative token length of entire conversation (including the LLM response). This is easier that `length_human` because we actually have direct control of length of LLM response.
  - `harmful`: Maximizes the harmlessness of the entire conversation (where harmlessness = negative of safety score given by `Meta-Llama-Guard-2-8B`)
- **agent**
    - `random`: Choose a random response
    - `pure_online`: Use vanilla MCTS to search for the next best response out of the proposed LLM candidates. (Yu et al., 2023; Koh et al., 2024)
    - `semantic_online`: Our method, which uses SCOPE to perform MCTS in semantic space. We use the transition and reward model learnt from the first learning phase.
- **cuda_for_llm_reward** and **cuda_for_q_embedding_transition** specifies which CUDA device to use. In general, setting them to the same device is fine, but you might encounter OOM issues if the conversation has many tokens. Using two GPUs is safer (especially when you run `run_evaluation.py`, which loads two LLMs, one to generate responses, and another for evaluation; so we load both onto different devices).
- **lr**: learning rate of the Q function used in SCOPE.
- **evaluation_depth**: This is only applicable for `run_evaluation.py` and indicates the number of evaluation steps for the entire conversation. For example, `--evaluation_depth=4` means we ask the LLM for responses 4 times (resulting in a conversation where the LLM and human each speaks for 4 times).
- **mcts_time**: time to run MCTS for. For SCOPE, we use around 3-10 seconds. Increasing this number allows us to plan more, but takes longer.
- **result_file**: the output file name which we print the results to.

# Define your own reward function
- We might be interested in adding new reward functions. Currently, reward functions are defined in e.g., `reward/Embedding_Length_Reward.py` and `reward\Llama_2_Guard_Reward.py`. Notice that both classes contain the function
`def get_reward(self, prev_state : Conversation | tuple, action : str | tuple, human_response : str | tuple | None) -> float:`. This functions tells us, given a previous conversation state, a specific action (LLM response) and a transition to the next human response, what is the associated rewards? This also corresponds to the instantaneous reward at one transition step in the MDP. A new reward class needs to have this function. Furthermore, this new reward class needs to calculate the reward associated with each point in semantic space (needs to be learnt) to perform planning with SCOPE and the reward associated with real conversation text, for evaluation purposes.
- To train a new reward model that knows the instantaneous reward associated with each point in semantic space, you can take any text data, find its ground-truth reward label and project the text into embedding space with `Meta-Llama-Guard-2-8B` (we use this as our embedding model in our paper). Hence, your reward model needs to learn the mapping between the embedding and the reward label (e.g., using a neural network). This can be learnt offline and loaded during planning. A good example to start is to look at `reward/Embedding_Length_Reward.py`, which loads a `embedding_length_reward` torch neural network model that predicts the reward associated with an embedding tuple.



