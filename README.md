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

# SETUP
0. From the repo home, run `mkdir transition_models/deterministic` to create an empty folder.
1. Download the files from https://drive.google.com/drive/folders/1NLK8f8aV476frtIuMwC8IgwTVPxbOB6S and move them into `transition_models/deterministic/`. There should be four folders (seed_0_batch_2048, seed_1_batch_2048, seed_2_batch_2048, seed_3_batch_2048) added.
2. `pip3 install -r requirements.txt`

# Given a conversation starter, get the best LLM response.
A simple use case is that given a conversation starter, we want to use SCOPE to simply
0. Go to `evaluation/conversation_starter.txt`, you should place one question that you want to ask the LLM with here. (Currently, we do not support multiple questions for this section, but later sections do).
1. Run `python3 -u evaluation/run_evaluation_singular.py --reward_func=length_human --cuda_for_llm_reward=0 --cuda_for_llm_reward=1 --lr=0.0001 --evaluation_depth=4 --mcts_time=5 --agent=pure_online --result_file=camera --trials=1 --evaluation_data=conversation_starter.txt 2>&1 | tee output.out` The reward function and parameters can be adjusted. We provide more details what they mean later.
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
1. Run `python3 -u evaluation/run_evaluation.py --reward_func=length_human --cuda_for_llm_reward=0 --cuda_for_llm_reward=1 --lr=0.0001 --evaluation_depth=4 --mcts_time=2 --agent=random --result_file=output --trials=1 --evaluation_data=camera_ready.txt 2>&1 | tee output.out`
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
Human   : ""Wow, that sounds like an incredible experience! Did you get a chance to explore the temple grounds and learn more about its history during your stay?""]
```



