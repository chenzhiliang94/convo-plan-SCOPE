import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from reward.rewards_import import *
import torch
import datasets
from tqdm import tqdm
import sys
import numpy as np
import os

if __name__ == "__main__":

    dataset = datasets.load_dataset("lmsys/lmsys-chat-1m", split=f"train", streaming=True)

    start = 0
    end = 1_000_000
    # end = 2000

    print("starting at", start, "ending at", end)

    # Random projection to reduce the dimension of the embedding vectors, set to None to use the original 8192 dim vectors
    # random_projection = None
    random_projection = 1024

    reward = Llama_2_Guard_Reward(random_projection=random_projection)

    embeddings = []
    failed = []
    for i, conversation in zip(tqdm(range(start, end)), iter(dataset)):
        conversation = conversation['conversation']
        e = []
        for j in range(len(conversation)):
            try:
                e.append(reward.embed(conversation[:j+1]).half())
            except:
                # Failed likely due to context length exceeding GPU VRAM, to be dealt with separately
                failed.append(i)
                break
        if len(e) == 0:
            embeddings.append(torch.zeros(0, random_projection, dtype=torch.float16))
        else:
            embeddings.append(torch.stack(e))

    # Save as float16 to save space
    dataset = datasets.Dataset.from_dict(
        {
            'embeddings': embeddings,
        }, 
        features=datasets.Features(
            {
                'embeddings': datasets.Array2D(shape=(None, random_projection), dtype='float16')
            }
        )
    )

    os.makedirs(f"embeddings", exist_ok=True)
    dataset.save_to_disk(f"embeddings/lmsys-chat-1m_embeddings_{random_projection}")

    # Save failed indices to deal with separately (Run on GPU with more VRAM)
    np.save(f"embeddings/lmsys-chat-1m_embeddings_{random_projection}/failed.npy", np.array(failed, dtype=int))