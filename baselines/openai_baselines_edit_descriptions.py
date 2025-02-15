import os
from openai import OpenAI
import os
from datetime import datetime
import torch.optim as optim
import transformers
import neptune
from neptune.utils import stringify_unsupported
import hydra
from omegaconf import OmegaConf
import sys
import numpy as np

import sys
sys.path.append("..")
from utils import *
from eval import *
import torch.nn.functional as F

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from itertools import zip_longest

import torch
from tqdm import tqdm
import seaborn as sns

from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput
from datatypes import *
import pandas as pd
from tqdm import tqdm

# Ensure the API key is set in your environment or explicitly provide it
api_key = os.getenv("OPENAI_API_KEY")  # Fetch from environment variable
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

# Initialize the OpenAI client with the API key
client = OpenAI(api_key=api_key)

# Create the chat completion
# completion = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Write a haiku about recursion in programming in C."}
#     ]
# )
# Print the result
# print(completion.choices[0].message.content)

# Function to generate edit representation
def generate_edit_representation(original_code, modified_code):
    """
    Step 1: Generate a textual representation of the edits.
    """
    prompt = (
        f"The original code snippet is:\n{original_code}\n\n"
        f"The modified code snippet is:\n{modified_code}\n\n"
        "Describe the edits required to transform the original code into the modified code. "
        "Make sure the edit description is not more than 5 lines."
        "Make it as generalized as possible."
        # "Provide a structured and detailed explanation of the changes."
    )
    
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=512,
        temperature=0.7
    )
    
    edit_representation = completion.choices[0].message.content
    return edit_representation


def generate_edit_desc(dataset):
    newData = []
    for iter, (index, row) in enumerate(tqdm(dataset.iterrows(), desc="Generating GPT4 Desc", leave=False)):
        a1 = row['code_i_1']
        a2 = row['code_j_1']
        b1 = row['code_i_2']
        b2 = row['code_j_2']
        a_edit = generate_edit_representation(a1, a2)
        b_edit = generate_edit_representation(b1, b2)
        newData.append([row['problemID'], row['problemDescription'], row['studentID_1'], 
                        row['test_case_verdict_i_1'], row['codeID_i_1'], a1, row['score_i_1'], row['score_calc_i_1'],                      
                        row['test_case_verdict_j_1'], row['codeID_j_1'], a2, row['score_j_1'], row['score_calc_j_1'],
                        a_edit                      
                        ])
        newData.append([row['problemID'], row['problemDescription'], row['studentID_2'], 
                        row['test_case_verdict_i_2'], row['codeID_i_2'], a1, row['score_i_2'], row['score_calc_i_2'],                      
                        row['test_case_verdict_j_2'], row['codeID_j_2'], a2, row['score_j_2'], row['score_calc_j_2'],
                        b_edit                      
                        ])
        # if iter == 2:
        #     break

    newdf = pd.DataFrame(newData, columns=['problemID', 'problemDescription',
                                            'studentID', 'test_case_verdict_i', 'codeID_i', 'code_i', 'score_i', 'score_calc_i',  
                                                         'test_case_verdict_j', 'codeID_j', 'code_j', 'score_j', 'score_calc_j', 
                                            'edit_desc'])
    newdf.to_csv('gpt4omini-edit_desc.csv', index=False )
    newdf.to_pickle('gpt4omini-edit_desc.pkl')

@hydra.main(version_base=None, config_path="..", config_name="configs_cer")
def main(configs):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Make reproducible
    set_random_seed(configs.seed)

    data_checkpoint_name = '20250130_212344' #cerd, all, reconstruction =.5
    _, train_set, valid_set, test_set = load_checkpoint_model_and_data(checkpoint_name=data_checkpoint_name, configs=configs)
    
    # Path to the checkpoint
    checkpoint_name = '20241209_165650' # with regularization, if else  
    # checkpoint_name = '20241209_194800' # with regularization, if else, exclusive problems between train and test
    # checkpoint_name = '20241211_195813' #with reg, student split, all problems.
    # checkpoint_name = '20241213_224930' #with reg, student split, all problems. higher reconstruction lambda
    # checkpoint_name = '20241214_000113' #with reg, student split, all problems. t5-large
    # checkpoint_name = '20241215_192723' #with reg, student split, all problems. reconstruction lambda = 1.5
    # checkpoint_name = '20241216_192316' #with reg, student split, all problems. reconstruction lambda = 2. t5-base
    # checkpoint_name = '20241217_212527' #with reg, student split, all problems. reconstruction lambda = 2. code-t5-base
    
    generate_edit_desc(dataset=train_set)

if __name__ == "__main__":
    main()

