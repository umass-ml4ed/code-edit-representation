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

def serialize_dataset(dataset):
    dataset.drop(columns=['is_similar'])#, inplace=True)

    df_a = pd.DataFrame(dataset[['problemID', 'problemDescription','studentID_1', 'test_case_verdict_i_1', 'codeID_i_1', 'code_i_1', 'score_i_1', 'score_calc_i_1', 'test_case_verdict_j_1', 'codeID_j_1', 'code_j_1', 'score_j_1', 'score_calc_j_1',]].values, columns=['problemID', 'problemDescription','studentID', 'test_case_verdict_i', 'codeID_i', 'code_i', 'score_i', 'score_calc_i', 'test_case_verdict_j', 'codeID_j', 'code_j', 'score_j', 'score_calc_j',])
    
    df_b = pd.DataFrame(dataset[['problemID', 'problemDescription','studentID_2', 'test_case_verdict_i_2', 'codeID_i_2', 'code_i_2', 'score_i_2', 'score_calc_i_2', 'test_case_verdict_j_2', 'codeID_j_2', 'code_j_2', 'score_j_2', 'score_calc_j_2',]].values, columns=['problemID', 'problemDescription','studentID', 'test_case_verdict_i', 'codeID_i', 'code_i', 'score_i', 'score_calc_i', 'test_case_verdict_j', 'codeID_j', 'code_j', 'score_j', 'score_calc_j',])

    dataset = pd.concat([df_a, df_b], ignore_index=True)
    return dataset


def find_history_code(history, pid, mask1, mask2):
    history_pid = history[history['problemID'] == pid]
    # print(len(history), len(history_pid))

    history_pid = history_pid[history_pid['test_case_verdict_i'] == mask1]
    # print(len(history), len(history_pid))

    history_pid = history_pid[history_pid['test_case_verdict_j'] == mask2]
    # print(len(history), len(history_pid))
    history_pid.reset_index(drop=True, inplace=True)
    # print(history_pid)
    if len(history_pid) > 0:
        a1 = history_pid.at[0, 'code_i']
        a2 = history_pid.at[0, 'code_i']
        return a2
    return None

# Function to generate edit representation
def generate_code(gptdf, cid1, cid2):
    gptdf = gptdf[gptdf['codeID_i'] == cid1]
    gptdf = gptdf[gptdf['codeID_j'] == cid2]
    gptdf.reset_index(drop=True, inplace=True)
    return gptdf.at[0, 'code_gpt']
    
def generate_codes_using_test_cases(dataset, history, configs):
    input_test_cases = pd.read_csv('../data/input_test_cases.csv', index_col=False)
    
    gptdf = pd.read_csv('gpt4o-code-test-case.csv')
    
    allowed_problemIDs = configs['allowed_problem_list']
    dataset = dataset[dataset['problemID'].isin(allowed_problemIDs)]

    dataset = serialize_dataset(dataset)
    history = serialize_dataset(history)

    history_bleu = []
    personal_bleu = []
    for iter, (index, row) in enumerate(tqdm(dataset.iterrows(), desc="Generating GPT4 Desc", leave=False)):
        a1 = row['code_i']
        a1_mask = row['test_case_verdict_i']
        a1_id = row['codeID_i']
        a2 = row['code_j']
        a2_mask = row['test_case_verdict_j']
        a2_id = row['codeID_j']

        pid = row['problemID']
        problem_desc = row['problemDescription']
        
        input_test_cases_for_this_pid = input_test_cases[input_test_cases['coding_prompt_id'] == int(pid)]
        zeroes = "0" * len(input_test_cases_for_this_pid)

        if a2_mask != a1_mask and a2_mask != zeroes:
            a2_gen = generate_code(gptdf=gptdf, cid1=a1_id, cid2=a2_id)
            # printCodePairSideBySide(a2, a2_gen, col_width=60)
            bleu = compute_code_bleu([a2], [a2_gen])
            history_bleu.append(bleu)
            b2 = find_history_code(history=history, pid=pid, mask1=a1_mask, mask2=a2_mask)
            if b2 != None:
                bleu = compute_code_bleu([b2], [a2_gen])
                personal_bleu.append(bleu)

    if len(history_bleu) > 0: 
        print(f"History Bleu: {np.mean(history_bleu)}")
        print(f"Personal Bleu: {np.mean(personal_bleu)}")
        sys.stdout.flush()

@hydra.main(version_base=None, config_path="..", config_name="configs_cer")
def main(configs):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Make reproducible
    set_random_seed(configs.seed)

    data_checkpoint_name = '20250130_212344' #cerd, all, reconstruction =.5
    _, train_set, valid_set, test_set = load_checkpoint_model_and_data(checkpoint_name=data_checkpoint_name, configs=configs)
    
    generate_codes_using_test_cases(dataset=test_set, history=train_set, configs=configs)

if __name__ == "__main__":
    main()

