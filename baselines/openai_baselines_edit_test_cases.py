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

# Function to generate edit representation
def generate_code(original_code, input_test_cases, desired_mask, problem_desc):
    prompt = (
        "You are given a code and the input test cases being used to test the code for a problem. After the code there is a mask . In the mask a 0 means the code passes the corresponding test case, else fails. You need to produce a code that is similar to the given code, but produces a mask as the given mask."
        f"\nProblem:\n{problem_desc}"
        f"\nCode:\n{original_code}\n"
        f"\nMask: {desired_mask}\n\n"
        f"\nInput:\n{input_test_cases['input'].to_list()}"#to_string(index=False)}"
        f"\nExpected Output:\n{input_test_cases['expected_output'].to_list()}"#to_string(index=False)}"
        "\nGenerate a code adhering to the given mask to the original code and provide only the modified code. Write nothing else. Not even the programming language and markdown tags."

    )
    # print(prompt)
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=512,
        temperature=0.7
    )
    
    edit_representation = completion.choices[0].message.content
    return edit_representation


def generate_codes_using_test_cases(dataset):
    input_test_cases = pd.read_csv('../data/input_test_cases.csv', index_col=False)
    history_bleu = []
    new_data = []
    for iter, (index, row) in enumerate(tqdm(dataset.iterrows(), desc="Generating GPT4 Desc", leave=False)):
        a1 = row['code_i_1']
        a1_mask = row['test_case_verdict_i_1']
        a2 = row['code_j_1']
        a2_mask = row['test_case_verdict_j_1']

        b1 = row['code_i_2']
        b1_mask = row['test_case_verdict_i_2']
        b2 = row['code_j_2']
        b2_mask = row['test_case_verdict_j_2']

        pid = row['problemID']
        problem_desc = row['problemDescription']
        input_test_cases_for_this_pid = input_test_cases[input_test_cases['coding_prompt_id'] == int(pid)]
        zeroes = "0" * len(input_test_cases_for_this_pid)

        if a2_mask != a1_mask and a2_mask != zeroes:
            a2_gen = generate_code(original_code=a1, input_test_cases=input_test_cases_for_this_pid, desired_mask=a2_mask, problem_desc=problem_desc)
            # printCodePairSideBySide(a2, a2_gen, col_width=60)
            # bleu = compute_code_bleu([a2], [a2_gen])
            # history_bleu.append(bleu)
            new_data.append([row['problemID'], row['problemDescription'], row['studentID_1'], 
                        row['test_case_verdict_i_1'], row['codeID_i_1'], a1, row['score_i_1'], row['score_calc_i_1'],                      
                        row['test_case_verdict_j_1'], row['codeID_j_1'], a2, row['score_j_1'], row['score_calc_j_1'],
                        a2_gen                      
                        ])

        if b2_mask != b1_mask and b2_mask != zeroes:
            b2_gen = generate_code(original_code=b1, input_test_cases=input_test_cases_for_this_pid, desired_mask=b2_mask, problem_desc=problem_desc)
            # printCodePairSideBySide(b2, b2_gen, col_width=60)
            # bleu = compute_code_bleu([b2], [b2_gen])
            # history_bleu.append(bleu)
            new_data.append([row['problemID'], row['problemDescription'], row['studentID_2'], 
                        row['test_case_verdict_i_2'], row['codeID_i_2'], a1, row['score_i_2'], row['score_calc_i_2'],                      
                        row['test_case_verdict_j_2'], row['codeID_j_2'], a2, row['score_j_2'], row['score_calc_j_2'],
                        b2_gen
                        ])
        if len(history_bleu) > 0: 
            print(f"History Bleu: {np.mean(history_bleu)}")
            sys.stdout.flush()
            
    newdf = pd.DataFrame(new_data, columns=['problemID', 'problemDescription',
                                            'studentID', 'test_case_verdict_i', 'codeID_i', 'code_i', 'score_i', 'score_calc_i',  
                                                         'test_case_verdict_j', 'codeID_j', 'code_j', 'score_j', 'score_calc_j', 
                                            'code_gpt'])
    newdf.to_csv('gpt4o-code-test-case.csv', index=False )
    newdf.to_pickle('gpt4o-code-test-case.pkl')

@hydra.main(version_base=None, config_path="..", config_name="configs_cer")
def main(configs):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Make reproducible
    set_random_seed(configs.seed)

    data_checkpoint_name = '20250130_212344' #cerd, all, reconstruction =.5
    _, train_set, valid_set, test_set = load_checkpoint_model_and_data(checkpoint_name=data_checkpoint_name, configs=configs)
    
    # Path to the checkpoint
    # checkpoint_name = '20241209_165650' # with regularization, if else  
    # checkpoint_name = '20241209_194800' # with regularization, if else, exclusive problems between train and test
    # checkpoint_name = '20241211_195813' #with reg, student split, all problems.
    # checkpoint_name = '20241213_224930' #with reg, student split, all problems. higher reconstruction lambda
    # checkpoint_name = '20241214_000113' #with reg, student split, all problems. t5-large
    # checkpoint_name = '20241215_192723' #with reg, student split, all problems. reconstruction lambda = 1.5
    # checkpoint_name = '20241216_192316' #with reg, student split, all problems. reconstruction lambda = 2. t5-base
    # checkpoint_name = '20241217_212527' #with reg, student split, all problems. reconstruction lambda = 2. code-t5-base
    
    generate_codes_using_test_cases(dataset=test_set)

if __name__ == "__main__":
    main()

