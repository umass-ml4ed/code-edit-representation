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

from data_loader import *
from model import *
from trainer import *
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
from collections import defaultdict

from tqdm import tqdm
# from evaluator.CodeBLEU import calc_code_bleu
from main_exp_gen_code_decoder import *


# Function to generate code from a given vector embedding
def generate_code_from_vector(encoder_embedding, model, tokenizer, device):
    """
    Generate code from an encoder embedding using a decoder model.

    Args:
        encoder_embedding (torch.Tensor): The encoder output embedding [batch_size, hidden_size].
        decoder_model (torch.nn.Module): The fine-tuned decoder model.
        tokenizer (transformers.T5Tokenizer): The tokenizer used with the model.
        device (torch.device): The device (CPU or GPU) for computation.

    Returns:
        str: The generated code.
    """
    # Prepare encoder outputs for the decoder
    encoder_embedding = encoder_embedding.unsqueeze(1)  # [batch_size, seq_length=1, hidden_size]
    encoder_outputs = BaseModelOutput(last_hidden_state=encoder_embedding)

    # Generate code using the model's generate method
    with torch.no_grad():
        generated_ids = model.pretrained_decoder.generate(
            encoder_outputs=encoder_outputs,
            max_length=128,
            decoder_start_token_id=tokenizer.pad_token_id  # Ensure correct token ID
        )

    # Decode the generated sequences
    generated_code = [
        tokenizer.decode(seq, skip_special_tokens=True) for seq in generated_ids
    ]

    return generated_code

def find_history_embedding(model, history, pid, mask1, mask2):
    history_pid = history[history['problemID'] == pid]
    # print(len(history), len(history_pid))

    history_pid = history_pid[history_pid['test_case_verdict_i'] == mask1]
    # print(len(history), len(history_pid))

    history_pid = history_pid[history_pid['test_case_verdict_j'] == mask2]
    print(len(history), len(history_pid))
    history_pid.reset_index(drop=True, inplace=True)
    # print(history_pid)
    if len(history_pid) > 0:
        a1 = history_pid.at[0, 'code_i']
        a2 = history_pid.at[0, 'code_i']
        delta, _ = model.get_edit_encodings([a1,a2,a1,a2])
        return delta, a2
    return None, None

def serialize_dataset(dataset):
    dataset.drop(columns=['is_similar'], inplace=True)

    df_a = pd.DataFrame(dataset[['problemID', 'problemDescription','studentID_1', 'test_case_verdict_i_1', 'codeID_i_1', 'code_i_1', 'score_i_1', 'score_calc_i_1', 'test_case_verdict_j_1', 'codeID_j_1', 'code_j_1', 'score_j_1', 'score_calc_j_1',]].values, columns=['problemID', 'problemDescription','studentID', 'test_case_verdict_i', 'codeID_i', 'code_i', 'score_i', 'score_calc_i', 'test_case_verdict_j', 'codeID_j', 'code_j', 'score_j', 'score_calc_j',])
    
    df_b = pd.DataFrame(dataset[['problemID', 'problemDescription','studentID_2', 'test_case_verdict_i_2', 'codeID_i_2', 'code_i_2', 'score_i_2', 'score_calc_i_2', 'test_case_verdict_j_2', 'codeID_j_2', 'code_j_2', 'score_j_2', 'score_calc_j_2',]].values, columns=['problemID', 'problemDescription','studentID', 'test_case_verdict_i', 'codeID_i', 'code_i', 'score_i', 'score_calc_i', 'test_case_verdict_j', 'codeID_j', 'code_j', 'score_j', 'score_calc_j',])

    dataset = pd.concat([df_a, df_b], ignore_index=True)
    return dataset

# Function to generate codes for a batch of inputs
def generate_code_in_batch(model, history, dataset, tokenizer, configs, device):
    collate_fn = CollateForCER(tokenizer=tokenizer, configs=configs, device=device)
    # historydataloader = make_dataloader_experiment(history, collate_fn=collate_fn, configs=configs)
    # testloader = make_dataloader_experiment(dataset, collate_fn=collate_fn, configs=configs)
    allowed_problemIDs = configs['allowed_problem_list']
    input_test_cases = pd.read_csv('data/input_test_cases.csv', index_col=False)
    print(len(history))
    history = serialize_dataset(history)
    print(len(history))

    print(len(dataset))
    dataset = serialize_dataset(dataset)
    print(len(dataset))

    history_bleu = []
    personal_bleu = []
    model.eval()
    with torch.no_grad():
        for iter, (index, row) in enumerate(tqdm(dataset.iterrows(), desc="Generating Model History", leave=False)):
            a1 = row['code_i']
            a1_mask = row['test_case_verdict_i']
            a2 = row['code_j']
            a2_mask = row['test_case_verdict_j']

            pid = row['problemID']
            problem_desc = row['problemDescription']

            input_test_cases_for_this_pid = input_test_cases[input_test_cases['coding_prompt_id'] == int(pid)]
            zeroes = "0" * len(input_test_cases_for_this_pid)

            if a2_mask != a1_mask and a2_mask != zeroes:
                a1_emb = model.get_embeddings(a1)
                Da_hist, b2 = find_history_embedding(model=model, history=history, pid=pid, mask1=a1_mask, mask2=a2_mask)
                if Da_hist != None:
                    a2_gen = generate_code_from_vector(a1_emb + Da_hist, model, tokenizer, device)[0]
                    printCodePairSideBySide(a2, a2_gen, col_width=60)
                    bleu = compute_code_bleu([a2], [a2_gen])
                    history_bleu.append(bleu)
                    bleu = compute_code_bleu([b2], [a2_gen])
                    personal_bleu.append(bleu)

            if len(history_bleu) > 0: 
                print(f"History Bleu: {np.mean(history_bleu)}")
                print(f"Personal Bleu: {np.mean(personal_bleu)}")
                sys.stdout.flush()
                # break

def make_finetuning_dataloader(dataset: pd.DataFrame, collate_fn: callable, tokenizer: T5Tokenizer, configs: dict, n_workers: int = 0, train: bool = True) -> torch.utils.data.DataLoader:
    shuffle = train and not configs.testing
    pytorch_dataset = CERDataset(dataset)
    return torch.utils.data.DataLoader(pytorch_dataset, collate_fn=collate_fn, shuffle=shuffle, batch_size=8, num_workers=n_workers)

@hydra.main(version_base=None, config_path=".", config_name="configs_cer")
def main(configs):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Make reproducible
    set_random_seed(configs.seed)

    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Current Device: ' + str(device))

    # Initialize the model
    tokenizer = create_tokenizer(configs)
    checkpoint_path = configs.model_save_dir

    data_checkpoint_name = '20250130_212344' #cerd, all, reconstruction =.5
    _, train_set, valid_set, test_set = load_checkpoint_model_and_data(checkpoint_name=data_checkpoint_name, configs=configs) #to keep the data constant over experiments
    
    # Path to the checkpoint
    # checkpoint_name = '20241209_165650' # with regularization, if else  
    # checkpoint_name = '20241209_194800' # with regularization, if else, exclusive problems between train and test
    # checkpoint_name = '20241211_195813' #with reg, student split, all problems.
    # checkpoint_name = '20241213_224930' #with reg, student split, all problems. higher reconstruction lambda
    # checkpoint_name = '20241214_000113' #with reg, student split, all problems. t5-large
    # checkpoint_name = '20241215_192723' #with reg, student split, all problems. reconstruction lambda = 1.5
    # checkpoint_name = '20241216_192316' #with reg, student split, all problems. reconstruction lambda = 2. t5-base
    # checkpoint_name = '20241217_212527' #with reg, student split, all problems. reconstruction lambda = 2. code-t5-base

    # checkpoint_name = '20250130_211733' #cerdd, all, reconstruction =.5
    # checkpoint_name = '20250130_212046' #cerdd, all, reconstruction = 1
    # checkpoint_name = '20250130_212102' #cerdd, all, reconstruction = 1.5
    # checkpoint_name = '20250130_212215' #cerdd, all, reconstruction = 2
    # checkpoint_name = '20250130_212223' #cerdd, all, reconstruction = 3

    # checkpoint_name = '20250130_212344' #cerd, all, reconstruction =.5
    # checkpoint_name = '20250130_212343' #cerd, all, reconstruction = 1
    # checkpoint_name = '20250130_213807' #cerd, all, reconstruction = 1.5
    checkpoint_name = '20250130_215832' #cerd, all, reconstruction = 2
    # checkpoint_name = '20250130_220007' #cerd, all, reconstruction = 3
    # checkpoint_name = '20250208_162240' #cerd, all, reconstruction = 4
    # checkpoint_name = '20250208_162301' #cerd, all, reconstruction = 5

    # checkpoint_name = '20250206_190729' #cerd, all, reconstruction = 3, regularization = 2

    # checkpoint_name = '20250211_212450' #cerd, all, reconstruction = 2, codet5-large
    # checkpoint_name = '20250211_212856' #cerd, all, reconstruction = 3, codet5-large
    # checkpoint_name = '20250211_212656' #cerdd, all, reconstruction = 2, codet5-large
    # checkpoint_name = '20250211_213144' #cerdd, all, reconstruction = 3, codet5-large

    print("checkpoint_name = '20250130_215832' #cerd, all, reconstruction = 2")
    cerd_model, _, _, _ = load_checkpoint_model_and_data(checkpoint_name=checkpoint_name, configs=configs)

    # Instantiate the finetune model
    # finetune_model = FinetuneDecoderModel(encoder_model, decoder_model, cer_model, tokenizer, configs, device)

    # Create a DataLoader for the finetuning task
    # trainset, validset, testset = read_data(configs)
    # train_dataloader = make_finetuning_dataloader(train_set, DecoderCollateForEdit(tokenizer, configs, device), tokenizer, configs)
    # test_dataloader = make_finetuning_dataloader(test_set, DecoderCollateForEdit(tokenizer, configs, device), tokenizer, configs)

    # seq_test_set = read_seq_data_with_filter(configs=configs, filterset=test_set)
 
    # Example usage
    generated_code = generate_code_in_batch(model=cerd_model, history=train_set, dataset=test_set, tokenizer=tokenizer, configs=configs, device=device)

if __name__ == "__main__":
    main()
