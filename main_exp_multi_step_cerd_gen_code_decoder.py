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

# Function to generate codes for a batch of inputs
def generate_code_in_batch(model, history, dataset, tokenizer, configs, device):
    collate_fn = CollateForCER(tokenizer=tokenizer, configs=configs, device=device)
    historydataloader = make_dataloader_experiment(history, collate_fn=collate_fn, configs=configs)
    allowed_problemIDs = configs['allowed_problem_list']

    model.eval()
    with torch.no_grad():
        history_vectors = []  # Initialize an empty list to store vectors
        for batch in tqdm(historydataloader, desc="Generating Code Embeddings", leave=False):
            concatenated_inputs = batch['inputs']
            labels = batch['labels']
            labels = labels.to(device).to(torch.float32)

            # Tokenize inputs
            tokenized_inputs = tokenizer(concatenated_inputs, return_tensors="pt", padding=True, truncation=True).to(device)

            # Get edit encodings
            Da, Db = model.get_edit_encodings_tokenized(tokenized_inputs)

            # Append the vectors to the list
            history_vectors.append(Da.cpu())  # Add Da to the list
            history_vectors.append(Db.cpu())  # Add Db to the list

        # Concatenate all tensors into a single tensor
        history_vectors = torch.cat(history_vectors, dim=0)
        vector_set_norm = torch.nn.functional.normalize(history_vectors, dim=1).to(device)
        # Print the shape of the final tensor
        print(history_vectors.shape)

        multi_step_bleu_by_length = {}  # Dictionary to store BLEU scores by sequence length
        
        for pid in tqdm(allowed_problemIDs):
            pid_dataset = dataset[dataset['problemID'].isin([pid])]
            all_students = pd.unique(pid_dataset['studentID'].values.ravel())
            for sid in all_students:
                sid_dataset = pid_dataset[pid_dataset['studentID'].isin([sid])]
                data = []
                if len(sid_dataset) > 2: 
                    for index, row in sid_dataset.iterrows():
                        data.append(row)
                    
                    for i in range(len(data) - 1):
                        inputs = [data[i]['code'], data[i+1]['code'], data[i]['code'], data[i+1]['code']]
                        tokenized_inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).to(device)
                        D, _ = model.get_edit_encodings_tokenized(tokenized_inputs)
                        embeddings = model.get_embeddings_tokenized(tokenized_inputs)
                        batch_size = embeddings.shape[0] // 4
                        A1, A2, B1, B2 = model.batch_unpack(inputs, batch_size)
                        A1_emb, A2_emb, B1_emb, B2_emb = model.batch_unpack(embeddings, batch_size)
                        for j in range(i + 2, len(data)):
                            inputs = [data[j-1]['code'], data[j]['code'], data[j-1]['code'], data[j]['code']]
                            Da, _ = model.get_edit_encodings(inputs)
                            D += Da
                            # Normalize the vectors to compute cosine similarity
                            input_vector_norm = torch.nn.functional.normalize(D, dim=1).to(device)
                            # Compute cosine similarity
                            similarities = torch.mm(vector_set_norm, input_vector_norm.T).squeeze()
                            closest_idx = torch.argmax(similarities)

                            D_closest = history_vectors[closest_idx].to(device)
                            code_gen = generate_code_from_vector(A1_emb + D_closest, model, tokenizer, device)
                            bleu = compute_code_bleu([data[j]['code']], code_gen)

                            # Store BLEU scores by sequence length
                            seq_length = j - i
                            if seq_length not in multi_step_bleu_by_length:
                                multi_step_bleu_by_length[seq_length] = []
                            multi_step_bleu_by_length[seq_length].append(bleu)

        # Print average BLEU scores for each sequence length
        for seq_length, bleu_scores in multi_step_bleu_by_length.items():
            avg_bleu = np.mean(bleu_scores)
            print(f"Average BLEU for sequence length {seq_length}: {avg_bleu}") 


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

    checkpoint_name = '20250130_212344' #cerd, all, reconstruction =.5
    # checkpoint_name = '20250130_212343' #cerd, all, reconstruction = 1
    # checkpoint_name = '20250130_213807' #cerd, all, reconstruction = 1.5
    # checkpoint_name = '20250130_215832' #cerd, all, reconstruction = 2
    # checkpoint_name = '20250130_220007' #cerd, all, reconstruction = 3
    print("checkpoint_name = '20250130_212344' #cerd, all, reconstruction =.5")
    cerd_model, train_set, valid_set, test_set = load_checkpoint_model_and_data(checkpoint_name=checkpoint_name, configs=configs)

    # Instantiate the finetune model
    # finetune_model = FinetuneDecoderModel(encoder_model, decoder_model, cer_model, tokenizer, configs, device)

    # Create a DataLoader for the finetuning task
    # trainset, validset, testset = read_data(configs)
    # train_dataloader = make_finetuning_dataloader(train_set, DecoderCollateForEdit(tokenizer, configs, device), tokenizer, configs)
    # test_dataloader = make_finetuning_dataloader(test_set, DecoderCollateForEdit(tokenizer, configs, device), tokenizer, configs)

    seq_test_set = read_seq_data_with_filter(configs=configs, filterset=test_set)
 
    # Example usage
    generated_code = generate_code_in_batch(model= cerd_model, history=train_set, dataset=seq_test_set, tokenizer=tokenizer, configs=configs, device=device)

if __name__ == "__main__":
    main()
