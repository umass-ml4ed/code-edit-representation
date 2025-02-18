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
def generate_code_in_batch(model, dataset, tokenizer, configs, device, flag_data_flow=False):
    collate_fn = CollateForCER(tokenizer=tokenizer, configs=configs, device=device)
    dataloader  = make_dataloader_experiment(dataset , collate_fn=collate_fn, configs=configs)

    model.eval()
    generated_codes = []
    code_bleu = []
    edit_bleu = []
    cross_bleu = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating Code Embeddings", leave=False):
            concatenated_inputs = batch['inputs']
            labels = batch['labels']
            labels = labels.to(device).to(torch.float32)

            tokenized_inputs = tokenizer(concatenated_inputs, return_tensors="pt", padding=True, truncation=True).to(device)
            Da, Db = model.get_edit_encodings_tokenized(tokenized_inputs)

            embeddings = model.get_embeddings_tokenized(tokenized_inputs)
            batch_size = embeddings.shape[0] // 4
            A1, A2, B1, B2 = model.batch_unpack(concatenated_inputs, batch_size)
            A1_emb, A2_emb, B1_emb, B2_emb = model.batch_unpack(embeddings, batch_size)

            # code_A1 = generate_code_from_vector(A1_emb, model, tokenizer, device)
            # bleu = compute_code_bleu(A1, code_A1, flag_data_flow)
            # code_bleu += bleu
            # code_A2 = generate_code_from_vector(A2_emb, model, tokenizer, device)
            # bleu = compute_code_bleu(A2, code_A2, flag_data_flow)
            # code_bleu += bleu
            # code_B1 = generate_code_from_vector(B1_emb, model, tokenizer, device)
            # bleu = compute_code_bleu(B1, code_B1,flag_data_flow)
            # code_bleu += bleu
            # code_B2 = generate_code_from_vector(B2_emb, model, tokenizer, device)
            # bleu = compute_code_bleu(B2, code_B2,flag_data_flow)
            # code_bleu += bleu

            code_edit_A2 = generate_code_from_vector(A1_emb + Da, model, tokenizer, device)
            bleu = compute_code_bleu(A2, code_edit_A2,flag_data_flow)
            edit_bleu += bleu
            code_edit_B2 = generate_code_from_vector(B1_emb + Db, model, tokenizer, device)
            bleu = compute_code_bleu(B2, code_edit_B2,flag_data_flow)
            edit_bleu += bleu

            code_cross_A2 = generate_code_from_vector(A1_emb + Db, model, tokenizer, device)
            bleu = compute_code_bleu(A2, code_cross_A2,flag_data_flow)
            bleu = [value for value, label in zip(bleu, labels) if label == 1]
            cross_bleu += bleu
            code_cross_B2 = generate_code_from_vector(B1_emb + Da, model, tokenizer, device)
            bleu = compute_code_bleu(B2, code_cross_B2,flag_data_flow)
            bleu = [value for value, label in zip(bleu, labels) if label == 1]
            cross_bleu += bleu
            '''
            for a1, a2, code_a1, code_a2 in zip(A1, A2, code_A1, code_edit_A2):
                print('-----------------------------------A1------------------------------------------------------------------------------A2-------------------------------------------------------')
                printCodePairSideBySide(a1, a2)
                print('----------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
                print('---------------------------------A1+Da----------------------------------------------------------------------------A1+Db-----------------------------------------------------')

                printCodePairSideBySide(format_java_code(code_a1), format_java_code(code_a2))
                print('----------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
                print('----------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
               '''
    # print('Code Bleu: ' + str(np.mean(code_bleu)))
    print('Edit Bleu: ' + str(np.mean(edit_bleu)))
    print('Cross Bleu: ' + str(np.mean(cross_bleu)))
    return generated_codes

# def generate_code(model, dataloader, tokenizer, device):
#     """
#     Generate code for inputs from a dataloader using encoder and decoder models.

#     Args:
#         decoder_model (torch.nn.Module): The fine-tuned decoder model.
#         cer_model (torch.nn.Module): The encoder model to generate embeddings.
#         dataloader (torch.utils.data.DataLoader): The dataloader for input data.
#         tokenizer (transformers.T5Tokenizer): The tokenizer used with the model.
#         device (torch.device): The device (CPU or GPU) for computation.

#     Returns:
#         list: A list of generated codes.
#     """
#     model.eval()
#     generated_codes = []
#     total_bleu = []
#     code_bleu = []
#     edit_bleu = []
#     with torch.no_grad():
#         for batch in tqdm(dataloader, desc="Generating Code Embeddings", leave=False):
#             # Tokenize inputs
#             A1 = batch['A1']
#             A2 = batch['A2']
#             B1 = batch['B1']
#             B2 = batch['B2']
#             labels = batch['labels']
#             a1, a2, b1, b2 = A1, A2, B1, B2
#             # for a1, a2, b1, b2, label in zip(A1, A2, B1, B2, labels):
#             a1_tokenized = tokenizer(a1, return_tensors="pt", padding=True, truncation=True).to(device)
#             a2_tokenized = tokenizer(a2, return_tensors="pt", padding=True, truncation=True).to(device)
#             b1_tokenized = tokenizer(b1, return_tensors="pt", padding=True, truncation=True).to(device)
#             b2_tokenized = tokenizer(b2, return_tensors="pt", padding=True, truncation=True).to(device)

#             a1_emb = model.get_embeddings_tokenized(a1_tokenized)
#             b1_emb = model.get_embeddings_tokenized(b1_tokenized)
#             # da, db = model.get_edit_encodings([a1, a2, b1, b2])
#             da, db = model.get_edit_encodings()
#             # print(inputs_emb.shape)
#             # Generate code for each embedding
#             code_a1 = generate_code_from_vector(a1_emb, model, tokenizer, device)
#             bleu = compute_code_bleu(a1, code_a1)
#             code_bleu.append(bleu)
#             code_b1 = generate_code_from_vector(b1_emb, model, tokenizer, device)
#             bleu = compute_code_bleu(b1, code_b1)
#             code_bleu.append(bleu)

#             code_edit_a2 = generate_code_from_vector(a1_emb + da, model, tokenizer, device)
#             bleu = compute_code_bleu(a2, code_edit_a2)
#             edit_bleu.append(bleu)

#             code_edit_b2 = generate_code_from_vector(b1_emb + db, model, tokenizer, device)
#             bleu = compute_code_bleu(b2, code_edit_b2)
#             edit_bleu.append(bleu)


#                 # if label == 1:
#                 #     print('-----------------------------------A1------------------------------------------------------------------------------A2-------------------------------------------------------')
#                 #     printCodePairSideBySide(a1, a2)
#                 #     print('----------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
#                 #     print('---------------------------------A1+Da----------------------------------------------------------------------------A1+Db-----------------------------------------------------')

#                 #     printCodePairSideBySide(format_java_code(code_a2), format_java_code(code_b2))
#                 #     print('----------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
#                 #     print('----------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
#                 # else:
#             print('-----------------------------------A1------------------------------------------------------------------------------A2-------------------------------------------------------')
#             printCodePairSideBySide(a1, a2)
#             print('----------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
#             print('---------------------------------A1+Da----------------------------------------------------------------------------A1+Db-----------------------------------------------------')

#             printCodePairSideBySide(format_java_code(code_a2), format_java_code(code_b2))
#             print('----------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
#             print('----------------------------------------------------------------------------------------------------------------------------------------------------------------------------')


#                 # sys.stdout.flush()              # Manually flush the output buffer
#     print('Code Bleu: ' + str(np.mean(code_bleu)))
#     print('Edit Bleu: ' + str(np.mean(edit_bleu)))

#     return generated_codes

# class FinetuneDecoderModel(nn.Module):
#     def __init__(self, encoder_model: nn.Module, decoder_model: T5ForConditionalGeneration, cer_model, tokenizer: T5Tokenizer, configs: dict, device: torch.device):
#         super(FinetuneDecoderModel, self).__init__()
#         self.encoder_model = encoder_model
#         self.decoder_model = decoder_model
#         self.cer_model = cer_model
#         self.tokenizer = tokenizer
#         self.configs = configs
#         self.device = device

#     def forward(self, concatenated_inputs: List[str], target_codes: List[str]) -> torch.Tensor:
#         # Tokenize the concatenated inputs (A1, A2, B1, B2)
#         tokenized_inputs = self.tokenizer(concatenated_inputs, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
#         # Get embeddings from the encoder
#         encoder_embeddings = self.cer_model.get_embeddings(tokenized_inputs).to(self.device)

#         # Reshape encoded vectors to simulate last_hidden_state: [batch_size, seq_length=1, hidden_size]
#         encoder_outputs = encoder_embeddings.unsqueeze(1)  # Add a sequence dimension
        

#         # Tokenize the target codes (decoder input)
#         tokenized_targets = self.tokenizer(target_codes, return_tensors="pt", padding=True, truncation=True).to(self.device)

#         # Pass through the decoder with encoder outputs
#         decoder_outputs = self.decoder_model(
#             encoder_outputs=BaseModelOutput(last_hidden_state=encoder_outputs),
#             labels=tokenized_targets.input_ids  # Using labels for supervised learning (teacher forcing)
#         )

#         # Get the decoder's loss (cross-entropy loss for language generation)
#         loss = decoder_outputs.loss
#         logits = decoder_outputs.logits

#         return loss, logits

# def make_finetuning_dataloader(dataset: pd.DataFrame, collate_fn: callable, tokenizer: T5Tokenizer, configs: dict, n_workers: int = 0, train: bool = True) -> torch.utils.data.DataLoader:
#     shuffle = train and not configs.testing
#     pytorch_dataset = CERDataset(dataset)
#     return torch.utils.data.DataLoader(pytorch_dataset, collate_fn=collate_fn, shuffle=shuffle, batch_size=8, num_workers=n_workers)

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
    # checkpoint_name = '20250130_215832' #cerd, all, reconstruction = 2
    # checkpoint_name = '20250130_220007' #cerd, all, reconstruction = 3
    # checkpoint_name = '20250208_162240' #cerd, all, reconstruction = 4
    # checkpoint_name = '20250208_162301' #cerd, all, reconstruction = 5

    # checkpoint_name = '20250206_190729' #cerd, all, reconstruction = 3, regularization = 2

    # checkpoint_name = '20250211_212450' #cerd, all, reconstruction = 2, codet5-large
    # checkpoint_name = '20250211_212856' #cerd, all, reconstruction = 3, codet5-large
    # checkpoint_name = '20250211_212656' #cerdd, all, reconstruction = 2, codet5-large
    # checkpoint_name = '20250211_213144' #cerdd, all, reconstruction = 3, codet5-large

    # checkpoint_name = '20250215_160225' #cerd, all, reconstruction = 2, codet5-base, rec and trans
    # checkpoint_name = '20250215_160712' #cerdd, all, reconstruction = 2, codet5-base, rec and trans

    # checkpoint_name = '20250216_021903' #cerd, all, recstruction = 1, contrastive = 1, regularization = 0
    # checkpoint_name = '20250216_022008' #cerd, all, recstruction = 0, contrastive = 1, regularization = 1
    # checkpoint_name = '20250216_022136' #cerd, all, recstruction = 1, contrastive = 0, regularization = 1

    print("Baseline CERD Model with codeT5-base")
    cerd_model = BaselineCERDModel(configs=configs, device=device).to(device)
    generated_code = generate_code_in_batch(model= cerd_model, dataset=test_set, tokenizer=tokenizer, configs=configs, device=device)

    checkpoint_name = '20250130_215832' #cerd, all, reconstruction = 2, base
    print("checkpoint_name = '20250130_215832' #cerd, all, reconstruction = 2, base")
    cerd_model, _, _, _ = load_checkpoint_model_and_data(checkpoint_name=checkpoint_name, configs=configs)
    generated_code = generate_code_in_batch(model= cerd_model, dataset=test_set, tokenizer=tokenizer, configs=configs, device=device)

    checkpoint_name = '20250211_212450' #cerd, all, reconstruction = 2, codet5-large
    print("checkpoint_name = '20250211_212450' #cerd, all, reconstruction = 2, codet5-large")
    cerd_model, _, _, _ = load_checkpoint_model_and_data(checkpoint_name=checkpoint_name, configs=configs)
    generated_code = generate_code_in_batch(model= cerd_model, dataset=test_set, tokenizer=tokenizer, configs=configs, device=device)

    checkpoint_name = '20250215_160225' #cerd, all, reconstruction = 2, codet5-base, rec and trans
    print("checkpoint_name = '20250215_160225' #cerd, all, reconstruction = 2, codet5-base, rec and trans")
    cerd_model, _, _, _ = load_checkpoint_model_and_data(checkpoint_name=checkpoint_name, configs=configs)
    generated_code = generate_code_in_batch(model= cerd_model, dataset=test_set, tokenizer=tokenizer, configs=configs, device=device)

    checkpoint_name = '20250216_021903' #cerd, all, recstruction = 1, contrastive = 1, regularization = 0
    print("checkpoint_name = '20250216_021903' #cerd, all, recstruction = 1, contrastive = 1, regularization = 0")
    cerd_model, _, _, _ = load_checkpoint_model_and_data(checkpoint_name=checkpoint_name, configs=configs)
    generated_code = generate_code_in_batch(model= cerd_model, dataset=test_set, tokenizer=tokenizer, configs=configs, device=device)

    checkpoint_name = '20250216_022008' #cerd, all, recstruction = 0, contrastive = 1, regularization = 1
    print("checkpoint_name = '20250216_022008' #cerd, all, recstruction = 0, contrastive = 1, regularization = 1")
    cerd_model, _, _, _ = load_checkpoint_model_and_data(checkpoint_name=checkpoint_name, configs=configs)
    generated_code = generate_code_in_batch(model= cerd_model, dataset=test_set, tokenizer=tokenizer, configs=configs, device=device)

    checkpoint_name = '20250216_022136' #cerd, all, recstruction = 1, contrastive = 0, regularization = 1
    print("# checkpoint_name = '20250216_022136' #cerd, all, recstruction = 1, contrastive = 0, regularization = 1")
    cerd_model, _, _, _ = load_checkpoint_model_and_data(checkpoint_name=checkpoint_name, configs=configs)
    generated_code = generate_code_in_batch(model= cerd_model, dataset=test_set, tokenizer=tokenizer, configs=configs, device=device)

if __name__ == "__main__":
    main()
