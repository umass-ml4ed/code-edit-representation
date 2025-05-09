import os
from datetime import datetime
import torch.optim as optim
import transformers
import neptune
from neptune.utils import stringify_unsupported
import hydra
from omegaconf import OmegaConf
import sys

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
from main_finetune_decoder import *

# Function to generate code from a given vector embedding
def generate_code_from_vector(encoder_embedding, decoder_model, tokenizer, device):
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
    # # Reshape encoder_embedding to match decoder input requirements
    # encoder_embedding = encoder_embedding.unsqueeze(1)  # [batch_size, seq_length=1, hidden_size]

    # # Generate code using the decoder
    # with torch.no_grad():
    #     generated_ids = decoder_model.generate(
    #         encoder_outputs=BaseModelOutput(last_hidden_state=encoder_embedding),
    #         max_length=128,
    #         decoder_start_token_id=tokenizer.pad_token_id  # Start decoding from <pad>
    #     )
    #     generated_code = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # return generated_code
    encoder_embedding = encoder_embedding.unsqueeze(1)  # [batch_size, seq_length=1, hidden_size]
    encoder_outputs = BaseModelOutput(last_hidden_state=encoder_embedding)

    # Generate code using the model's generate method
    with torch.no_grad():
        generated_ids = decoder_model.generate(
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
def generate_code_in_batch(decoder_model, cer_model, dataset, tokenizer, configs, device, isBaseline=False):
    collate_fn = CollateForCER(tokenizer=tokenizer, configs=configs, device=device)
    dataloader  = make_dataloader_experiment(dataset , collate_fn=collate_fn, configs=configs)

    decoder_model.eval()
    generated_codes = []
    code_bleu = []
    edit_bleu = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating Code Embeddings", leave=False):
            concatenated_inputs = batch['inputs']
            labels = batch['labels']
            labels = labels.to(device).to(torch.float32)

            tokenized_inputs = tokenizer(concatenated_inputs, return_tensors="pt", padding=True, truncation=True).to(device)
            if isBaseline == False:
                Da, Db = cer_model.get_edit_encodings_tokenized(tokenized_inputs)

            embeddings = cer_model.get_embeddings_tokenized(tokenized_inputs)
            batch_size = embeddings.shape[0] // 4
            A1, A2, B1, B2 = cer_model.batch_unpack(concatenated_inputs, batch_size)
            A1_emb, A2_emb, B1_emb, B2_emb = cer_model.batch_unpack(embeddings, batch_size)
            
            code_A1 = generate_code_from_vector(A1_emb, decoder_model, tokenizer, device)
            bleu = compute_code_bleu(A1, code_A1)
            code_bleu += bleu
            code_B1 = generate_code_from_vector(B1_emb, decoder_model, tokenizer, device)
            bleu = compute_code_bleu(B1, code_B1)
            code_bleu += bleu

            if isBaseline == False:
                code_edit_A2 = generate_code_from_vector(A1_emb + Da, decoder_model, tokenizer, device)
                bleu = compute_code_bleu(A2, code_edit_A2)
                edit_bleu += bleu
                code_edit_B2 = generate_code_from_vector(B1_emb + Db, decoder_model, tokenizer, device)
                bleu = compute_code_bleu(B2, code_edit_B2)
                edit_bleu += bleu
    # print(code_bleu)
    print('Code Bleu: ' + str(np.mean(code_bleu)))
    if isBaseline == False: print('Edit Bleu: ' + str(np.mean(edit_bleu)))
    return generated_codes

# Function to generate codes for a batch of inputs
def generate_code(decoder_model, cer_model, dataloader, tokenizer, device):
    """
    Generate code for inputs from a dataloader using encoder and decoder models.

    Args:
        decoder_model (torch.nn.Module): The fine-tuned decoder model.
        cer_model (torch.nn.Module): The encoder model to generate embeddings.
        dataloader (torch.utils.data.DataLoader): The dataloader for input data.
        tokenizer (transformers.T5Tokenizer): The tokenizer used with the model.
        device (torch.device): The device (CPU or GPU) for computation.

    Returns:
        list: A list of generated codes.
    """
    decoder_model.eval()
    generated_codes = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating Code Embeddings", leave=False):
            # Tokenize inputs
            A1 = batch['A1']
            A2 = batch['A2']
            B1 = batch['B1']
            B2 = batch['B2']
            labels = batch['labels']
            for a1, a2, b1, b2, label in zip(A1, A2, B1, B2, labels):
                a1_tokenized = tokenizer(a1, return_tensors="pt", padding=True, truncation=True).to(device)
                # a2_tokenized = tokenizer(a2, return_tensors="pt", padding=True, truncation=True).to(device)
                b1_tokenized = tokenizer(b1, return_tensors="pt", padding=True, truncation=True).to(device)
                # b2_tokenized = tokenizer(b2, return_tensors="pt", padding=True, truncation=True).to(device)

                a1_emb = cer_model.get_embeddings_tokenized(a1_tokenized)
                b1_emb = cer_model.get_embeddings_tokenized(b1_tokenized)
                da, db = cer_model.get_edit_encodings([a1, a2, b1, b2])
                # print(inputs_emb.shape)
                # Generate code for each embedding
                code_a2 = generate_code_from_vector(a1_emb + da, decoder_model, tokenizer, device)
                code_b2 = generate_code_from_vector(a1_emb + db, decoder_model, tokenizer, device)

                if label == 1:
                    print('-----------------------------------A1------------------------------------------------------------------------------A2-------------------------------------------------------')
                    printCodePairSideBySide(a1, a2)
                    print('----------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
                    print('---------------------------------A1+Da----------------------------------------------------------------------------A1+Db-----------------------------------------------------')

                    printCodePairSideBySide(format_java_code(code_a2), format_java_code(code_b2))
                    print('----------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
                    print('----------------------------------------------------------------------------------------------------------------------------------------------------------------------------')

                # code_b2 = generate_code_from_vector(b1_emb + db, decoder_model, tokenizer, device)
                # printCodePairSideBySide(b1, format_java_code(code_b2))
                # print('------------------------------------------------------------------------------------')
                # printCodePairSideBySide(b2, format_java_code(code_b2))
                # print('------------------------------------------------------------------------------------')
                # print('------------------------------------------------------------------------------------')
                # print('------------------------------------------------------------------------------------')


                sys.stdout.flush()              # Manually flush the output buffer
                

    return generated_codes


class DecoderCollateForEdit(object):
    def __init__(self, tokenizer: T5Tokenizer, configs: dict, device: torch.device):
        self.tokenizer = tokenizer
        self.configs = configs
        self.device = device

    def __call__(self, batch: List[Dict[str, Union[str, int]]]) -> Dict[str, torch.Tensor]:
        # Create a single list where each A1, A2, B1, and B2 will be concatenated consecutively
        concatenated_inputs = []
        target_codes = []

        # Build input and target code sequences
        A1 = [item['A1'] for item in batch]
        A2 = [item['A2'] for item in batch]
        B1 = [item['B1'] for item in batch]
        B2 = [item['B2'] for item in batch]
        # concatenated_inputs = A1 + B1

        # target_codes = A2 + B2  # In this case, the target is the same as the input for finetuning.
        labels = [item['label'] for item in batch]
        return {
            'A1': A1, 
            'A2': A2,
            'B1': B1,
            'B2': B2,
            'labels': labels,
        }



@hydra.main(version_base=None, config_path=".", config_name="configs_cer")
def main(configs):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Make reproducible
    set_random_seed(configs.seed)

    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Current Device: ' + str(device))

    # Initialize the model
    encoder_model0, tokenizer = create_cer_model(configs, device)

    # Path to the checkpoint
    cer_checkpoint_path = configs.model_save_dir
    # cer_checkpoint_path = '/20241118_191604' #all problems, dim 768
    # cer_checkpoint_path += '/20241030_163548' #random (epoch 2) all problem
    # cer_checkpoint_path += '/20241031_190036' #epoch 8, margin 1
    # cer_checkpoint_path += '/20241208_204527' # with regularization, allowed_problem_list: ['12', '17', '21'] # only if else related problems
    cer_checkpoint_path += '/20241208_214644' # with regularization, all problems


    # cer_model = torch.load(cer_checkpoint_path + '/model')
    cer_model = BaselineCERModel(configs, device).to(device) # Baseline

    # Create a new decoder (from T5)
    finetuned_decoder = torch.load(configs.model_save_dir + '/decoder_models/decoder_model_all_768_code_t5_base_baseline')
    decoder_model = T5ForConditionalGeneration.from_pretrained(configs.model_name)
    decoder_model.load_state_dict(finetuned_decoder.state_dict(), strict=False)
    decoder_model = decoder_model.to(device)

    # train_set = torch.load(cer_checkpoint_path + '/train_set')
    # test_set = torch.load(cer_checkpoint_path + '/test_set')
    # valid_set = torch.load(cer_checkpoint_path + '/valid_set')
    test_set = torch.load(cer_checkpoint_path + '/test_set' ) # different problems than what observed during training
    # Instantiate the finetune model
    # finetune_model = FinetuneDecoderModel(encoder_model, decoder_model, cer_model, tokenizer, configs, device)

    # Create a DataLoader for the finetuning task
    # trainset, validset, testset = read_data(configs)
    # train_dataloader = make_finetuning_dataloader(train_set, DecoderCollateForEdit(tokenizer, configs, device), tokenizer, configs)
    # test_dataloader = make_finetuning_dataloader(test_set, DecoderCollateForEdit(tokenizer, configs, device), tokenizer, configs)

    # Example usage
    generated_code = generate_code_in_batch(decoder_model=decoder_model, cer_model= cer_model, dataset=test_set, tokenizer=tokenizer, configs=configs, device=device, isBaseline=True)

if __name__ == "__main__":
    main()
