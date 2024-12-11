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

from tqdm import tqdm
from evaluator.CodeBLEU import calc_code_bleu
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
    # Reshape encoder_embedding to match decoder input requirements
    encoder_embedding = encoder_embedding.unsqueeze(1)  # [batch_size, seq_length=1, hidden_size]

    # Generate code using the decoder
    with torch.no_grad():
        generated_ids = model.pretrained_decoder.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=encoder_embedding),
            max_length=128,
            decoder_start_token_id=tokenizer.pad_token_id  # Start decoding from <pad>
        )
        generated_code = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return generated_code


# def compute_code_bleu(ground_truth_codes, generated_codes):
#     params='0.25,0.25,0.25,0.25'
#     lang='java'
#     codebleu_score, detailed_codebleu_score = calc_code_bleu.get_codebleu(
#                         pre_references=[ground_truth_codes], hypothesis=generated_codes, lang=lang, params=params)
#     return codebleu_score, detailed_codebleu_score

# Function to generate codes for a batch of inputs
def generate_code(model, dataloader, tokenizer, device):
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
    model.eval()
    generated_codes = []
    total_bleu = 0
    total = 0
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

                a1_emb = model.get_embeddings_tokenized(a1_tokenized)
                b1_emb = model.get_embeddings_tokenized(b1_tokenized)
                da, db = model.get_edit_encodings([a1, a2, b1, b2])
                # print(inputs_emb.shape)
                # Generate code for each embedding
                code_a2 = generate_code_from_vector(a1_emb + da, model, tokenizer, device)
                code_b2 = generate_code_from_vector(a1_emb + db, model, tokenizer, device)

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

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.utils.data import DataLoader
import torch.nn.functional as F
from datatypes import *

class FinetuneDecoderModel(nn.Module):
    def __init__(self, encoder_model: nn.Module, decoder_model: T5ForConditionalGeneration, cer_model, tokenizer: T5Tokenizer, configs: dict, device: torch.device):
        super(FinetuneDecoderModel, self).__init__()
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
        self.cer_model = cer_model
        self.tokenizer = tokenizer
        self.configs = configs
        self.device = device

    def forward(self, concatenated_inputs: List[str], target_codes: List[str]) -> torch.Tensor:
        # Tokenize the concatenated inputs (A1, A2, B1, B2)
        tokenized_inputs = self.tokenizer(concatenated_inputs, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        # Get embeddings from the encoder
        encoder_embeddings = self.cer_model.get_embeddings(tokenized_inputs).to(self.device)

        # Reshape encoded vectors to simulate last_hidden_state: [batch_size, seq_length=1, hidden_size]
        encoder_outputs = encoder_embeddings.unsqueeze(1)  # Add a sequence dimension
        

        # Tokenize the target codes (decoder input)
        tokenized_targets = self.tokenizer(target_codes, return_tensors="pt", padding=True, truncation=True).to(self.device)

        # Pass through the decoder with encoder outputs
        decoder_outputs = self.decoder_model(
            encoder_outputs=BaseModelOutput(last_hidden_state=encoder_outputs),
            labels=tokenized_targets.input_ids  # Using labels for supervised learning (teacher forcing)
        )

        # Get the decoder's loss (cross-entropy loss for language generation)
        loss = decoder_outputs.loss
        logits = decoder_outputs.logits

        return loss, logits

def make_finetuning_dataloader(dataset: pd.DataFrame, collate_fn: callable, tokenizer: T5Tokenizer, configs: dict, n_workers: int = 0, train: bool = True) -> torch.utils.data.DataLoader:
    shuffle = train and not configs.testing
    pytorch_dataset = CERDataset(dataset)
    return torch.utils.data.DataLoader(pytorch_dataset, collate_fn=collate_fn, shuffle=shuffle, batch_size=4, num_workers=n_workers)


# class DecoderCollateForEdit(object):
#     def __init__(self, tokenizer: T5Tokenizer, configs: dict, device: torch.device):
#         self.tokenizer = tokenizer
#         self.configs = configs
#         self.device = device

#     def __call__(self, batch: List[Dict[str, Union[str, int]]]) -> Dict[str, torch.Tensor]:
#         # Create a single list where each A1, A2, B1, and B2 will be concatenated consecutively
#         concatenated_inputs = []
#         target_codes = []

#         # Build input and target code sequences
#         A1 = [item['A1'] for item in batch]
#         A2 = [item['A2'] for item in batch]
#         B1 = [item['B1'] for item in batch]
#         B2 = [item['B2'] for item in batch]
#         # concatenated_inputs = A1 + B1

#         # target_codes = A2 + B2  # In this case, the target is the same as the input for finetuning.

#         return {
#             'A1': A1, 
#             'A2': A2,
#             'B1': B1,
#             'B2': B2,
#         }



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

    # Path to the checkpoint
    # checkpoint_path = 'checkpoints/20241126_214208' # allowed_problem_list: ['12', '17', '21'] # only if else related problems; epoch 100
    # checkpoint_path = 'checkpoints/20241127_140131' # allowed_problem_list: ['12', '17', '21'] # only if else related problems; epoch 200
    # checkpoint_path = 'checkpoints/20241127_161838'  
    # checkpoint_path = 'checkpoints/20241209_165650' # with regularization, if else  
    checkpoint_path = 'checkpoints/20241209_194800' # with regularization, if else, exclusive problems between train and test

    cerd_model = torch.load(checkpoint_path + '/model')

    train_set = torch.load(checkpoint_path + '/train_set')
    test_set = torch.load(checkpoint_path + '/test_set')
    valid_set = torch.load(checkpoint_path + '/valid_set')

    # Instantiate the finetune model
    # finetune_model = FinetuneDecoderModel(encoder_model, decoder_model, cer_model, tokenizer, configs, device)

    # Create a DataLoader for the finetuning task
    # trainset, validset, testset = read_data(configs)
    # train_dataloader = make_finetuning_dataloader(train_set, DecoderCollateForEdit(tokenizer, configs, device), tokenizer, configs)
    test_dataloader = make_finetuning_dataloader(test_set, DecoderCollateForEdit(tokenizer, configs, device), tokenizer, configs)

    
    # Example usage
    generated_code = generate_code(model= cerd_model, dataloader=test_dataloader, tokenizer=tokenizer, device=device)

if __name__ == "__main__":
    main()
