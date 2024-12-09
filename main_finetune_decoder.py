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

from transformers.modeling_outputs import BaseModelOutput
from tqdm import tqdm

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
    # Reshape encoder_embedding to match decoder input requirements
    encoder_embedding = encoder_embedding.unsqueeze(1)  # [batch_size, seq_length=1, hidden_size]

    # Generate code using the decoder
    with torch.no_grad():
        generated_ids = decoder_model.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=encoder_embedding),
            max_length=1024,
            decoder_start_token_id=tokenizer.pad_token_id  # Start decoding from <pad>
        )
        generated_code = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return generated_code


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
            inputs = batch['inputs']
            for input in inputs:
                inputs_tokenized = tokenizer(input, return_tensors="pt", padding=True, truncation=True).to(device)

                # Get encoder embeddings
                inputs_emb = cer_model.get_embeddings(inputs_tokenized)  # [batch_size, hidden_size]
                # print(inputs_emb.shape)
                # Generate code for each embedding
                code = generate_code_from_vector(inputs_emb, decoder_model, tokenizer, device)
                generated_codes.append(code)
                # for input, code in zip(inputs, codes):
                print(len(input), len(code))
                printCodePairSideBySide(input, format_java_code(code))
                

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
        encoder_embeddings = self.cer_model.get_embeddings_tokenized(tokenized_inputs).to(self.device)

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


class FinetuneCollateForCER(object):
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
        concatenated_inputs = A1 + A2 + B1 + B2

        target_codes = A1 + A2 + B1 + B2  # In this case, the target is the same as the input for finetuning.

        return {
            'inputs': concatenated_inputs,  # A1, A2, B1, B2 concatenated
            'target_codes': target_codes,   # The target code sequences for supervised finetuning
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
    # checkpoint_path = 'checkpoints/20241021_174314' # allowed_problem_list: ['12', '17', '21'] # only if else related problems
    # checkpoint_path = 'checkpoints/20241021_200242' #allowed_problem_list: ['34', '39', '40'] # string problems requiring loops
    # checkpoint_path = 'checkpoints/20241028_201125' # allowed_problem_list: ['46', '71'] # array problems requiring loops
    # checkpoint_path = 'checkpoints/20241029_134451' #all problems, dim 128
    # checkpoint_path = 'checkpoints/20241118_191604' #all problems, dim 768
    # checkpoint_path = 'checkpoints/20241030_163548' #random (epoch 2) all problem
    # checkpoint_path = 'checkpoints/20241031_175148' # epoch 2, with margin .5
    # checkpoint_path = 'checkpoints/20241031_175058' # all problems, with margin .5
    # checkpoint_path = 'checkpoints/20241031_190036' #epoch 8, margin 1

    checkpoint_path = 'checkpoints/20241208_204527' # with regularization, allowed_problem_list: ['12', '17', '21'] # only if else related problems

    cer_model = torch.load(checkpoint_path + '/model')
    encoder_model = T5ForConditionalGeneration.from_pretrained('t5-base')
    encoder_model.load_state_dict(cer_model.pretrained_encoder.state_dict(),strict=False)
    # encoder_model = cer_model.pretrained_encoder

    # Freeze the encoder weights
    for param in encoder_model.encoder.parameters():
        param.requires_grad = False

    # Create a new decoder (from T5)
    decoder_model = T5ForConditionalGeneration.from_pretrained('t5-base')
    # decoder_model = T5ForConditionalGeneration.from_pretrained(configs.model_name)
    decoder_model = decoder_model.to(device)

    train_set = torch.load(checkpoint_path + '/train_set')
    test_set = torch.load(checkpoint_path + '/test_set')
    valid_set = torch.load(checkpoint_path + '/valid_set')

    # Instantiate the finetune model
    finetune_model = FinetuneDecoderModel(encoder_model, decoder_model, cer_model, tokenizer, configs, device)

    # Create a DataLoader for the finetuning task
    # trainset, validset, testset = read_data(configs)
    train_dataloader = make_finetuning_dataloader(train_set, FinetuneCollateForCER(tokenizer, configs, device), tokenizer, configs)
    test_dataloader = make_finetuning_dataloader(test_set, FinetuneCollateForCER(tokenizer, configs, device), tokenizer, configs)

    optimizer = AdamW(decoder_model.parameters(), lr=1e-4)
    # loss_fn = CrossEntropyLoss()

    # Training loop
    decoder_model.train()
    num_epochs = 100

    for epoch in tqdm(range(num_epochs)):
        for batch in tqdm(train_dataloader, desc='Training'):
            concatenated_inputs = batch['inputs']
            target_codes = batch['target_codes']
            
            # Forward pass (finetuning the decoder)
            loss, logits = finetune_model(concatenated_inputs, target_codes)
            
            # Backward pass (you will need to add optimizer step and loss backpropagation)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # print(f"Loss: {loss.item()}")
        torch.save(decoder_model, 'checkpoints/decoder_models/decoder_model_all_768_reg')
    
    # Example usage
    generated_code = generate_code(decoder_model=decoder_model, cer_model= cer_model, dataloader=test_dataloader, tokenizer=tokenizer, device=device)

if __name__ == "__main__":
    main()
