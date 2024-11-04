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

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the model and tokenizer
model_name = "t5-base"  # Use the name of your fine-tuned model
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Function to generate code from a given vector embedding
def generate_code_from_vector(encoder_embedding, device):
    # Ensure model and tensors are on the correct device
    model.to(device)
    model.eval()
    encoder_embedding = encoder_embedding.to(device)
    
    # Generate dummy input to set up encoder-decoder attention
    input_ids = tokenizer("<s>", return_tensors="pt").input_ids.to(device)

    # Forward pass through the encoder to get base encoder outputs
    with torch.no_grad():
        encoder_outputs = model.encoder(input_ids=input_ids)

    # Replace encoder hidden states with your embedding
    encoder_outputs.last_hidden_state = encoder_embedding

    # Generate output from the decoder using the modified encoder output
    generated_ids = model.generate(
        encoder_outputs=encoder_outputs,
        max_length=100,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True
    )

    # Decode the generated tokens to get the code
    generated_code = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_code

def generate_code(model, dataloader, device):
    model.to(device)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating Code Embeddings", leave=False):
            inputs = batch['inputs']
            labels = batch['labels']
            masks = batch['masks']
            batch_size = labels.shape[0]
            A1 = inputs[:batch_size]
            A2 = inputs[batch_size:2 * batch_size]
            B1 = inputs[2 * batch_size:3 * batch_size]
            B2 = inputs[3 * batch_size:]
            A1_mask = masks[:batch_size]
            A2_mask = masks[batch_size:2 * batch_size]
            B1_mask = masks[2 * batch_size:3 * batch_size]
            B2_mask = masks[3 * batch_size:]

            A1 = tokenizer(A1, return_tensors="pt", padding=True, truncation=True).to(device)
            A1_emb = model.get_embeddings(A1).unsqueeze(0).to(device)
            print(A1_emb.shape)

            return generate_code_from_vector(A1_emb, device)
    return embeddings

@hydra.main(version_base=None, config_path=".", config_name="configs_cer")
def main(configs):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Make reproducible
    set_random_seed(configs.seed)

    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Current Device: ' + str(device))

    # Initialize the model
    model0, tokenizer = create_cer_model(configs, device)

    # Path to the checkpoint
    checkpoint_path = 'checkpoints/20241029_134451'
    model = torch.load(checkpoint_path + '/model', map_location=device)

    train_set = torch.load(checkpoint_path + '/train_set')
    test_set = torch.load(checkpoint_path + '/test_set')
    valid_set = torch.load(checkpoint_path + '/valid_set')
    
    ## load data
    collate_fn = CollateForCER(tokenizer=tokenizer, configs=configs, device=device)
    train_loader = make_dataloader(train_set, collate_fn=collate_fn, configs=configs)
    valid_loader = make_dataloader(valid_set, collate_fn=collate_fn, configs=configs)
    test_loader  = make_dataloader(test_set, collate_fn=collate_fn, configs=configs)

    data_loader = test_loader
    
    # Example usage
    generated_code = generate_code(model, data_loader, device)
    print("Generated code:", generated_code)

if __name__ == "__main__":
    main()
