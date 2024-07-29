import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
# from transformers import AutoTokenizer, AutoModelWithLMHead
from transformers import T5Tokenizer, T5Model, RobertaTokenizer

from pdb import set_trace
from datatypes import *

# def create_lstm_model(configs, device):
#     lstm = nn.LSTM(configs.lstm_inp_dim, configs.lstm_hid_dim)
#     lstm.to(device)
        
#     return lstm


def create_tokenizer(configs: dict) -> tokenizer:
    if configs.model_name == 't5-base' or configs.model_name == 't5-large':
        tokenizer = T5Tokenizer.from_pretrained(configs.model_name)
    else :
        tokenizer = RobertaTokenizer.from_pretrained(configs.model_name)
    
    return tokenizer


def create_cer_model(configs: dict, device: torch.device) -> nn.Module:
    # ## load the code generator model
    tokenizer = create_tokenizer(configs)
    return CustomCERModel(configs,device).to(device), tokenizer



class CustomCERModel(nn.Module):
    def __init__(self, configs: dict, device: torch.device):
        super(CustomCERModel, self).__init__()
        if configs.model_name == 't5-base' or configs.model_name == 't5-large':
            self.tokenizer = T5Tokenizer.from_pretrained(configs.model_name)
        else :
            self.tokenizer = RobertaTokenizer.from_pretrained(configs.model_name)
        self.model = T5Model.from_pretrained(configs.model_name)
        self.embedding_size = self.model.config.d_model
        self.configfile = configs

        # Fully connected layers
        self.fc1 = nn.Linear(self.embedding_size, self.embedding_size)
        self.fc2 = nn.Linear(2 * self.embedding_size, 1)

        self.device = device

    def get_embeddings(self, code: str) -> torch.Tensor:
            inputs = self.tokenizer(code, return_tensors="pt", padding=True, truncation=True).to(self.device)
            outputs = self.model.encoder(**inputs)
            embeddings = outputs.last_hidden_state
            # Average pooling
            embeddings = torch.mean(embeddings, dim=1)
            return embeddings

    def forward(self, A1: str, A2: str, B1: str, B2: str) -> torch.Tensor:
        A1_emb = self.get_embeddings(A1).to(self.device)
        A2_emb = self.get_embeddings(A2).to(self.device)
        B1_emb = self.get_embeddings(B1).to(self.device)
        B2_emb = self.get_embeddings(B2).to(self.device)

        # Compute differences
        Da = A2_emb - A1_emb
        Db = B2_emb - B1_emb

        # Pass through the first FC layer
        Da_fc = self.fc1(Da)
        Db_fc = self.fc1(Db)


        if self.configfile.loss_fn == 'ContrastiveLoss':
            return (Da_fc, Db_fc)
        elif self.configfile.loss_fn == 'BCEWithLogitsLoss':
            # Concatenate Da and Db
            combined = torch.cat((Da_fc, Db_fc), dim=1)

            # Pass through the second FC layer
            output = torch.sigmoid(self.fc2(combined))

            return output
