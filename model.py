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
    
    # if configs.okt_model == 'student':
    #     model = AutoModelWithLMHead.from_pretrained("model/gpt_code_v1_student")
    # elif configs.okt_model == 'funcom':
    #     model = AutoModelWithLMHead.from_pretrained("model/gpt_code_v1")
    # else:
    #     model = AutoModelWithLMHead.from_pretrained('gpt2')
    # model.to(device)
    
    # linear = nn.Linear(configs.lstm_hid_dim + configs.h_bar_static_dim + configs.dim_normal + (configs.dim_categorical * configs.num_classes_categorical), 768).to(device)
    
    # # Create LSTM to compute knowledge states of students over time
    # lstm = None
    # if configs.use_lstm:
    #     lstm = create_lstm_model(configs, device)
    
    # # Create Q model
    # q_model = None
    # # Create per student h hat distribution parameters
    # student_params_h_bar_static = None
    # student_params_h_hat_mu = None
    # student_params_h_hat_sigma = None
    # student_params_h_hat_discrete = None
    # student_params_h_hat_discrete_copy = None

    # if( configs.use_h_bar_static ):
    #     student_params_h_bar_static = torch.nn.Parameter(torch.empty((len(students), configs.h_bar_static_dim)).to(device), requires_grad=True)
    #     nn.init.normal_(student_params_h_bar_static.data, mean=0.0, std=0.1)

    # if configs.use_q_model:
    #     q_model = QModel(configs).to(device)

    #     if( configs.dim_normal > 0 ):
    #         student_params_h_hat_mu = torch.nn.Parameter(torch.empty((len(students), configs.dim_normal)).to(device), requires_grad=True)
    #         if( configs.learn_sigma ):
    #             student_params_h_hat_sigma = torch.nn.Parameter(torch.empty((len(students), configs.dim_normal)).to(device), requires_grad=True)
    #         else:
    #             student_params_h_hat_sigma = torch.nn.Parameter(torch.empty((len(students), configs.dim_normal)).to(device), requires_grad=False)
    #         nn.init.normal_(student_params_h_hat_mu.data, mean=0.0, std=0.1)
    #         # Initialize log sigma as zeros (i.e., sigma as ones since exp(0) = 1)
    #         nn.init.zeros_(student_params_h_hat_sigma.data)

    #     if( configs.dim_categorical > 0 ):
    #         student_params_h_hat_discrete = torch.nn.Parameter(torch.empty((len(students), configs.dim_categorical, configs.num_classes_categorical)).to(device), requires_grad=True)
    #         # Initialize logits (categorical defines logits as log probabilities) to parameterize close to uniform categorical distribution
    #         nn.init.normal_(student_params_h_hat_discrete.data, mean=torch.log(torch.tensor(1.0 / configs.num_classes_categorical)), std=0.1)
    #         student_params_h_hat_discrete_copy = student_params_h_hat_discrete.data.clone()

    # return lstm, tokenizer, model, linear, q_model, student_params_h_bar_static, student_params_h_hat_mu, student_params_h_hat_sigma, student_params_h_hat_discrete, student_params_h_hat_discrete_copy
    return CustomCERModel(configs,device).to(device),tokenizer



class CustomCERModel(nn.Module):
    def __init__(self, configs: dict, device: torch.device):
        super(CustomCERModel, self).__init__()
        if configs.model_name == 't5-base' or configs.model_name == 't5-large':
            self.tokenizer = T5Tokenizer.from_pretrained(configs.model_name)
        else :
            self.tokenizer = RobertaTokenizer.from_pretrained(configs.model_name)
        self.model = T5Model.from_pretrained(configs.model_name)
        self.embedding_size = self.model.config.d_model

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

        # Concatenate Da and Db
        combined = torch.cat((Da_fc, Db_fc), dim=1)

        # Pass through the second FC layer
        output = torch.sigmoid(self.fc2(combined))

        return output
