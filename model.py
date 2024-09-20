import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
# from transformers import AutoTokenizer, AutoModelWithLMHead
from transformers import T5Tokenizer, T5Model, RobertaTokenizer

from pdb import set_trace
from datatypes import *

def create_tokenizer(configs: dict) -> tokenizer:
    if configs.model_name == 't5-base' or configs.model_name == 't5-large':
        tokenizer = T5Tokenizer.from_pretrained(configs.model_name)
    else :
        tokenizer = RobertaTokenizer.from_pretrained(configs.model_name)
    
    return tokenizer

def create_cer_model(configs: dict, device: torch.device) -> nn.Module:
    tokenizer = create_tokenizer(configs)
    return CustomCERModel(configs,device).to(device), tokenizer

class CustomCERModel(nn.Module):
    def __init__(self, configs: dict, device: torch.device):
        super(CustomCERModel, self).__init__()

        # Initialize tokenizer and encoder based on the model name
        if configs.model_name in ['t5-base', 't5-large']:
            self.tokenizer = T5Tokenizer.from_pretrained(configs.model_name)
        else:
            self.tokenizer = RobertaTokenizer.from_pretrained(configs.model_name)

        self.pretrained_encoder = T5Model.from_pretrained(configs.model_name).encoder

        self.embedding_size = self.pretrained_encoder.config.d_model
        self.configs = configs

        # Single fully connected layer for edit encoding (merged for both A and B)
        self.fc_edit_encoder = nn.Sequential(
            nn.Linear(2 * self.embedding_size, self.configs.code_change_vector_size),
            nn.Sigmoid()
        )

        self.device = device

    def get_embeddings(self, tokenized_inputs) -> torch.Tensor:
        # Pass inputs through the encoder
        outputs = self.pretrained_encoder(**tokenized_inputs).last_hidden_state
        # Calculate mean of the embeddings
        embeddings = torch.mean(outputs, dim=1)
        return embeddings

    def forward(self, concatenated_inputs: List[str]) -> torch.Tensor:
        # inputs = self.tokenizer(code, return_tensors="pt", padding=True, truncation=True).to(self.device) # need to remove tokenization from here, because it's inefficient

        # Tokenize the concatenated inputs in one go
        tokenized_inputs = self.tokenizer(
            concatenated_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        # Get embeddings for all concatenated inputs (A1, A2, B1, B2 in sequence)
        embeddings = self.get_embeddings(tokenized_inputs)

        # Split embeddings into A1, A2, B1, and B2
        batch_size = embeddings.shape[0] // 4  # Since A1, A2, B1, B2 are concatenated
        A1_emb = embeddings[:batch_size]
        A2_emb = embeddings[batch_size:2 * batch_size]
        B1_emb = embeddings[2 * batch_size:3 * batch_size]
        B2_emb = embeddings[3 * batch_size:]

        # Compute Da and Db by concatenating corresponding embeddings
        Da = torch.cat((A1_emb, A2_emb), dim=1)
        Db = torch.cat((B1_emb, B2_emb), dim=1)

        # Batch the edit encodings together (Da and Db concatenated)
        all_edit_encodings = torch.cat((Da, Db), dim=0)

        # Pass the concatenated edit encodings through the single edit encoder
        all_edit_fc = self.fc_edit_encoder(all_edit_encodings)

        # Split the results back into Da_fc and Db_fc
        Da_fc = all_edit_fc[:batch_size]
        Db_fc = all_edit_fc[batch_size:]

        # Handle different loss functions
        if self.configs.loss_fn in ['ContrastiveLoss', 'NTXentLoss']:
            return Da_fc, Db_fc

# class CustomCERModel(nn.Module):
#     def __init__(self, configs: dict, device: torch.device):
#         super(CustomCERModel, self).__init__()
#         if configs.model_name == 't5-base' or configs.model_name == 't5-large':
#             self.tokenizer = T5Tokenizer.from_pretrained(configs.model_name)
#         else :
#             self.tokenizer = RobertaTokenizer.from_pretrained(configs.model_name)

#         # Initialize separate encoders for each input
#         self.encoder_A1 = T5Model.from_pretrained(configs.model_name)
#         self.encoder_A2 = T5Model.from_pretrained(configs.model_name)
#         self.encoder_B1 = T5Model.from_pretrained(configs.model_name)
#         self.encoder_B2 = T5Model.from_pretrained(configs.model_name)
        
#         self.embedding_size = self.encoder_A1.config.d_model
#         self.configfile = configs

#         # Fully connected layer for encoding the edits between two corresponding code snippets
#         code_change_vector_size = 128
#         self.fc_edit_encoder_A = nn.Sequential(
#             nn.Linear(2*self.embedding_size, code_change_vector_size),
#             nn.Sigmoid()
#         )

#         self.fc_edit_encoder_B = nn.Sequential(
#             nn.Linear(2*self.embedding_size, code_change_vector_size),
#             nn.Sigmoid()
#         )

#         # Fully connected layer to take two edit encodings and output a single value. This is only needed for the BCEWithLogitsLoss
#         self.fc_classifier = nn.Linear(2 * self.embedding_size, 1)

#         self.device = device

#     def get_embeddings(self, code: str, encoder) -> torch.Tensor:
#         inputs = self.tokenizer(code, return_tensors="pt", padding=True, truncation=True).to(self.device) # need to remove tokenization from here, because it's inefficient
#         outputs = encoder.encoder(**inputs)
#         embeddings = outputs.last_hidden_state
#         embeddings = torch.mean(embeddings, dim=1)
        
#         return embeddings

#     def forward(self, A1: str, A2: str, B1: str, B2: str) -> torch.Tensor:
#         A1_emb = self.get_embeddings(A1, self.encoder_A1).to(self.device)
#         A2_emb = self.get_embeddings(A2, self.encoder_A2).to(self.device)
#         B1_emb = self.get_embeddings(B1, self.encoder_B1).to(self.device)
#         B2_emb = self.get_embeddings(B2, self.encoder_B2).to(self.device)

#         # print('AB cosine similarity: ' + str(F.cosine_similarity(A1_emb-A2_emb, B1_emb-B2_emb, dim=1)))
#         # print('A Norm: ' + str(torch.norm(A1_emb-A2_emb, dim=1)))
#         # print('B Norm: ' + str(torch.norm(B1_emb-B2_emb, dim=1)))
#         # print('A cosine similarity: ' + str(F.cosine_similarity(A1_emb, A2_emb, dim=1)))
#         # print('B cosine similarity: ' + str(F.cosine_similarity(B1_emb, B2_emb, dim=1)))

#         # Compute differences
#         # Da = A2_emb - A1_emb
#         # Db = B2_emb - B1_emb
#         Da = torch.cat((A1_emb, A2_emb), dim = 1)
#         Db = torch.cat((B1_emb, B2_emb), dim = 1)

#         # Pass through the first FC layer
#         Da_fc = self.fc_edit_encoder_A(Da)
#         Db_fc = self.fc_edit_encoder_B(Db)

#         if self.configfile.loss_fn == 'ContrastiveLoss':
#             return Da_fc, Db_fc
#         elif self.configfile.loss_fn == 'NTXentLoss' :
#             return Da_fc, Db_fc
#         elif self.configfile.loss_fn == 'BCEWithLogitsLoss':
#             # Concatenate Da and Db
#             combined = torch.cat((Da_fc, Db_fc), dim=1)

#             # Pass through the second FC layer
#             output = torch.sigmoid(self.fc_classifier(combined))

#             return output

# class CustomCERModel(nn.Module):
#     def __init__(self, configs: dict, device: torch.device):
#         super(CustomCERModel, self).__init__()
#         if configs.model_name == 't5-base' or configs.model_name == 't5-large':
#             self.tokenizer = T5Tokenizer.from_pretrained(configs.model_name)
#         else :
#             self.tokenizer = RobertaTokenizer.from_pretrained(configs.model_name)
#         self.pretrained_encoder = T5Model.from_pretrained(configs.model_name)
#         self.embedding_size = self.pretrained_encoder.config.d_model
#         self.configfile = configs

#         # Fully connected layer for encoding the edits between two corresponding code snippets
#         self.fc_edit_encoder = nn.Sequential()
#         self.fc_edit_encoder.add_module('fc_edit_encoder_linear', nn.Linear(self.embedding_size, self.embedding_size))
#         self.fc_edit_encoder.add_module('fc_edit_encoder_activation', nn.Sigmoid())

#         # Fully connected layer to take to two edit encodings and output a single value. This is only needed for the BCEWithLogitsLoss
#         self.fc_classifier = nn.Linear(2 * self.embedding_size, 1)

#         self.device = device

#         if configs.verbose == True:
#             # print(self)
#             pass

#     def get_embeddings(self, code: str) -> torch.Tensor:
#             inputs = self.tokenizer(code, return_tensors="pt", padding=True, truncation=True).to(self.device)
#             outputs = self.pretrained_encoder.encoder(**inputs)
#             embeddings = outputs.last_hidden_state
#             # Average pooling
#             embeddings = torch.mean(embeddings, dim=1)
#             if self.configfile.verbose == True:
#                 # print(len(code), embeddings.shape)
#                 # print(embeddings)
#                 pass
#             return embeddings

#     def forward(self, A1: str, A2: str, B1: str, B2: str) -> torch.Tensor:
#         A1_emb = self.get_embeddings(A1).to(self.device)
#         A2_emb = self.get_embeddings(A2).to(self.device)
#         B1_emb = self.get_embeddings(B1).to(self.device)
#         B2_emb = self.get_embeddings(B2).to(self.device)

#         # Compute differences
#         Da = A2_emb - A1_emb
#         Db = B2_emb - B1_emb

#         # Pass through the first FC layer
#         Da_fc = self.fc_edit_encoder(Da)
#         if self.configfile.verbose == True:
#             # print(Da_fc.shape)
#             # print(Da_fc)
#             pass
#         Db_fc = self.fc_edit_encoder(Db)
#         if self.configfile.verbose == True:
#             # print(Db_fc.shape)
#             # print(Db_fc)
#             pass


#         if self.configfile.loss_fn == 'ContrastiveLoss':
#             return (Da_fc, Db_fc)
#         elif self.configfile.loss_fn == 'BCEWithLogitsLoss':
#             # Concatenate Da and Db
#             combined = torch.cat((Da_fc, Db_fc), dim=1)

#             # Pass through the second FC layer
#             output = torch.sigmoid(self.fc_classifier(combined))

#             return output
