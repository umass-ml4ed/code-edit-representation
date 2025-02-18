import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
# from transformers import AutoTokenizer, AutoModelWithLMHead
from transformers import T5Tokenizer, T5Model, RobertaTokenizer, T5ForConditionalGeneration
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput

from pdb import set_trace
from datatypes import *
from utils import *
from trainer import *
from tqdm import tqdm
import itertools

def create_tokenizer(configs: dict) -> tokenizer:
    if configs.model_name == 't5-base' or configs.model_name == 't5-large':
        tokenizer = T5Tokenizer.from_pretrained(configs.model_name)
    elif configs.model_name == 'google/flan-t5-base' or configs.model_name == 'google/flan-t5-large':
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    else :
        tokenizer = RobertaTokenizer.from_pretrained(configs.model_name)
    
    return tokenizer

def create_cer_model(configs: dict, device: torch.device) -> nn.Module:
    tokenizer = create_tokenizer(configs)
    if configs.exp_name == 'cer':
        model = ExtendedCERModel(configs,device).to(device)
    elif configs.exp_name == 'cerd':
        model = ExtendedCERDModel(configs,device).to(device)
    elif configs.exp_name == 'cerdd':
        model = ExtendedCERDDModel(configs,device).to(device)
    return model, tokenizer

class BaseCERModel(nn.Module):
    def __init__(self, configs: dict, device: torch.device):
        super(BaseCERModel, self).__init__()

        self.tokenizer = create_tokenizer(configs)
        self.configs = configs
        self.device = device

    def batch_unpack(self, inputs, batch_size):
        A1 = inputs[:batch_size]
        A2 = inputs[batch_size:2 * batch_size]
        B1 = inputs[2 * batch_size:3 * batch_size]
        B2 = inputs[3 * batch_size:]
        return A1, A2, B1, B2
    
    def get_embeddings_tokenized(self, tokenized_inputs) -> torch.Tensor:
        # Pass inputs through the encoder
        outputs = self.pretrained_encoder(**tokenized_inputs).last_hidden_state
        seq_lens = tokenized_inputs.attention_mask.sum(dim=1)
        masked_hidden_states = outputs * tokenized_inputs.attention_mask.unsqueeze(2)
        embeddings = masked_hidden_states.sum(dim=1) / seq_lens.unsqueeze(1)
        return embeddings

    def get_embeddings(self, text_inputs) -> torch.tensor:
        tokenized_inputs = self.tokenizer(text_inputs, return_tensors="pt",padding=True,truncation=True).to(self.device)
        return self.get_embeddings_tokenized(tokenized_inputs)
    
    def get_edit_encodings_tokenized(self, tokenized_inputs):
        """Get edit encodings for tokenized inputs."""
        embeddings = self.get_embeddings_tokenized(tokenized_inputs)
        batch_size = embeddings.shape[0] // 4
        A1_emb, A2_emb, B1_emb, B2_emb = self.batch_unpack(embeddings, batch_size)
        Da = A2_emb - A1_emb
        Db = B2_emb - B1_emb
        all_edit_encodings = torch.cat((Da, Db), dim=0)
        all_edit_fc = self.fc_edit_encoder(all_edit_encodings)
        # Split back into Da_fc and Db_fc
        Da_fc = all_edit_fc[:batch_size]
        Db_fc = all_edit_fc[batch_size:]
        return Da_fc, Db_fc

    def get_edit_encodings(self, text_inputs):
        tokenized_inputs = self.tokenizer(text_inputs, return_tensors="pt",padding=True,truncation=True).to(self.device)
        return self.get_edit_encodings_tokenized(tokenized_inputs)
    
    def get_latent_states(self, dataloader):
        self.eval()
        res = np.empty((0,self.configs.model_inp_dim))
        problemIDs = []
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(dataloader, desc="Generating Latent States", leave=False)):
                inputs = batch['inputs']
                pID = batch['problemIDs']
                print(pID)
                Da, Db = self.get_edit_encodings(inputs)
                Da = Da.cpu().detach().numpy()
                Db = Db.cpu().detach().numpy()
                res = np.concatenate((res, Da), axis=0)
                res = np.concatenate((res, Db), axis=0)
                problemIDs = problemIDs + pID
                problemIDs = problemIDs + pID
        return res, problemIDs


class BaselineCERModel(BaseCERModel):
    def __init__(self, configs: dict, device: torch.device):
        super(BaselineCERModel, self).__init__(configs, device)

        self.pretrained_encoder = T5Model.from_pretrained(configs.model_name).encoder
        self.embedding_size = self.pretrained_encoder.config.d_model

class BaselineCERDModel(BaseCERModel):
    def __init__(self, configs: dict, device: torch.device):
        super(BaselineCERDModel, self).__init__(configs, device)

        self.pretrained_encoder = T5Model.from_pretrained(configs.model_name).encoder
        self.embedding_size = self.pretrained_encoder.config.d_model

        self.pretrained_decoder = T5ForConditionalGeneration.from_pretrained(configs.model_name)#.decoder

    def get_edit_encodings_tokenized(self, tokenized_inputs):
        embeddings = self.get_embeddings_tokenized(tokenized_inputs)
        batch_size = embeddings.shape[0] // 4
        A1_emb, A2_emb, B1_emb, B2_emb = self.batch_unpack(embeddings, batch_size)
        Da = A2_emb - A1_emb
        Db = B2_emb - B1_emb
        return Da, Db

class ExtendedCERModel(BaseCERModel):
    def __init__(self, configs: dict, device: torch.device):
        super(ExtendedCERModel, self).__init__(configs, device)

        self.pretrained_encoder = T5Model.from_pretrained(configs.model_name).encoder

        self.embedding_size = self.pretrained_encoder.config.d_model

        # Single fully connected layer for edit encoding (merged for both A and B)
        self.fc_edit_encoder = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            # nn.ReLU(),
            nn.Tanh(),
            nn.Linear(self.embedding_size, self.configs.code_change_vector_size),
        )

    def forward(self, concatenated_inputs: List[str], is_similar: torch.tensor) -> torch.Tensor:
        # Tokenize the concatenated inputs in one go
        tokenized_inputs = self.tokenizer(concatenated_inputs, return_tensors="pt",padding=True,truncation=True).to(self.device)

        Da_fc, Db_fc = self.get_edit_encodings_tokenized(tokenized_inputs=tokenized_inputs)
        contrastiveObjective = getContrastiveLossObjective(self.configs, self.device)

        embeddings = self.get_embeddings_tokenized(tokenized_inputs)
        batch_size = embeddings.shape[0] // 4
        A1_emb, A2_emb, B1_emb, B2_emb = self.batch_unpack(embeddings, batch_size)
        regularizationObjective = getRegularizationLossObjective(self.configs, self.device)

        total_loss = contrastiveObjective((Da_fc, Db_fc), is_similar) * self.configs.lambda_contrastive + (regularizationObjective(A1_emb + Da_fc, A2_emb) + regularizationObjective(B1_emb + Db_fc, B2_emb)/2) * self.configs.lambda_regularization

        return total_loss

class ExtendedCERDModel(BaseCERModel):
    def __init__(self, configs: dict, device: torch.device):
        super(ExtendedCERDModel, self).__init__(configs, device)

        self.pretrained_encoder = T5ForConditionalGeneration.from_pretrained(configs.model_name).encoder
        self.embedding_size = self.pretrained_encoder.config.d_model

        # Fully connected layer for encoding deltas (Da, Db)
        self.fc_edit_encoder = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.Tanh(),
            nn.Linear(self.embedding_size, self.embedding_size),
        )

        self.pretrained_decoder = T5ForConditionalGeneration.from_pretrained(configs.model_name)#.decoder

        # Loss functions and weights
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, concatenated_inputs, is_similar):
        """
        Forward pass for both contrastive and reconstruction objectives.
        """
        tokenized_inputs = self.tokenizer(concatenated_inputs, return_tensors="pt", padding=True, truncation=True).to(self.device)
        # Get edit encodings for concatenated inputs
        Da_fc, Db_fc = self.get_edit_encodings_tokenized(tokenized_inputs)

        # Compute contrastive loss
        contrastiveObjective = getContrastiveLossObjective(self.configs, self.device)
        contrastive_loss = contrastiveObjective((Da_fc, Db_fc), is_similar)
        
        embeddings = self.get_embeddings_tokenized(tokenized_inputs)
        batch_size = embeddings.shape[0] // 4
        A1, A2, B1, B2 = self.batch_unpack(concatenated_inputs, batch_size)
        A1_emb, A2_emb, B1_emb, B2_emb = self.batch_unpack(embeddings, batch_size)

        if self.configs.reconstruction_edit_flag:
            decoder_inputs = torch.cat((A1_emb, A2_emb, B1_emb, B2_emb, A1_emb + Da_fc, B1_emb + Db_fc), dim = 0).unsqueeze(1)
            decoder_targets = self.tokenizer(A1 + A2 + B1 + B2 + A2 + B2, return_tensors="pt", padding=True, truncation=True).to(self.device)
        else:
            decoder_inputs = torch.cat((A1_emb, A2_emb, B1_emb, B2_emb), dim=0).unsqueeze(1)
            decoder_targets = self.tokenizer(A1 + A2 + B1 + B2, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        # Decoder reconstruction for A2 and B2
        decoder_outputs = self.pretrained_decoder(
            encoder_outputs=BaseModelOutput(last_hidden_state=decoder_inputs),
            labels=decoder_targets.input_ids
        ).logits
   
        reconstruction_loss = self.cross_entropy_loss(decoder_outputs.view(-1, decoder_outputs.size(-1)), decoder_targets.input_ids.view(-1))
   
        regularizationObjective = getRegularizationLossObjective(self.configs, self.device)
        regularization_loss = (regularizationObjective(A1_emb + Da_fc, A2_emb) + regularizationObjective(B1_emb + Db_fc, B2_emb)/2)

        # Total loss
        total_loss = (
            self.configs.lambda_contrastive * contrastive_loss
            + self.configs.lambda_reconstruction * reconstruction_loss
            + self.configs.lambda_regularization * regularization_loss
        )

        return total_loss

class ExtendedCERDDModel(BaseCERModel):
    def __init__(self, configs: dict, device: torch.device):
        super(ExtendedCERDDModel, self).__init__(configs, device)

        self.pretrained_encoder = T5ForConditionalGeneration.from_pretrained(configs.model_name).encoder
        self.embedding_size = self.pretrained_encoder.config.d_model

        # Fully connected layer for encoding deltas (Da, Db)
        self.fc_edit_encoder = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.Tanh(),
            nn.Linear(self.embedding_size, self.embedding_size),
        )

        self.fc_dense_edit_encoder = nn.Sequential(
            nn.Linear(self.embedding_size, self.configs.code_change_vector_size)
        )

        self.pretrained_decoder = T5ForConditionalGeneration.from_pretrained(configs.model_name)#.decoder

        # Loss functions and weights
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, concatenated_inputs, is_similar):
        """
        Forward pass for both contrastive and reconstruction objectives.
        """
        tokenized_inputs = self.tokenizer(concatenated_inputs, return_tensors="pt", padding=True, truncation=True).to(self.device)
        # Get edit encodings for concatenated inputs
        Da_fc, Db_fc = self.get_edit_encodings_tokenized(tokenized_inputs)

        Da_fc_dense = self.fc_dense_edit_encoder(Da_fc)
        Db_fc_dense = self.fc_dense_edit_encoder(Db_fc)

        # Compute contrastive loss
        contrastiveObjective = getContrastiveLossObjective(self.configs, self.device)
        contrastive_loss = contrastiveObjective((Da_fc_dense, Db_fc_dense), is_similar)
        
        embeddings = self.get_embeddings_tokenized(tokenized_inputs)
        batch_size = embeddings.shape[0] // 4
        A1, A2, B1, B2 = self.batch_unpack(concatenated_inputs, batch_size)
        A1_emb, A2_emb, B1_emb, B2_emb = self.batch_unpack(embeddings, batch_size)

        decoder_inputs = torch.cat((A1_emb, A2_emb, B1_emb, B2_emb, A1_emb + Da_fc, B1_emb + Db_fc), dim = 0).unsqueeze(1)
        # decoder_inputs = torch.cat((A2_emb, B2_emb), dim=0).unsqueeze(1)
        decoder_targets = self.tokenizer(A1 + A2 + B1 + B2 + A2 + B2, return_tensors="pt", padding=True, truncation=True).to(self.device)
        # Decoder reconstruction for A2 and B2
        decoder_outputs = self.pretrained_decoder(
            encoder_outputs=BaseModelOutput(last_hidden_state=decoder_inputs),
            labels=decoder_targets.input_ids
        ).logits
   
        reconstruction_loss = self.cross_entropy_loss(decoder_outputs.view(-1, decoder_outputs.size(-1)), decoder_targets.input_ids.view(-1))
   
        regularizationObjective = getRegularizationLossObjective(self.configs, self.device)
        regularization_loss = (regularizationObjective(A1_emb + Da_fc, A2_emb) + regularizationObjective(B1_emb + Db_fc, B2_emb)/2)

        # Total loss
        total_loss = (
            self.configs.lambda_contrastive * contrastive_loss
            + self.configs.lambda_reconstruction * reconstruction_loss
            + self.configs.lambda_regularization * regularization_loss
        )

        return total_loss

class CustomCERModel(nn.Module):
    def __init__(self, configs: dict, device: torch.device):
        super(CustomCERModel, self).__init__()

        self.tokenizer = create_tokenizer(configs)

        self.pretrained_encoder = T5Model.from_pretrained(configs.model_name).encoder

        self.embedding_size = self.pretrained_encoder.config.d_model
        self.configs = configs

        # Single fully connected layer for edit encoding (merged for both A and B)
        self.fc_edit_encoder = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            # nn.ReLU(),
            nn.Tanh(),
            nn.Linear(self.embedding_size, self.configs.code_change_vector_size),
        )

        self.device = device

    def get_embeddings(self, tokenized_inputs) -> torch.Tensor:
        # Pass inputs through the encoder
        outputs = self.pretrained_encoder(**tokenized_inputs).last_hidden_state
        seq_lens = tokenized_inputs.attention_mask.sum(dim=1)
        masked_hidden_states = outputs * tokenized_inputs.attention_mask.unsqueeze(2)
        embeddings = masked_hidden_states.sum(dim=1) / seq_lens.unsqueeze(1)
        # Calculate mean of the embeddings
        # embeddings = torch.mean(outputs, dim=1)
        return embeddings

    def forward(self, concatenated_inputs: List[str]) -> torch.Tensor:
        # inputs = self.tokenizer(code, return_tensors="pt", padding=True, truncation=True).to(self.device) # need to remove tokenization from here, because it's inefficient

        # Tokenize the concatenated inputs in one go
        tokenized_inputs = self.tokenizer( concatenated_inputs, return_tensors="pt",padding=True,truncation=True).to(self.device)

        # Get embeddings for all concatenated inputs (A1, A2, B1, B2 in sequence)
        embeddings = self.get_embeddings(tokenized_inputs)

        # Split embeddings into A1, A2, B1, and B2
        batch_size = embeddings.shape[0] // 4  # Since A1, A2, B1, B2 are concatenated
        A1_emb = embeddings[:batch_size]
        A2_emb = embeddings[batch_size:2 * batch_size]
        B1_emb = embeddings[2 * batch_size:3 * batch_size]
        B2_emb = embeddings[3 * batch_size:]

        # Compute Da and Db by concatenating corresponding embeddings
        # Da = torch.cat((A1_emb, A2_emb), dim=1)
        # Db = torch.cat((B1_emb, B2_emb), dim=1)
        Da = A1_emb - A2_emb
        Db = B1_emb - B2_emb

        # Batch the edit encodings together (Da and Db concatenated)
        all_edit_encodings = torch.cat((Da, Db), dim=0)

        # Pass the concatenated edit encodings through the single edit encoder
        all_edit_fc = self.fc_edit_encoder(all_edit_encodings)

        # Split the results back into Da_fc and Db_fcamb
        Da_fc = all_edit_fc[:batch_size]
        Db_fc = all_edit_fc[batch_size:]

        # if self.configs.verbose == True:
        #     print("Outputs")
        #     print(Da_fc, Db_fc)
        # Handle different loss functions
        if self.configs.loss_fn in ['ContrastiveLoss', 'NTXentLoss','CosineSimilarityLoss', 'MultipleNegativesRankingLoss']:
            return Da_fc, Db_fc

class CustomCERDModel(nn.Module):
    def __init__(self, configs: dict, device: torch.device):
        super(CustomCERDModel, self).__init__()

        self.tokenizer = create_tokenizer(configs)    
        # self.pretrained_encoder = AutoModelForSeq2SeqLM.from_pretrained(configs.model_name).encoder
        # self.pretrained_decoder = AutoModelForSeq2SeqLM.from_pretrained(configs.model_name).decoder

        self.pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(configs.model_name)
        self.pretrained_encoder = self.pretrained_model.encoder
        self.pretrained_decoder = self.pretrained_model.decoder  # This stays as T5Stack
        self.lm_head = self.pretrained_model.lm_head  # Language modeling head for logits projection


        self.embedding_size = self.pretrained_encoder.config.d_model
        self.configs = configs

        # Fully connected layer for encoding deltas (Da, Db)
        self.fc_edit_encoder = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.Tanh(),
            nn.Linear(self.embedding_size, self.configs.code_change_vector_size),
        )

        # Loss functions and weights
        self.contrastive_margin = configs.margin
        self.lambda_contrastive = configs.lambda_contrastive    
        self.lambda_reconstruction = configs.lambda_reconstruction  
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.device = device



    def compute_contrastive_loss(self, d_a, d_b, is_similar):
        """Compute contrastive loss between delta embeddings."""
        distance = torch.norm(d_a - d_b, dim=1)  # Euclidean distance
        loss = is_similar * distance**2 + (1 - is_similar) * torch.clamp(self.contrastive_margin - distance, min=0)**2
        return loss.mean()

    def get_embeddings(self, tokenized_inputs):
        """Get sequence-level embeddings."""
        outputs = self.pretrained_encoder(**tokenized_inputs).last_hidden_state
        seq_lens = tokenized_inputs.attention_mask.sum(dim=1)
        masked_hidden_states = outputs * tokenized_inputs.attention_mask.unsqueeze(2)
        embeddings = masked_hidden_states.sum(dim=1) / seq_lens.unsqueeze(1)
        return embeddings

    def get_edit_encodings(self, concatenated_inputs):
        """Get edit encodings for concatenated inputs."""
        tokenized_inputs = self.tokenizer(
            concatenated_inputs, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        embeddings = self.get_embeddings(tokenized_inputs)
        batch_size = embeddings.shape[0] // 4
        A1_emb, A2_emb, B1_emb, B2_emb = self.batch_unpack(embeddings, batch_size)
        Da = A2_emb - A1_emb
        Db = B2_emb - B1_emb
        all_edit_encodings = torch.cat((Da, Db), dim=0)
        all_edit_fc = self.fc_edit_encoder(all_edit_encodings)
        # Split back into Da_fc and Db_fc
        Da_fc = all_edit_fc[:batch_size]
        Db_fc = all_edit_fc[batch_size:]

        return Da_fc, Db_fc
    
    def forward(self, concatenated_inputs, is_similar):
        """
        Forward pass for both contrastive and reconstruction objectives.
        """
        # Get edit encodings for concatenated inputs
        Da_fc, Db_fc = self.get_edit_encodings(concatenated_inputs)

        # Compute contrastive loss
        contrastive_loss = self.compute_contrastive_loss(Da_fc, Db_fc, is_similar)
        
        tokenized_inputs = self.tokenizer(concatenated_inputs, return_tensors="pt", padding=True, truncation=True).to(self.device)
        embeddings = self.get_embeddings(tokenized_inputs)
        batch_size = embeddings.shape[0] // 4
        A1, A2, B1, B2 = self.batch_unpack(concatenated_inputs, batch_size)
        A1_emb, A2_emb, B1_emb, B2_emb = self.batch_unpack(embeddings, batch_size)
        # decoder_inputs = torch.cat((A1_emb + Da_fc, B1_emb + Db_fc), dim = 0).unsqueeze(1)
        decoder_inputs = torch.cat((A2_emb, B2_emb), dim=0).unsqueeze(1)
        decoder_targets = self.tokenizer(A2 + B2, return_tensors="pt", padding=True, truncation=True).to(self.device)
        # Decoder reconstruction for A2 and B2
        # decoder_outputs = self.pretrained_decoder(
        #     # encoder_outputs=BaseModelOutput(last_hidden_state=decoder_inputs),
        #     # labels=decoder_targets.input_ids
        #     encoder_hidden_states=decoder_inputs,  # Pass the encoder outputs here
        #     decoder_start_token_id=self.tokenizer.pad_token_id  # Start decoding from <pad>

        # ).logits
        # Decoder reconstruction for A2 and B2
        # decoder_outputs = self.pretrained_decoder(
        #     input_ids=decoder_targets.input_ids,  # Target input IDs for the decoder
        #     attention_mask=decoder_targets.attention_mask,
        #     encoder_hidden_states=decoder_inputs,  # Pass the encoder outputs here
        # ).logits

        # Decoder reconstruction for A2 and B2
        decoder_outputs = self.pretrained_decoder(
            input_ids=decoder_targets.input_ids,  # Target input IDs for the decoder
            attention_mask=decoder_targets.attention_mask,
            encoder_hidden_states=decoder_inputs,  # Pass the encoder outputs here
        )

        # Use the lm_head to project decoder hidden states to logits
        decoder_logits = self.lm_head(decoder_outputs.last_hidden_state)

        # decoder_outputs_b2 = self.pretrained_decoder(
        #     input_ids=decoder_inputs_b2.input_ids.to(self.device),
        #     attention_mask=decoder_inputs_b2.attention_mask.to(self.device),
        #     encoder_hidden_states=B1_emb.unsqueeze(1),  # Use B1 embeddings for B2 decoding
        # ).logits

        # # Compute reconstruction loss
        # reconstruction_loss_a2 = self.cross_entropy_loss(
        #     decoder_outputs_a2.view(-1, decoder_outputs_a2.size(-1)),
        #     decoder_inputs_a2.labels.view(-1).to(self.device)
        # )
        # reconstruction_loss_b2 = self.cross_entropy_loss(
        #     decoder_outputs_b2.view(-1, decoder_outputs_b2.size(-1)),
        #     decoder_inputs_b2.labels.view(-1).to(self.device)
        # )
        # reconstruction_loss = (reconstruction_loss_a2 + reconstruction_loss_b2) / 2

        # reconstruction_loss = self.cross_entropy_loss(decoder_outputs.view(-1, decoder_outputs.size(-1)), decoder_targets.view(-1, decoder_targets.size(-1)))
        reconstruction_loss = self.cross_entropy_loss(decoder_logits.view(-1, decoder_logits.size(-1)),decoder_targets.input_ids.view(-1).to(self.device),)


        # Total loss
        total_loss = (
            self.lambda_contrastive * contrastive_loss
            + self.lambda_reconstruction * reconstruction_loss
        )

        return total_loss#, contrastive_loss, reconstruction_loss
