import sys
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5Model
from torch.nn import functional as F
import pandas as pd
import matplotlib.pyplot as plt
from typing import List

# Load Tokenizer and Encoder
tokenizer = T5Tokenizer.from_pretrained('t5-base')
t5_encoder = T5Model.from_pretrained('t5-base').encoder

# Function to dynamically plot the loss and accuracy
def plot_metrics(train_losses, accuracies):
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label='Train Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Over Time')
    plt.legend()

    # plt.show()
    plt.savefig('plot.png')

# Dataset class
class ContrastiveDataset(Dataset):
    def __init__(self, df):
        self.data = df
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        code_i_1 = row['code_i_1']
        code_j_1 = row['code_j_1']
        code_i_2 = row['code_i_2']
        code_j_2 = row['code_j_2']
        is_similar = row['is_similar']
        
        # Tokenize code_i_1 and code_j_1 separately
        input1_i = tokenizer(code_i_1, return_tensors="pt", padding=True, truncation=True)
        input1_j = tokenizer(code_j_1, return_tensors="pt", padding=True, truncation=True)
        
        # Tokenize code_i_2 and code_j_2 separately
        input2_i = tokenizer(code_i_2, return_tensors="pt", padding=True, truncation=True)
        input2_j = tokenizer(code_j_2, return_tensors="pt", padding=True, truncation=True)
        
        return (input1_i, input1_j), (input2_i, input2_j), torch.tensor(is_similar, dtype=torch.float)

# Modified collate function
def collate_fn(batch):
    input1_batch = [item[0] for item in batch]
    input2_batch = [item[1] for item in batch]
    labels_batch = torch.stack([item[2] for item in batch])

    # Tokenize and pad input1 pairs (i_1, j_1)
    input1_i_ids = [item[0]['input_ids'].squeeze() for item in input1_batch]
    input1_i_attention = [item[0]['attention_mask'].squeeze() for item in input1_batch]
    input1_j_ids = [item[1]['input_ids'].squeeze() for item in input1_batch]
    input1_j_attention = [item[1]['attention_mask'].squeeze() for item in input1_batch]

    # Tokenize and pad input2 pairs (i_2, j_2)
    input2_i_ids = [item[0]['input_ids'].squeeze() for item in input2_batch]
    input2_i_attention = [item[0]['attention_mask'].squeeze() for item in input2_batch]
    input2_j_ids = [item[1]['input_ids'].squeeze() for item in input2_batch]
    input2_j_attention = [item[1]['attention_mask'].squeeze() for item in input2_batch]

    # Padding the sequences
    input1_i_ids_padded = pad_sequence(input1_i_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    input1_i_attention_padded = pad_sequence(input1_i_attention, batch_first=True, padding_value=0)
    input1_j_ids_padded = pad_sequence(input1_j_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    input1_j_attention_padded = pad_sequence(input1_j_attention, batch_first=True, padding_value=0)

    input2_i_ids_padded = pad_sequence(input2_i_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    input2_i_attention_padded = pad_sequence(input2_i_attention, batch_first=True, padding_value=0)
    input2_j_ids_padded = pad_sequence(input2_j_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    input2_j_attention_padded = pad_sequence(input2_j_attention, batch_first=True, padding_value=0)

    # Return the padded inputs and labels
    return {
        'input1_i': {'input_ids': input1_i_ids_padded, 'attention_mask': input1_i_attention_padded},
        'input1_j': {'input_ids': input1_j_ids_padded, 'attention_mask': input1_j_attention_padded},
        'input2_i': {'input_ids': input2_i_ids_padded, 'attention_mask': input2_i_attention_padded},
        'input2_j': {'input_ids': input2_j_ids_padded, 'attention_mask': input2_j_attention_padded}
    }, labels_batch

# CustomCERModel
class CustomCERModel(nn.Module):
    def __init__(self, configs: dict, device: torch.device):
        super(CustomCERModel, self).__init__()

        # Initialize tokenizer and encoder based on the model name
        self.configs = configs
        if configs['model_name'] in ['t5-base', 't5-large']:
            self.tokenizer = T5Tokenizer.from_pretrained(configs['model_name'])
            self.pretrained_encoder = T5Model.from_pretrained(configs['model_name']).encoder
            self.embedding_size = self.pretrained_encoder.config.d_model
        else:
            raise NotImplementedError("Only t5-base and t5-large are supported in this code.")

        # Single fully connected layer for edit encoding (merged for both A and B)
        self.fc_edit_encoder = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.Tanh(),
            nn.Linear(self.embedding_size, self.configs['code_change_vector_size']),
            nn.ReLU()
        )

        self.device = device

    def get_embeddings(self, tokenized_inputs) -> torch.Tensor:
        # Pass inputs through the encoder
        tokenized_inputs = {k: v.to(self.device) for k, v in tokenized_inputs.items()}
        outputs = self.pretrained_encoder(**tokenized_inputs).last_hidden_state
        seq_lens = tokenized_inputs['attention_mask'].sum(dim=1)
        masked_hidden_states = outputs * tokenized_inputs['attention_mask'].unsqueeze(2)
        embeddings = masked_hidden_states.sum(dim=1) / seq_lens.unsqueeze(1)
        return embeddings

    # def forward(self, concatenated_inputs: List[str]) -> torch.Tensor:
    #     # Tokenize concatenated inputs
    #     tokenized_inputs = self.tokenizer(concatenated_inputs, return_tensors='pt', padding=True, truncation=True).to(self.device)
    #     # tokenized_inputs = {k: v.to(self.device) for k, v in tokenized_inputs.items()}

    #     # # Get embeddings for all concatenated inputs (A1, A2, B1, B2 in sequence)
    #     # embeddings = self.get_embeddings(tokenized_inputs)
    #     output = self.pretrained_encoder(tokenized_inputs['input_ids'],attention_mask=tokenized_inputs['attention_mask'])[0]
    #     embeddings = output[:,0,:]

    #     # Split embeddings into A1, A2, B1, and B2
    #     batch_size = embeddings.shape[0] // 4  # Since A1, A2, B1, B2 are concatenated
    #     A1_emb = embeddings[:batch_size]
    #     A2_emb = embeddings[batch_size:2 * batch_size]
    #     B1_emb = embeddings[2 * batch_size:3 * batch_size]
    #     B2_emb = embeddings[3 * batch_size:]

    #     # Compute Da and Db by subtracting corresponding embeddings
    #     # Da = A1_emb - A2_emb
    #     # Db = B1_emb - B2_emb
    #     Da = (A1_emb + A2_emb) / 2
    #     Db = (B1_emb + B2_emb) / 2
    def forward(self, input1, input2):
        # Encode each input separately
        output1_i = self.pretrained_encoder(input1['input1_i']['input_ids'], attention_mask=input1['input1_i']['attention_mask'])[0]
        output1_j = self.pretrained_encoder(input1['input1_j']['input_ids'], attention_mask=input1['input1_j']['attention_mask'])[0]
        output2_i = self.pretrained_encoder(input2['input2_i']['input_ids'], attention_mask=input2['input2_i']['attention_mask'])[0]
        output2_j = self.pretrained_encoder(input2['input2_j']['input_ids'], attention_mask=input2['input2_j']['attention_mask'])[0]
        
        # Get CLS token embeddings (first token in T5)
        cls_output1_i = output1_i[:, 0, :]  # CLS token for code_i_1
        cls_output1_j = output1_j[:, 0, :]  # CLS token for code_j_1
        cls_output2_i = output2_i[:, 0, :]  # CLS token for code_i_2
        cls_output2_j = output2_j[:, 0, :]  # CLS token for code_j_2
        
        # Aggregate the embeddings (here we use average, but you can use other techniques like concatenation)
        Da = (cls_output1_i + cls_output1_j) / 2
        Db = (cls_output2_i + cls_output2_j) / 2
        batch_size = Da.shape[0]
        # Batch the edit encodings together (Da and Db concatenated)
        all_edit_encodings = torch.cat((Da, Db), dim=0)

        # Pass the concatenated edit encodings through the single edit encoder
        all_edit_fc = self.fc_edit_encoder(all_edit_encodings)

        # Split the results back into Da_fc and Db_fc
        Da_fc = all_edit_fc[:batch_size]
        Db_fc = all_edit_fc[batch_size:]

        # Handle different loss functions
        if self.configs['loss_fn'] in ['ContrastiveLoss', 'NTXentLoss','CosineSimilarityLoss', 'MultipleNegativesRankingLoss']:
            return Da_fc, Db_fc
        else:
            raise NotImplementedError("Loss function not implemented.")

# Contrastive Loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss = torch.mean((label) * torch.pow(euclidean_distance, 2) +
                          (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss

# Define a function to compute Euclidean distance
def compute_distance(output1, output2):
    return F.pairwise_distance(output1, output2)

# Accuracy calculation
def calculate_accuracy(model, dataloader, device, threshold=0.5):
    model.eval()  # Set the model to evaluation mode
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():  # Disable gradient computation for evaluation
        for batch in dataloader:
            inputs, labels = batch

            # Move inputs and labels to the device
            input1_i = {key: val.squeeze().to(device) for key, val in inputs['input1_i'].items()}
            input1_j = {key: val.squeeze().to(device) for key, val in inputs['input1_j'].items()}
            input2_i = {key: val.squeeze().to(device) for key, val in inputs['input2_i'].items()}
            input2_j = {key: val.squeeze().to(device) for key, val in inputs['input2_j'].items()}
            labels = labels.to(device)

            # Forward pass: Get projections from the model
            proj1, proj2 = model(
                {
                    'input1_i': input1_i,
                    'input1_j': input1_j,
                },
                {
                    'input2_i': input2_i,
                    'input2_j': input2_j,
                }
            )

            # Compute Euclidean distance between embeddings
            distances = compute_distance(proj1, proj2)

            # Predict: if the distance is less than the threshold, predict "similar"
            predictions = (distances < threshold).float()

            # Compare predictions with labels
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

    # Calculate accuracy
    accuracy = correct_predictions / total_predictions
    return accuracy

# Training Loop
def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    
    for batch in dataloader:
        inputs, labels = batch

        # Move inputs and labels to the device (GPU/CPU)
        input1_i = {key: val.squeeze().to(device) for key, val in inputs['input1_i'].items()}
        input1_j = {key: val.squeeze().to(device) for key, val in inputs['input1_j'].items()}
        input2_i = {key: val.squeeze().to(device) for key, val in inputs['input2_i'].items()}
        input2_j = {key: val.squeeze().to(device) for key, val in inputs['input2_j'].items()}
        labels = labels.to(device)

        optimizer.zero_grad()  # Reset gradients

        # Forward pass: Get projections from the model
        proj1, proj2 = model(
            {
                'input1_i': input1_i,
                'input1_j': input1_j,
            },
            {
                'input2_i': input2_i,
                'input2_j': input2_j,
            }
        )

        # Compute contrastive loss
        loss = criterion(proj1, proj2, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)

# Main function
def main():
    # Load the dataset
    df = pd.read_pickle('data/test_dataset.pkl')
    dataset = ContrastiveDataset(df)
    # DataLoader with custom collate_fn
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    # Initialize model, criterion, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    configs = {
        'model_name': 't5-base',
        'code_change_vector_size': 128,
        'loss_fn': 'ContrastiveLoss'
    }

    model = CustomCERModel(configs=configs, device=device).to(device)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Train the model
    num_epochs = 100
    train_losses = []
    accuracies = []
    for epoch in range(num_epochs):
        epoch_loss = train_model(model, dataloader, optimizer, criterion, device)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
        train_losses.append(epoch_loss)

        # Calculate accuracy on the training dataset
        train_accuracy = calculate_accuracy(model, dataloader, device, threshold=1.0)
        print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
        accuracies.append(train_accuracy)
        sys.stdout.flush()  # Manually flush the buffer

    plot_metrics(train_losses, accuracies)
if __name__ == "__main__":
    main()
