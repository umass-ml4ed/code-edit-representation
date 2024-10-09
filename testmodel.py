import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5EncoderModel
from torch.nn import functional as F
import pandas as pd
# from model import CustomCERModel
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
import matplotlib.pyplot as plt
from IPython.display import clear_output


# Load Tokenizer and Encoder
tokenizer = T5Tokenizer.from_pretrained('t5-base')
t5_encoder = T5EncoderModel.from_pretrained('t5-base')


# Function to dynamically plot the loss and accuracy
def plot_metrics(train_losses, accuracies):
    # clear_output(wait=True)
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
    plt.plot(accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy Over Time')
    plt.legend()

    plt.show()
    plt.pause(0.001)

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
        
        # Tokenize the inputs
        input1 = tokenizer(code_i_1 + " " + code_j_1, return_tensors="pt", padding=True, truncation=True)
        input2 = tokenizer(code_i_2 + " " + code_j_2, return_tensors="pt", padding=True, truncation=True)
        
        return input1, input2, torch.tensor(is_similar, dtype=torch.float)

# Custom collate function to pad inputs
def collate_fn(batch):
    input1_batch = [item[0] for item in batch]
    input2_batch = [item[1] for item in batch]
    labels_batch = torch.stack([item[2] for item in batch])

    # Padding input1 and input2
    input1_ids = [item['input_ids'].squeeze() for item in input1_batch]
    input1_attention = [item['attention_mask'].squeeze() for item in input1_batch]

    input2_ids = [item['input_ids'].squeeze() for item in input2_batch]
    input2_attention = [item['attention_mask'].squeeze() for item in input2_batch]

    # Pad sequences to the maximum length within the batch
    input1_ids_padded = pad_sequence(input1_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    input1_attention_padded = pad_sequence(input1_attention, batch_first=True, padding_value=0)

    input2_ids_padded = pad_sequence(input2_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    input2_attention_padded = pad_sequence(input2_attention, batch_first=True, padding_value=0)

    input1_padded = {
        'input_ids': input1_ids_padded,
        'attention_mask': input1_attention_padded
    }
    
    input2_padded = {
        'input_ids': input2_ids_padded,
        'attention_mask': input2_attention_padded
    }

    return input1_padded, input2_padded, labels_batch



# Neural Network Model
class ContrastiveModel(nn.Module):
    def __init__(self, encoder):
        super(ContrastiveModel, self).__init__()
        self.encoder = encoder
        self.fc = nn.Linear(768, 128)  # Project embeddings to a smaller space
    
    def forward(self, input1, input2):
        # Pass inputs through encoder
        output1 = self.encoder(input1['input_ids'], attention_mask=input1['attention_mask'])[0]
        output2 = self.encoder(input2['input_ids'], attention_mask=input2['attention_mask'])[0]
        
        # Get CLS token embeddings
        cls_output1 = output1[:, 0, :]  # First token (CLS token in T5)
        cls_output2 = output2[:, 0, :]  # First token (CLS token in T5)
        
        # Pass through a fully connected layer
        proj1 = F.relu(self.fc(cls_output1))
        proj2 = F.relu(self.fc(cls_output2))
        
        return proj1, proj2
        
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

# Training Loop
def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    
    for batch in dataloader:
        input1, input2, labels = batch
        input1 = {key: val.squeeze().to(device) for key, val in input1.items()}
        input2 = {key: val.squeeze().to(device) for key, val in input2.items()}
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        proj1, proj2 = model(input1, input2)
        
        # Compute contrastive loss
        loss = criterion(proj1, proj2, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(dataloader)
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
            input1, input2, labels = batch
            input1 = {key: val.squeeze().to(device) for key, val in input1.items()}
            input2 = {key: val.squeeze().to(device) for key, val in input2.items()}
            labels = labels.to(device)

            # Forward pass
            proj1, proj2 = model(input1, input2)

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


@hydra.main(version_base=None, config_path=".", config_name="configs_cer")
def main(configs):
    # Load the dataset
    df = pd.read_pickle('data/current_dataset.pkl')
    dataset = ContrastiveDataset(df)
    # DataLoader with custom collate_fn
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    # Initialize model, criterion, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ContrastiveModel(t5_encoder).to(device)
    # model = CustomCERModel(configs=configs, device=device)
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

        # After training, calculate accuracy
        train_accuracy = calculate_accuracy(model, dataloader, device, threshold=1.0)
        print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
        accuracies.append(train_accuracy)

        # plot_metrics(train_losses, accuracies)
        sys.stdout.flush()  # Manually flush the buffer

if __name__ == "__main__":
    main()
