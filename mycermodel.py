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
# class ContrastiveDataset(Dataset):
#     def __init__(self, df):
#         self.data = df
            
#     def __len__(self):
#         return len(self.data)
        
#     def __getitem__(self, idx):
#         row = self.data.iloc[idx]
#         code_i_1 = row['code_i_1']
#         code_j_1 = row['code_j_1']
#         code_i_2 = row['code_i_2']
#         code_j_2 = row['code_j_2']
#         is_similar = row['is_similar']
        
#         # Prepare the concatenated inputs as per CustomCERModel's expectations
#         # A1, A2, B1, B2
#         A1 = code_i_1
#         A2 = code_j_1
#         B1 = code_i_2
#         B2 = code_j_2
        
#         # Return the concatenated inputs and label
#         return [A1, A2, B1, B2], torch.tensor(is_similar, dtype=torch.float)

# # Custom collate function to pad inputs
# def collate_fn(batch):
#     # batch is a list of tuples: ([A1, A2, B1, B2], label)
#     concatenated_inputs_batch = []
#     labels_batch = []
    
#     for item in batch:
#         concatenated_inputs_batch.extend(item[0])  # extend the list with A1, A2, B1, B2
#         labels_batch.append(item[1])
        
#     labels_batch = torch.stack(labels_batch)
#     return concatenated_inputs_batch, labels_batch


class CERDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        return {
            'A1': row['code_i_1'],
            'A2': row['code_j_1'],
            'B1': row['code_i_2'],
            'B2': row['code_j_2'],
            'label': 1 if row['is_similar'] else 0,
        }
    
class CollateForCER(object):
    def __init__(self, tokenizer, configs: dict, device: torch.device):
        self.tokenizer = tokenizer
        self.configs = configs
        self.device = device

    def __call__(self, batch):
        # Create a single list where each A1, A2, B1, and B2 will be concatenated consecutively
        concatenated_inputs = []
        labels = []

        # for item in batch:
        #     concatenated_inputs.append(item['A1'])  # Add A1
        #     concatenated_inputs.append(item['A2'])  # Add A2
        #     concatenated_inputs.append(item['B1'])  # Add B1
        #     concatenated_inputs.append(item['B2'])  # Add B2
        #     labels.append(item['label'])
        A1 = [item['A1'] for item in batch]
        A2 = [item['A2'] for item in batch]
        B1 = [item['B1'] for item in batch]
        B2 = [item['B2'] for item in batch]
        labels = [item['label'] for item in batch]
        concatenated_inputs = A1 + A2 + B1 + B2

        # Need to tokenize here for efficiency
        # inputs = self.tokenizer(code, return_tensors="pt", padding=True, truncation=True).to(self.device)
        # concatenated_inputs = tokenizer(concatenated_inputs, return_tensors='pt', padding=True, truncation=True)
        
        return {
            'inputs': concatenated_inputs,  # This is a single list containing A1, A2, B1, B2 in order
            'labels': torch.tensor(labels, dtype=torch.float)
        }
   
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
            # nn.ReLU()
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

    def forward(self, concatenated_inputs: List[str]) -> torch.Tensor:
        # Tokenize concatenated inputs
        tokenized_inputs = self.tokenizer(concatenated_inputs, return_tensors='pt', padding=True, truncation=True).to(self.device)
        # tokenized_inputs = {k: v.to(self.device) for k, v in tokenized_inputs.items()}

        # # Get embeddings for all concatenated inputs (A1, A2, B1, B2 in sequence)
        embeddings = self.get_embeddings(tokenized_inputs)
        # output = self.pretrained_encoder(tokenized_inputs['input_ids'],attention_mask=tokenized_inputs['attention_mask'])[0]
        # print(tokenized_inputs['input_ids'])
        # print(tokenized_inputs['attention_mask'])
        # embeddings = output[:,0,:]

        # Split embeddings into A1, A2, B1, and B2
        batch_size = embeddings.shape[0] // 4  # Since A1, A2, B1, B2 are concatenated
        A1_emb = embeddings[:batch_size]
        A2_emb = embeddings[batch_size:2 * batch_size]
        B1_emb = embeddings[2 * batch_size:3 * batch_size]
        B2_emb = embeddings[3 * batch_size:]

        # Compute Da and Db by subtracting corresponding embeddings
        Da = A1_emb - A2_emb
        Db = B1_emb - B2_emb
        # Da = (A1_emb + A2_emb) / 2
        # Db = (B1_emb + B2_emb) / 2

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

# # Contrastive Loss
# class ContrastiveLoss(nn.Module):
#     def __init__(self, margin=1.0):
#         super(ContrastiveLoss, self).__init__()
#         self.margin = margin
        
#     def forward(self, output1, output2, label):
#         euclidean_distance = F.pairwise_distance(output1, output2)
#         loss = torch.mean((label) * torch.pow(euclidean_distance, 2) +
#                           (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
#         return loss

class ContrastiveLoss(nn.Module):
    def __init__(self, device, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.device = device

    def forward(self, outputs: tuple[torch.tensor, torch.tensor], labels: torch.tensor) -> torch.tensor:
        output1, output2 = outputs
        output1 = output1.to(self.device)
        output2 = output2.to(self.device)
        labels = labels.to(self.device)
        distance = F.pairwise_distance(output1, output2)
        # print('Distance: ' + str(euclidean_distance))
        # print('Label: ' + str(label))
        
        # Assuming label is 1 for similar pairs and 0 for dissimilar pairs
        loss = (labels) * torch.pow(distance, 2) + (1-labels) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        # loss = torch.mean(loss)
        # loss = 0.5 * (label * output**2 + (1 - label) * F.relu(self.margin - output).pow(2))
        return loss.mean().to(self.device)
    
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
            concatenated_inputs_batch = batch['inputs']
            labels = batch['labels'].to(device)
            # Forward pass
            proj1, proj2 = model(concatenated_inputs_batch)

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
    accumulation_steps = 1
    len_data = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        concatenated_inputs_batch = batch['inputs']
        labels = batch['labels'].to(device)

        # optimizer.zero_grad()

        # Forward pass
        proj1, proj2 = model(concatenated_inputs_batch)

        # Compute contrastive loss
        loss = criterion((proj1, proj2), labels)

        # Backward pass and optimization
        loss.backward()
        # optimizer.step()
        if ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == len_data):
        # print('inside optimizer')
            optimizer.step()
            optimizer.zero_grad()


        running_loss += loss.item()
        
    return running_loss / len(dataloader)

# Main function
def main():
    # Load the dataset
    df = pd.read_pickle('data/current_dataset.pkl')
    print(df)
    dataset = CERDataset(df)
    # DataLoader with custom collate_fn
    configs = {
        'model_name': 't5-base',
        'code_change_vector_size': 128,
        'loss_fn': 'ContrastiveLoss'
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=CollateForCER(tokenizer, configs, device))

    # Initialize model, criterion, and optimizer
    

   

    model = CustomCERModel(configs=configs, device=device).to(device)
    criterion = ContrastiveLoss(device=device,margin=1.0)
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
