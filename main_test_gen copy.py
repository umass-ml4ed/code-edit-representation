import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Dummy Dataset (Replace with your actual dataset)
class CodeDataset(Dataset):
    def __init__(self, encoded_vectors, target_sequences, tokenizer, max_length=512):
        self.encoded_vectors = encoded_vectors
        self.target_sequences = target_sequences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.encoded_vectors)

    def __getitem__(self, idx):
        encoded_vector = self.encoded_vectors[idx]
        target_sequence = self.target_sequences[idx]
        target_encoding = self.tokenizer(target_sequence, return_tensors="pt", max_length=self.max_length, padding='max_length', truncation=True)

        # Convert encoded_vector to LongTensor
        return encoded_vector.long(), target_encoding.input_ids.squeeze()

# Load tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-base')

# Load the encoder checkpoint
checkpoint = 'checkpoints/20241029_134451/' # All problems
cer_model = torch.load(checkpoint + '/model')
encoder_model = T5ForConditionalGeneration.from_pretrained('t5-base')
encoder_model.encoder.load_state_dict(cer_model.pretrained_encoder.state_dict(), strict=False)

# Freeze the encoder weights
for param in encoder_model.encoder.parameters():
    param.requires_grad = False

# Create a new decoder (from T5)
decoder_model = T5ForConditionalGeneration.from_pretrained('t5-base')

# Sample data (replace with your actual data)
encoded_vectors = [torch.randint(0, 32128, (512,)) for _ in range(10)]  # Dummy token IDs (0-32127 is usual T5 range)
target_sequences = ['public void example() { ... }'] * 10  # Dummy target sequences

# Prepare dataset and dataloader
dataset = CodeDataset(encoded_vectors, target_sequences, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Training settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
decoder_model.to(device)
optimizer = AdamW(decoder_model.parameters(), lr=1e-4)
loss_fn = CrossEntropyLoss()

# Training loop
decoder_model.train()
num_epochs = 3

for epoch in range(num_epochs):
    for batch in dataloader:
        encoded_vectors, target_tokens = batch
        encoded_vectors = encoded_vectors.to(device)
        target_tokens = target_tokens.to(device)

        # Forward pass
        outputs = decoder_model(
            input_ids=encoded_vectors,
            labels=target_tokens
        )
        loss = outputs.loss

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Save the fine-tuned decoder
# torch.save(decoder_model.state_dict(), 'fine_tuned_decoder.pth')

# Code generation example
decoder_model.eval()

# Example vector for generation (use actual data)
example_vector = torch.randint(0, 32128, (1, 512)).to(device)  # Single batch for demonstration

# Generating code from the vector
with torch.no_grad():
    generated_ids = decoder_model.generate(input_ids=example_vector, max_length=100)
    generated_code = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

print(f"Generated Code:\n{generated_code}")