import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader, TensorDataset
from LLM import create_txt_dataloader, load_tabular_data, load_text_data, preprocess_data, TransformerBlock, train_model, evaluate_model, save_model, load_model, GPTModel
from LLM.data_preprocessing import tokenize_text
import tiktoken
from LLM.utils import calculate_loss_loader
from LLM.training import train_model
from visualization import plot_losses

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256, #represents the model's maximum input token count
    "embedding_dim": 768, #is the embedding size for token inputs, converting each input token into a 768-dimensional vector
    "num_heads": 12,
    "num_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False #No bias for query, key, and value
    }
# Load and preprocess data
#load text data
raw_text = load_text_data("/Users/pegah/Desktop/KOM/Codes/Data/the-verdict.txt")
Train_ratio = 0.9
split_idx = int(Train_ratio*len(raw_text))
train_data = raw_text[:split_idx]
validation_data = raw_text[split_idx:]

# Tokenize text
tokenizer = tiktoken.get_encoding("gpt2")
train_data = tokenize_text(train_data, tokenizer)  # Convert list to string
validation_data = tokenize_text(validation_data, tokenizer)  # Convert list to string



torch.manual_seed(123)
train_loader = create_txt_dataloader(train_data, 
                                    batch_size = 2,
                                    max_length=GPT_CONFIG_124M["context_length"], 
                                    stride = GPT_CONFIG_124M["context_length"], 
                                    shuffle = True, 
                                    drop_last= True, 
                                    num_workers = 0)

val_loader = create_txt_dataloader(validation_data,
                                   batch_size = 2,
                                    max_length=GPT_CONFIG_124M["context_length"], 
                                    stride = GPT_CONFIG_124M["context_length"], 
                                    shuffle = False, 
                                    drop_last= False, 
                                    num_workers = 0)

print("Train loader:")
for x, y in train_loader:
    print(x.shape, y.shape)

print("\nValidation loader:")
for x, y in val_loader:
    print(x.shape, y.shape)

train_tokens = 0
for input_batch, target_batch in train_loader:
    train_tokens += input_batch.numel()

val_tokens = 0
for input_batch, target_batch in val_loader:
    val_tokens += input_batch.numel()

print("Training tokens:", train_tokens)
print("Validation tokens:", val_tokens)
print("All tokens:", train_tokens + val_tokens)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GPTModel(GPT_CONFIG_124M).to(device)
model.to(device)
torch.manual_seed(123) # For reproducibility due to the shuffling in the data loader

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
num_epochs = 30

train_losses, val_losses, track_tokens_seen = train_model(
    model = model, 
    train_loader = train_loader,
    val_loader = val_loader, 
    optimizer = optimizer,
    epochs = num_epochs,
    eval_freq = 5, 
    eval_iter = 5, 
    start_context = "Every effort moves you", 
    tokenizer = tokenizer, 
    device = device)

#visualize the training and validation losses
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
save_path = str(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Results", "train_losses.png")))
plot_losses(epochs_tensor, track_tokens_seen, train_losses, val_losses, save_path)

"""
with torch.no_grad():
    train_loss = calculate_loss_loader(train_loader, model, device)
    val_loss = calculate_loss_loader(val_loader, model, device)
print("Initial training loss:", train_loss)
print("Initial validation loss:", val_loss)

#test if it is working
print(dataloader)
for batch in dataloader:
    x, y = batch
    #x = x.float()
    #x = x.unsqueeze(-1)
    print(x.shape, y.shape)
    break

#x = torch.rand(2, 4, 768)  # Shape: [batch_size, num_tokens, emb_dim]
#block = TransformerBlock(GPT_CONFIG_124M)
#output = block(x)
#print("Input shape:", x.shape)
#print("Output shape:", output.shape)
print("Input shape:", x.shape)
print("Input:", x)

model = GPTModel(GPT_CONFIG_124M)
output = model(x)
print("Output shape:", output.shape)
print(output)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")

#load correct data
data = load_tabular_data("data.csv")
X_train, X_test, y_train, y_test = preprocess_data(data, target_column="target")

# Convert to PyTorch tensors
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize model
#adding the correct configuration for the TransformerBlock
model = TransformerBlock(input_dim=X_train.shape[1], num_classes=len(set(y_train)))

# Define loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_model(model, train_loader, criterion, optimizer)
"""