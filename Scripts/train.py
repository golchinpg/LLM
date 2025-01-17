import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader, TensorDataset
from LLM import create_txt_dataloader, load_tabular_data, load_text_data, preprocess_data, TransformerBlock, train_model, evaluate_model, save_model, load_model, GPTModel
from LLM.data_preprocessing import tokenize_text
import tiktoken

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024, #represents the model's maximum input token count
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
print(type(train_data))
print(train_data[:10])
validation_data = raw_text[split_idx:]

# Tokenize text
tokenizer = tiktoken.get_encoding("gpt2")
#train_data = tokenizer.encode(train_data)
train_data = tokenize_text(train_data, tokenizer)  # Convert list to string
#validation_data = tokenizer.encode(validation_data)
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

"""
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