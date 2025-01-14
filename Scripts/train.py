import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader, TensorDataset
from LLM import create_txt_dataloader, load_tabular_data, load_text_data, preprocess_data, TransformerBlock, train_model, evaluate_model, save_model, load_model

# Load and preprocess data
#load text data
raw_text = load_text_data("/Users/pegah/Desktop/KOM/Codes/Data/the-verdict.txt")
print(raw_text[:99])
dataloader = create_txt_dataloader(raw_text, batch_size = 4, max_length=256, stride = 128, 
                         shuffle = True, drop_last= True, num_workers = 0)
#test if it is working
print(dataloader)
for batch in dataloader:
    x, y = batch
    print(x.shape, y.shape)
    break

"""
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