import torch
from torch.utils.data import DataLoader, TensorDataset
from LLM import load_data, preprocess_data, TransformerBlock, train_model, evaluate_model, save_model, load_model

# Load and preprocess data
#load correct data
data = load_data("data.csv")
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