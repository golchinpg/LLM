import torch
from torch.utils.data import DataLoader, TensorDataset
from LLM import load_data, preprocess_data, TransformerBlock, evaluate_model

# Load and preprocess data
data = load_data("data.csv")
_, X_test, _, y_test = preprocess_data(data, target_column="target")

# Convert to PyTorch tensors
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load trained model
model = TransformerBlock(input_dim=X_test.shape[1], num_classes=len(set(y_test)))
model.load_state_dict(torch.load("model.pth"))

# Evaluate the model
evaluate_model(model, test_loader)