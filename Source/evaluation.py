import torch
from sklearn.metrics import accuracy_score

def evaluate_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, label in test_loader:
            output = model(inputs)
            loss = criterion(output, label)
            total_loss += loss.item()
            preds = torch.argmax(output, dim=1)
            all_preds.extend(preds)
            all_labels.extend(label)
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Loss: {total_loss/len(test_loader)}, Accuracy: {accuracy}")
    return accuracy