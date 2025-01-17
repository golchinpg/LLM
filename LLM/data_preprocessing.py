import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tiktoken
import torch

def load_tabular_data(file_path):
    data = pd.read_csv(file_path)
    return data

def load_text_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    return raw_text

def tokenize_text(text, tokenizer):
    tokenized_text = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    #tokenized_tensor = torch.tensor(tokenized_text).unsqueeze(0)
    return tokenized_text

def IDs_to_text(token_IDs, tokenizer):
    flat = token_IDs.squeeze(0)
    return tokenizer.decode(flat.tolist())



def preprocess_data(data,target_name):
    X = data.drop(target_name, axis=1)
    y = data[target_name]
    normalizer = StandardScaler()
    X_scaled = normalizer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
