import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader, TensorDataset
from Source.data_preprocessing import tokenize_text, IDs_to_text
from Source.dataloader import create_txt_dataloader
from Source import *
import tiktoken

GPT_CONFIG_124M = {
    "vocab_size": 50257,   # Vocabulary size
    "context_length": 1024, # Shortened context length (orig: 1024)
    "embedding_dim": 768,        # Embedding dimension
    "num_heads": 12,         # Number of attention heads
    "num_layers": 12,        # Number of layers
    "drop_rate": 0.1,      # Dropout rate
    "qkv_bias": False      # Query-key-value bias
    }

# Load and preprocess data
def generate_text_simple(model, idx, max_new_tokens, context_size,
                        temprature = 0.0, top_k = None, eos_id = None):
    if idx.dim() == 1:
        idx = idx.unsqueeze(0)
    # idx is (batch, n_tokens) array of indices in the current context
    for _ in range(max_new_tokens):
        
        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        
        idx_cond = idx[:, -context_size:]
        
        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)
        
        # Focus only on the last time step
        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]  

        if top_k is not None:
            # Apply top-k sampling
            top_logits, _ = torch.topk(logits, top_k)
            min_value = top_logits[:, -1]
            logits = torch.where(logits<min_value, torch.tensor(float('-inf')).to(logits.device), logits)

        if temprature != 0.0:
            # Apply temperature scaling
            logits = logits / temprature

            # Apply softmax to get probabilities
            probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

            #sample from the distribution
            idx_next = torch.multinomial(probas, num_samples=1)  # (batch, 1)
        else:
            # Get the idx of the vocab entry with the highest probability value
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if idx_next.item() == eos_id:
            break
        

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx

"""
tokenizer = tiktoken.get_encoding("gpt2")
start_context = "Hello, I am"
encoded = tokenize_text(start_context, tokenizer)
#encoded = torch.tensor([tokenizer.encode(start_context)])
print(encoded.shape)
model = GPTModel(GPT_CONFIG_124M)
model.eval()

out = generate_text_simple(model, encoded, 
                           max_new_tokens=6, 
                           context_size=GPT_CONFIG_124M["context_length"])

print(out)
#convert the output to text
decoded = IDs_to_text(out, tokenizer)
print(decoded)
"""