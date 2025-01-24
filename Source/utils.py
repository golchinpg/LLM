import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Source.data_preprocessing import tokenize_text, IDs_to_text
from Test.word_generation import generate_text_simple
class save_model:
    def __init__(self, model, path):
        torch.save(model.state_dict(), path)


class load_model:
    def __init__(self, model, path):
        model.load_state_dict(torch.load(path))
        model.eval()


class LayerNorm(nn.Module):
    def __init__(self, embedding_dim):
        super(LayerNorm, self).__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(embedding_dim))
        self.shift = nn.Parameter(torch.zeros(embedding_dim))
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True) 
        norm_x = (x-mean)/torch.sqrt(var+self.eps)
        return self.scale * norm_x + self.shift


class FeedForward(nn.Module):
    def __init__(self, config):
        super(FeedForward, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(config["embedding_dim"], config["embedding_dim"]*4),
            nn.GELU(),
            nn.Linear(config["embedding_dim"]*4, config["embedding_dim"])
        )
    def forward(self, x):
        return self.layers(x)

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256, #represents the model's maximum input token count
    "embedding_dim": 768, #is the embedding size for token inputs, converting each input token into a 768-dimensional vector
    "num_heads": 12,
    "num_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False #No bias for query, key, and value
    }
class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg["embedding_dim"] % cfg["num_head"] == 0, "d_out must be divisible by num_heads"

        self.d_out = cfg["embedding_dim"]
        self.num_heads = cfg["num_head"]
        self.head_dim = cfg["embedding_dim"] // cfg["num_head"] # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(cfg["embedding_dim"], cfg["embedding_dim"], True)
        self.W_key = nn.Linear(cfg["embedding_dim"], cfg["embedding_dim"], True)
        self.W_value = nn.Linear(cfg["embedding_dim"], cfg["embedding_dim"], True)
        self.out_proj = nn.Linear(cfg["embedding_dim"], cfg["embedding_dim"])  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(cfg["drop_rate"])
        self.register_buffer('mask', torch.triu(torch.ones(cfg["context_length"], cfg["context_length"]), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x) # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head
        
        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2) 
        
        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # optional projection

        return context_vec
    
def calculate_loss_batch(model, input_batch, target_batch, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calculate_loss_loader(dataloader, model, device, num_batches = None):
    total_loss = 0
    
    if len(dataloader) == 0:
        return (float("nan"))
    elif num_batches == None:
        num_batches = len(dataloader)
    else:
    
        num_batches = min(num_batches, len(dataloader))
    
    for i, (input_batch, target_batch) in enumerate(dataloader):
        loss = calculate_loss_batch(model, input_batch, target_batch, device)
        total_loss += loss.item()
        if i >= num_batches - 1:
            break
    return total_loss / num_batches

def generate_and_print_samples(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = torch.tensor(tokenize_text(start_context, tokenizer)).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(model = model,
                                        idx = encoded, 
                                        max_new_tokens = 50, 
                                        context_size = context_size)
    decoded_text = IDs_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()

