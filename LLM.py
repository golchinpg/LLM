import matplotlib
import tiktoken
import torch
import torch.nn as nn

#Configuration of GPT2 with 124 million parameters
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024, #represents the model's maximum input token count
    "embedding_dim": 768, #is the embedding size for token inputs, converting each input token into a 768-dimensional vector
    "num_heads": 12,
    "num_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False #No bias for query, key, and value
    }

class DummyGPTModel(nn.Module):
    def __init__(self, config):
        super(DummyGPTModel, self).__init__()
        self.tok_embed = nn.Embedding(config["vocat_size"], config["embedding_dim"])
        self.pos_embed = nn.Embedding(config["context_length"], config["embedding_dim"])
        self.drop = nn.Dropout(config["drop_rate"])
        self.blocks = nn.Sequential(*[DummyTransformerBlock(config) for _ in range(config["num_layers"])])
        self.final_norm = DummyLayerNorm(config["embedding_dim"])
        self.out_head = nn.Linear(config["embedding_dim"], config["vocab_size"])
    
    def forward(self, x):
        tokens = self.tok_embed(x)
        positions = self.pos_embed(torch.arange(x.size(1)))
        x = self.drop(tokens+positions)
        x = self.blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
