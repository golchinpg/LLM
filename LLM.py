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
        self.tok_embed = nn.Embedding(config["vocab_size"], config["embedding_dim"])
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

class DummyTransformerBlock(nn.Module):
    def __init__(self, config):
        super(DummyTransformerBlock, self).__init__()
        
    
    def forward(self, x):
        return x
    
class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super(DummyLayerNorm, self).__init__()
    
    def forward(self, x):
        return x
    
test = nn.Parameter(torch.ones(5))
print("test", test)
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
    

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
text1 = "Every effort moves you"
text2 = "Every day holds a"

#Encode the text
batch.append(torch.tensor(tokenizer.encode(text1)))
batch.append(torch.tensor(tokenizer.encode(text2)))
batch = torch.stack(batch, dim=0)
#print(batch)

torch.manual_seed(123)
"""
model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)
print ("Logits shape:", logits.shape)
print ("Logits:", logits)
"""
batch_example = torch.randn(2, 5)
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)
print(out)
mean = out.mean(dim=1)
var = out.var(dim=1)
print("mean:", mean)
print("var:", var)