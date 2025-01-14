import matplotlib
import tiktoken
import torch
import torch.nn as nn
from Attention_mechanism import MultiHeadAttention
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
    
class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()
    def forward(self, x):
        return x*0.5*(1+torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi))*(x+0.044715*torch.pow(x, 3))))
    
class FeedForward(nn.Module):
    def __init__(self, config):
        super(FeedForward, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(config["embedding_dim"], config["embedding_dim"]*4),
            GELU(),
            nn.Linear(config["embedding_dim"]*4, config["embedding_dim"])
        )
    def forward(self, x):
        return self.layers(x)


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
#print(out)
mean = out.mean(dim=1) # to take a mean vertically
var = out.var(dim=1)    # to take a variance vertically
#print("mean:", mean)
#print("var:", var)

out_norm = (out-mean.unsqueeze(1))/torch.sqrt(var.unsqueeze(1))
#print("Normalized layer output:", out_norm)
mean = out_norm.mean(dim=1, keepdim=True)   
var = out_norm.var(dim=1, keepdim=True)
#print("mean:", mean)
#print("var:", var)

#Adding residual connection to avoid vanishing gradient. In fact, the residual connection is added before the normalization
#Let's check the differences:

class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())
        ])
        #print("Layers:", self.layers)
    def forward(self, x):
        for layer in self.layers:
            layer_out = layer(x)
            """
            print("Layer output:", layer_out)
            print("Layer output shape:", layer_out.shape)
            print("Input shape:", x.shape)
            print(self.use_shortcut)
            """
        if self.use_shortcut == "True" and x.shape == layer_out.shape:
            #print("Adding Shortcut")
            return layer_out + x
        else:
            #print("No Shortcut")
            return layer_out
        
def print_gradient(model, x):
    #forward pass
    output = model(x)
    target = torch.tensor([[0.]])

    #calculate the loss
    loss = nn.MSELoss()(output, target)

    #backward pass
    loss.backward()

    for name, param in model.named_parameters():
        if "weight" in name:
            print(f"Gradient of {name}:", param.grad)

layer_sizes = [3, 3, 3, 3, 3, 1]  

sample_input = torch.tensor([[1., 0., -1.]])
model = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut="True")

#print_gradient(model, sample_input)

#
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["embedding_dim"],
            d_out=cfg["embedding_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["num_heads"], 
            dropout=cfg["drop_rate"]
            )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["embedding_dim"])
        self.norm2 = LayerNorm(cfg["embedding_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x

torch.manual_seed(123)

x = torch.rand(2, 4, 768)  # Shape: [batch_size, num_tokens, emb_dim]
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)
