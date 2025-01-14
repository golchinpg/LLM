import torch
import torch.nn as nn
#attention mechanism
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)
"""
# Query, Key, Value
query = inputs[1]
print("Query:", query)
atten_scores_2 = torch.matmul(query, inputs.T)
print("Attention Scores:", atten_scores_2)
atten_weights_2 = torch.nn.functional.softmax(atten_scores_2, dim=0)
print("Attention Weights:", atten_weights_2)
atten_output_2 = torch.matmul(atten_weights_2, inputs)
print("Attention Output:", atten_output_2)
"""
"""
atten_scores = torch.matmul(inputs, inputs.T)
print("Attention Scores:", atten_scores)
atten_weights = torch.nn.functional.softmax(atten_scores, dim=0)
print("Attention Weights:", atten_weights)
context_vector = torch.matmul(atten_weights, inputs)
print("Context Vector:", context_vector)
"""
#self_attention
x_2 = inputs[1]
d_in = inputs.shape[1]
d_out = 2

w_q = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)
print(w_q)
w_k = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)
w_v = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)

query_2 = torch.matmul(x_2, w_q)
key_2 = torch.matmul(x_2, w_k)
value_2 = torch.matmul(x_2, w_v)

print("Query:", query_2)

keys = torch.matmul(inputs, w_k)
values = torch.matmul(inputs, w_v)

print("Keys shape:", keys.shape)
print("Values shape:", values.shape)

atten_score_2 = torch.matmul(query_2, keys.T)
print("Attention Score:", atten_score_2)
atten_weight_2 = torch.nn.functional.softmax(atten_score_2, dim=0)
print("Attention Weight:", atten_weight_2)
context_vector_2 = torch.matmul(atten_weight_2, values)
print("Context Vector:", context_vector_2)


class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out):
        super(SelfAttention, self).__init__()
        self.w_q = nn.Linear(d_in, d_out)
        self.w_k = nn.Linear(d_in, d_out)
        self.w_v = nn.Linear(d_in, d_out)

    def forward(self, x):
        query = self.w_q(x)
        keys = self.w_k(x)
        values = self.w_v(x)

        atten_score = torch.matmul(query, keys.T)
        atten_weight = torch.nn.functional.softmax(atten_score, dim=0)
        context_vector = torch.matmul(atten_weight, values)
        return context_vector

torch.manual_seed(789)
sa_v2 = SelfAttention(d_in, d_out)
print(sa_v2(inputs))    

#mask the above diagonal elements
context_length = inputs.shape[0]
maks = torch.triu(torch.ones(context_length, context_length), diagonal=1)
print(maks)
mask_simple = torch.tril(torch.ones(context_length, context_length))
print(mask_simple)
#Dropout
dropout = nn.Dropout(0.5)
example = torch.ones(6,6)
print(example)
print(dropout(example))
print(inputs.shape)
batch = torch.stack([inputs, inputs], dim=0)
print(batch.shape)


class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, dropout=0.1, context_length = 6):
        super(CausalAttention, self).__init__()
        self.w_q = nn.Linear(d_in, d_out)
        self.w_k = nn.Linear(d_in, d_out)   
        self.w_v = nn.Linear(d_in, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        batch, seq_length, d_in = x.shape
        query = self.w_q(x)
        keys = self.w_k(x)
        values = self.w_v(x)

        atten_score = torch.matmul(query, keys.T)
        atten_score.masked_fill_(self.mask[:seq_length, :seq_length] == 0, float("-inf"))
        atten_weights = nn.function.softmax(atten_score, dim=0)
        atten_weights = self.dropout(atten_weights)
        context_vector = torch.matmul(atten_weights, values)
        return context_vector

#Multi-head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

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

        
