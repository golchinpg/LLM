import os
import urllib.request
import re
import importlib
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

if not os.path.exists("the-verdict.txt"):
    url = ("https://raw.githubusercontent.com/rasbt/"
           "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
           "the-verdict.txt")
    file_path = "/Users/pegah/Desktop/KOM/Codes/Data/the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)

#reading the text:
with open(file_path, 'r', encoding='utf-8') as f:
    raw_text = f.read()

print(f"Total number of Characters:", len(raw_text))
print(raw_text[:99])

#Tokenize the text
def Tokenization(raw_text):
    result = re.split(r'([.,:;?!_"()\']|--|\s)', raw_text)
    result = [item for item in result if item.strip()]  
    return result
"""
#Tokenize the text
text = "Hello, world. This, is a test."
result = re.split(r'([.,:;?!_"()\']|--|\s)', text)
result = [item for item in result if item.strip()]
print(result)
"""
"""
preprocessed = Tokenization(raw_text)
print(Tokenization(raw_text)[:30])
print(len(Tokenization(raw_text)))

#Converting Tokens into Token IDs
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer, token in enumerate(all_tokens)}

for i, item in enumerate(vocab.items()):
    print(item)
    if i>50:
        break

#write a Tokenizer class
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [item if item in self.str_to_int
                        else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

tokenizer = SimpleTokenizerV1(vocab)
text = "Hello, do you like tea. Is this-- a test?"


ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))
"""
#####
#Using BPE (BytePairEncoding) to handle the unknown words
tokenizer = tiktoken.get_encoding("gpt2")
text = ("Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
     )
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)
strings = tokenizer.decode(integers)
print(strings)

#Create Dataset and Dataloader
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt, allowed_special = {"<|endoftext|>"})
        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids)-max_length, stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+1+max_length]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index) :
        return self.input_ids[index], self.target_ids[index]

#Create Dataloader
def create_dataloader_v1(txt, batch_size = 4, max_length=256, stride = 128, 
                         shuffle = True, drop_last= True, num_workers = 0):
    #initilize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    #create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    #create Dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size = batch_size, 
        shuffle    = shuffle, 
        drop_last  = drop_last, 
        num_workers= num_workers
    )
    return dataloader

#let's test the dataloader with the batchsize of 1 for an LLM with a context of 4
with open("/Users/pegah/Desktop/KOM/Codes/Data/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

vocab_size = 50257
output_dim = 256
context_length = 1024
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
position_embedding_layer = torch.nn.Embedding(context_length, output_dim)

batch_size = 8
max_length = 4

dataloader = create_dataloader_v1(raw_text, batch_size=batch_size, max_length=max_length, 
                                  stride=max_length, shuffle=False)

for batch in dataloader:
    x, y = batch
    token_embeddings = token_embedding_layer(x)
    pos_embeddings = position_embedding_layer(torch.arange(max_length))

    input_embeddings = token_embeddings+pos_embeddings

    break
