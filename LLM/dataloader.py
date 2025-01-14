import torch
from torch.utils.data import Dataset, DataLoader
from .data_preprocessing import tokenize_text
import tiktoken

class GPTDatasetV1(Dataset):
    def __init__(self, txt, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenize_text(txt)
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

"""
class create_txt_dataloader(DataLoader):
    def __init__(self, txt, batch_size = 4, max_length=256, stride = 128, 
                 shuffle = True, drop_last= True, num_workers = 1):
        self.txt = txt
        self.batch_size = batch_size
        self.max_length = max_length
        self.stride = stride
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.dataset = GPTDatasetV1(txt, max_length, stride)
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size = batch_size, 
            shuffle    = shuffle, 
            drop_last  = drop_last, 
            num_workers= num_workers
        )
        
    def __call__(self):
        return self.dataloader
"""
def create_txt_dataloader(txt, batch_size = 4, max_length=256, stride = 128, 
                         shuffle = True, drop_last= True, num_workers = 0):
    #initilize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    #create dataset
    dataset = GPTDatasetV1(txt, max_length, stride)
    #create Dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size = batch_size, 
        shuffle    = shuffle, 
        drop_last  = drop_last, 
        num_workers= num_workers
    )
    return dataloader


