import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from utils import load_data
from dataset_collate_fn import CollateFnForGwR

def min_max_normalize_dataset(train_data, val_data, test_data):
    labels = [e[-1] for e in train_data]
    min_label, max_label = min(labels), max(labels)
    def normalize(data):
        return [(e[:-1] + ((e[-1] - min_label) / (max_label - min_label),)) for e in data]
    return normalize(train_data), normalize(val_data), normalize(test_data)

class DatasetForGwR(Dataset):
    
    def __init__(self, data): self.data = data
    
    def __len__(self): return len(self.data)
    
    def __getitem__(self, index): return self.data[index]
    

class DataModule(nn.Module):
    def __init__(self, args, train=True):
        super(DataModule, self).__init__()
        self.args = args
        self.train = train
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        
        if train:
            train_data, val_data = load_data(args, 'train'), load_data(args, 'valid')

        test_data = load_data(args, 'test')

        if train and args.task == 'regression':
            train_data, val_data, test_data = min_max_normalize_dataset(train_data, val_data, test_data)

        if train:
            self.train_dataset, self.val_dataset = DatasetForGwR(train_data), DatasetForGwR(val_data)

        self.test_dataset = DatasetForGwR(test_data)

        self.collate_fn_gwr = CollateFnForGwR(self.args)
        
    def forward(self):
        
        test_dl = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn_gwr,
            num_workers=self.num_workers,
            drop_last=False
        )
        if self.train:
            train_dl = DataLoader(
                dataset=self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=self.collate_fn_gwr,
                num_workers=self.num_workers,
                drop_last=False
            )
            val_dl = DataLoader(
                dataset=self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=self.collate_fn_gwr,
                num_workers=self.num_workers,
                drop_last=False 
            )
            return train_dl, val_dl, test_dl
        else:
            return test_dl