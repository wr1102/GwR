from model.module.kan import KANLinear
import torch.nn as nn
import torch

class ClassificationHead(nn.Module):
   
    def __init__(self, args):
        super().__init__()
        
        self.args = args
        
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(args.dropout)
        
        self.linear_in = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.linear_out = nn.Linear(args.hidden_dim, args.num_labels)
        
    def forward(self, x):
        
        x = x.transpose(1, 2)
        x = self.pooling(x).squeeze(-1)
        x = self.linear_in(x)
        x = self.dropout(x)
        x = torch.relu(x)
        x = self.linear_out(x)
        
        return x
