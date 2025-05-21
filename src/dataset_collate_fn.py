import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import (
    EsmTokenizer,
    BertTokenizer,
    AutoTokenizer,
    T5Tokenizer
)

import re
import torch


class CollateFnForGwR:
    
    def __init__(self, args):

        self.args = args
        self.device = args.device

        if args.plm_type == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(args.plm_dir)
        elif args.plm_type in ['esm', 'ankh']:
            self.tokenizer = AutoTokenizer.from_pretrained(args.plm_dir)
        elif args.plm_type == 'saprot':
            self.tokenizer = EsmTokenizer.from_pretrained(args.plm_dir)
        elif args.plm_type == 't5':
            self.tokenizer = T5Tokenizer.from_pretrained(args.plm_dir, do_lower_case=False)
        else:
            raise ValueError(f"Unsupported PLM type: {args.plm_type}")

    def __call__(self, batch):
        
        sequences = [protein["seq"] for protein in batch]

        if self.args.plm_type == 'bert': 
            sequences = [' '.join(seq) for seq in sequences]
        elif self.args.plm_type == 't5': 
            sequences = [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in sequences]
        
        if self.args.plm_type == 'saprot':
            max_len = max([len(seq) // 2 for seq in sequences])
        else:
            max_len = max([len(seq) for seq in sequences])

        if max_len > self.args.max_length: 
            max_len = self.args.max_length

        if self.args.plm_type == 't5':
            results = self.tokenizer(sequences, add_special_tokens=True, padding="longest")
        else:
            results = self.tokenizer(
                sequences,
                return_tensors="pt",
                padding=True,
                max_length=max_len,
                truncation=True,
            )
        if self.args.structgwr:
            coords = [protein["coords"] for protein in batch]
            coords = torch.stack(coords, dim=0).to(self.device)
            coord_mask = [protein["coord_mask"] for protein in batch]
            coord_mask = torch.stack(coord_mask, dim=0).to(self.device)
            padding_mask = [protein["padding_mask"] for protein in batch]
            padding_mask = torch.stack(padding_mask, dim=0).to(self.device)
            confidence = [protein["confidence"] for protein in batch]
            confidence = torch.stack(confidence, dim=0).to(self.device)
            results["coords"] = coords
            results["coord_mask"] = coord_mask
            results["padding_mask"] = padding_mask
            results["confidence"] = confidence
            if self.args.dssp_token or args.foldseek_seq:
                struc_seqs = [protein["struc_seq"] for protein in batch]
                struc_seqs = torch.stack(struc_seqs, dim=0).to(self.device)
                results["struc_seqs"] = struc_seqs
        target = [protein["label"] for protein in batch]
        target = torch.tensor(target).to(self.device)
        results["target"] = target

        return results
