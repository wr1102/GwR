import os 
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    EsmModel,
    T5EncoderModel,
    AutoModelForMaskedLM, 
    AutoTokenizer,
    EsmTokenizer, 
    BertModel,
    BertTokenizer,
    T5Tokenizer)
from peft import get_peft_model, LoraConfig



class SequenceEncoder(nn.Module):
    
    def __init__(self, args):
        super().__init__()
        
        self.args = args
        
        self.device = args.device
        
        assert args.tuning ^ args.plm_freeze, 'PLM can only be finetuned or freezed.'
        
        if args.plm_type in ['esm', 'saprot']:
            self.encoder = EsmModel.from_pretrained(args.plm_dir)
        if args.plm_type == 'bert':
            self.encoder = BertModel.from_pretrained(args.plm_dir)
        if args.plm_type == 'ankh':
            self.encoder = T5EncoderModel.from_pretrained(args.plm_dir)
        if args.plm_type == 't5':
            self.encoder = T5EncoderModel.from_pretrained(args.plm_dir)
            

        if args.tuning:
            if args.lora:
                if args.plm_type in ['esm', 'bert']:
                    target_modules = ["attention.self.query", "attention.self.value"]
                elif args.plm_type in ['ankh', 't5']:
                    target_modules = ["SelfAttention.q", "SelfAttention.v"]
                lora_config = LoraConfig(
                    r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.linear_dropout,
                    target_modules=target_modules,
                    fan_in_fan_out=False
                )
                self.encoder = get_peft_model(self.encoder, lora_config)
                
        if args.plm_freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False 
        
        self.max_len = args.max_length
         
        self.atten_map_layers = args.atten_map_layers
        
        self.rpa_threshold = torch.nn.Parameter(torch.tensor(args.rpa_threshold))
        
        self.output_dim = self.encoder.config.hidden_size
        
    def forward(self, input_ids, atten_mask=None):

        with torch.no_grad():
            plm_output = self.get_encoder_embedding(input_ids, atten_mask)
            output = plm_output.last_hidden_state
            
        rpa = self.residue_paired_association(plm_output.attentions[-self.atten_map_layers:])
            
        return output, rpa

    @torch.no_grad()
    def get_encoder_embedding(self, input_ids, atten_mask=None):
        
        plm_output = self.encoder(
            input_ids=input_ids,
            attention_mask=atten_mask,
            output_attentions=True
        )
        
        return plm_output

    def residue_paired_association(self, x):

        with torch.no_grad():
            attens = []
            for atten in x:
                attens.append(atten.mean(dim=1, keepdim=False))
            attens = torch.stack(attens).mean(dim=0, keepdim=False)
       
        
        return torch.where(attens < self.rpa_threshold, torch.tensor(0.0).to(self.device), attens)

    


    
    