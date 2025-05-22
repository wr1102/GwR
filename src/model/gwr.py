from model.module.aa_seq import SequenceEncoder, ProSST
from model.module.gwr_module import Layer_Averaged_GwR, Struct_GwR
from model.module.pooling import Attention1dPoolingHead
from model.module.classify import ClassificationHead
from .base import BaseModelOutput

import torch 
import torch.nn as nn
from torch.nn.functional import (
    binary_cross_entropy_with_logits,
    cross_entropy,
    one_hot
)
from torch.nn import (
    MSELoss,
    BCEWithLogitsLoss
)


class GwR(torch.nn.Module):
    
    def __init__(self, args):
        super().__init__()
        
        self.device = args.device
        self.args = args

        if args.plm_dir is not None:
            self.seq_encoder = SequenceEncoder(args)
        else:
            self.seq_encoder = None 
             
        self._initialize_gwr_modules(args)
        
        if not args.gwr_plm and not args.gwr_struc:
            self.embed_seq_encoder_output = nn.Linear(
                self.seq_encoder.output_dim, args.hidden_dim)
        
        self.classification_head = Attention1dPoolingHead(
            args.hidden_dim, args.num_labels, args.dropout
        ) if args.attention_pooling else ClassificationHead(args)

        if args.task == 'regression':
            self.loss_fn = MSELoss(reduction="mean")
        elif args.task == 'multilabel':
            self.loss_fn = BCEWithLogitsLoss(reduction="mean")
        elif args.task == 'multiclass':
            self.loss_fn = cross_entropy
        else:
            self.loss_fn = binary_cross_entropy_with_logits
        self.num_labels = args.num_labels
    
    def _initialize_gwr_modules(self, args):
        
        if args.gwr_struc:
            self.struct_gwr = Struct_GwR(args)
        
        if args.gwr_plm:
            self.la_gwr = Layer_Averaged_GwR(self.seq_encoder.output_dim, args)
            
            if args.intergrate_method == 'concat' and args.gwr_struc:
                args.hidden_dim = self.struct_gwr.output_dim + self.la_gwr.output_dim
            
            elif args.intergrate_method == 'cross_atten' and args.gwr_struc:
                self.embed_la_gwr_output = nn.Linear(self.la_gwr.output_dim, args.hidden_dim)
        
        if args.gwr_plm and args.gwr_struc:
            
            if args.intergrate_method == 'concat':
                self.gwr_plm_out_layernorm = nn.LayerNorm(self.la_gwr.output_dim)
                self.gwr_struc_out_layernorm = nn.LayerNorm(self.struct_gwr.output_dim)
            
            if args.intergrate_method == 'cross_atten':
                self.gwr_out_layer_norm = nn.LayerNorm(args.hidden_dim)
                self.mh_atten = nn.MultiheadAttention(
                    embed_dim=args.hidden_dim, num_heads=args.multihead_heads)
    
    def forward(self, batch, inference=False):
        '''
        inference: bool
            Whether to run the model in inference mode. If True, no gradients are calculated.
        '''
        if self.args.plm_dir is not None:
            output, rpa = self.seq_encoder(batch['input_ids'].to(self.device), batch['attention_mask'].to(self.device))
            atten_mask = batch['attention_mask']
        
        if self.args.gwr_plm: 
            output = self.la_gwr(output, rpa)
            if self.args.intergrate_method == 'cross_atten' and self.args.gwr_struc:
                output = self.embed_la_gwr_output(output)
            
            if self.args.gwr_struc:   
                struct_gwr_rep = self.struct_gwr(
                    batch['struc_seqs'].to(self.device),
                    batch['coords'].to(self.device),
                    batch['coord_mask'].to(self.device),
                    batch['padding_mask'].to(self.device),
                    batch['confidence'].to(self.device)
                )
                if self.args.intergrate_method == 'concat':
                    la_gwr_rep = self.gwr_plm_out_layernorm(output)
                    struct_gwr_rep = self.gwr_struc_out_layernorm(struct_gwr_rep)
                    output = torch.cat([la_gwr_rep, struct_gwr_rep], dim=-1)
                    
                elif self.args.intergrate_method == 'cross_atten':
                    la_gwr_rep = self.gwr_out_layer_norm(output)
                    struct_gwr_rep = self.gwr_out_layer_norm(struct_gwr_rep)
                    la_gwr_rep = la_gwr_rep.transpose(0, 1)
                    struct_gwr_rep = struct_gwr_rep.transpose(0, 1)
                    atten_output, _ = self.mh_atten(
                        query=struct_gwr_rep if self.args.query == 'struct_gwr' else la_gwr_rep,
                        key=la_gwr_rep if self.args.query == 'struct_gwr' else struct_gwr_rep,
                        value=la_gwr_rep if self.args.query == 'struct_gwr' else struct_gwr_rep
                    )
                    output = atten_output.transpose(0, 1)
                atten_mask = (~batch['padding_mask']).long()  
        
        elif self.args.gwr_struc:  
            output = self.struct_gwr(
                batch['struc_seqs'].to(self.device),
                batch['coords'].to(self.device),
                batch['coord_mask'].to(self.device),
                batch['padding_mask'].to(self.device),
                batch['confidence'].to(self.device)
            )
            atten_mask = (~batch['padding_mask'].to(self.device)).long()
        
        if not self.args.gwr_plm and not self.args.gwr_struc:
            output = self.embed_seq_encoder_output(output)
        
        if self.args.attention_pooling:
            logits = self.classification_head(output, atten_mask)
        else:
            logits = self.classification_head(output)

        if not inference:

            if self.args.task in ['regression', 'multilabel', 'binaryclass']:
                target = batch['target'].float().to(self.device)
            else:
                target = batch['target'].to(self.device)

            if self.args.task in ['regression', 'binaryclass']:
                logits = logits.squeeze(1)
                
            loss = self.loss_fn(logits,  target)
            return logits

        return BaseModelOutput(
            hidden_states=output,
            logits=logits,
            loss=loss if not inference else None
        )
        
    
    def inference(self, batch):
        
        with torch.no_grad():
            return self.forward(batch, inference=True)
