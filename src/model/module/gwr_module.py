import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from model.module.gvp.features import GVPGraphEmbedding
from model.module.gvp.gvp_modules import GVPConvLayer
from model.module.gvp.gvp_utils import unflatten_graph, flatten_graph
from model.module.gvp.util import rotate, get_rotation_frames


class Struct_GwR(nn.Module):
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        self.embed_graph = GVPGraphEmbedding(args)

        node_hidden_dim = (args.node_hidden_dim_scalar,
                           args.node_hidden_dim_vector)
        edge_hidden_dim = (args.edge_hidden_dim_scalar,
                           args.edge_hidden_dim_vector)

        conv_activations = (F.relu, torch.sigmoid)
        self.encoder_layers = nn.ModuleList(
            GVPConvLayer(
                node_hidden_dim,
                edge_hidden_dim,
                drop_rate=args.dropout,
                vector_gate=True,
                attention_heads=0,
                n_message=3,
                conv_activations=conv_activations,
                n_edge_gvps=0,
                eps=1e-4,
                layernorm=True,
            )
            for i in range(args.num_encoder_layers)
        )
        
        if args.dssp_token or args.foldseek_seq:
            self.embed_struc_seq = nn.Embedding(
                args.dssp_dim if args.dssp_token else args.foldseek_seq_dim,
                args.node_hidden_dim_scalar) 
            self.linear_struc_seq = nn.Linear(args.node_hidden_dim_scalar, args.node_hidden_dim_scalar)
            self.linear_norm = nn.LayerNorm(args.node_hidden_dim_scalar)
        
        self.output_dim = args.node_hidden_dim_scalar + (3 * args.node_hidden_dim_vector)
    
    def forward(self, struc_seqs, coords, coord_mask, padding_mask, confidence):
        
        embeddings = self.get_embeddings(
            struc_seqs, coords, coord_mask, padding_mask, confidence)
        
        struc_seq_embeddings, graph_embeddings = embeddings
        
        if self.args.dssp_token or self.args.foldseek_seq:
            struc_seq_embeddings = self.linear_struc_seq(struc_seq_embeddings)
            struc_seq_embeddings = self.linear_norm(struc_seq_embeddings)
        
            
        return self.process_embeddings(struc_seq_embeddings, graph_embeddings, coords)

    def get_embeddings(self, struc_seqs, coords, coord_mask, padding_mask, confidence):
        
            return (
                self.embed_struc_seq(struc_seqs) if self.args.dssp_token or self.args.foldseek_seq else None, 
                self.embed_graph(coords, coord_mask, padding_mask, confidence)
            )

    def process_embeddings(self, struc_seq_embeddings, graph_embeddings, coords):
        
        node_embeddings, edge_embeddings, edge_index = graph_embeddings
        
        if struc_seq_embeddings is not None:
            node_embeddings_scalars, node_embeddings_vectors = unflatten_graph(
                node_embeddings, coords.shape[0])
            node_embeddings_scalars = self.linear_norm(node_embeddings_scalars) + struc_seq_embeddings
            node_embeddings = flatten_graph(
                node_embeddings=(node_embeddings_scalars, node_embeddings_vectors))
        
        for layer in self.encoder_layers:
            node_embeddings, edge_embeddings = layer(node_embeddings, edge_index, edge_embeddings)
        
        struct_embeddings = unflatten_graph(node_embeddings, coords.shape[0])
        
        return self.output_pattern(coords, struct_embeddings)
        
    def output_pattern(self, coords, struct_embeddings):
        struct_rep_scalars, struct_rep_vectors = struct_embeddings
        R = get_rotation_frames(coords)
        output = torch.cat([struct_rep_scalars,
                            rotate(struct_rep_vectors, R.transpose(-2, -1)).flatten(-2, -1),
                            ], dim=-1)

        return output 



class Layer_Averaged_GwR(nn.Module):
    
    def __init__(self, gi_dim, args):
        super().__init__()
        
        self.layer_in = GCNConv(gi_dim, args.gh_dim)
        
        self.layers_hidden = nn.ModuleList(
            [GCNConv(args.gh_dim, args.gh_dim) for _ in range(args.g_layers-2)])
        
        self.layer_out = GCNConv(args.gh_dim, args.go_dim)
        
        self.output_dim = args.go_dim
        
    def forward(self, aa_rep, rpa):
        
        outputs = []
        
        for i in range(aa_rep.shape[0]):
            rep = aa_rep[i]
            edge = self.get_graph_edge(rpa[i])
            reps = [rep]
        
            reps.append(torch.relu(self.layer_in(reps[-1], edge)))
            for layer in self.layers_hidden:
                reps.append(torch.relu(layer(reps[-1], edge)))
            reps.append(torch.relu(self.layer_out(reps[-1], edge)))
            
            outputs.append(torch.mean(torch.stack(reps[1:], dim=0), dim=0))
        
        return torch.stack(outputs, dim=0)

    def get_graph_edge(self, rpa):
        
        edge_indices = torch.nonzero(rpa, as_tuple=False)
        edge_indices = edge_indices[edge_indices[:, 0] != edge_indices[:, 1]]

        edge_start = edge_indices[:, 0]
        edge_end = edge_indices[:, 1]
        edge = torch.stack([edge_start, edge_end], dim=0)
        
        return edge
