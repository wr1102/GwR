import torch
import torch.nn as nn
import torch.nn.functional as F
import random 
from Bio import SeqIO
import os
import re
import json
from json import load
import numpy as np
from tqdm import tqdm
import csv
from esm.data import BatchConverter, Alphabet
from typing import Sequence, Tuple, List

from model.module.gvp.util import load_coords

class CoordBatchConverter(BatchConverter):

    def __call__(self, 
                 raw_batch: Sequence[Tuple[Sequence, str]], 
                 coords_max_shape, 
                 confidence_max_shape,
                 device=None):
        self.alphabet.cls_idx = self.alphabet.get_idx("<cath>")
        batch = []
        for coords, confidence, seq in raw_batch:
            if confidence is None:
                confidence = 1.
            if isinstance(confidence, float) or isinstance(confidence, int):
                confidence = [float(confidence)] * len(coords)
            if seq is None:
                seq = 'X' * len(coords)
            batch.append(((coords, confidence), seq))
        coords_and_confidence, strs, tokens = super().__call__(batch)
        # pad beginning and end of each protein due to legacy reasons
        coords = [
            F.pad(torch.tensor(cd), (0, 0, 0, 0, 1, 1), value=np.inf)
            for cd, _ in coords_and_confidence
        ]
        confidence = [
            F.pad(torch.tensor(cf), (1, 1), value=-1.)
            for _, cf in coords_and_confidence
        ]
        coords = self.collate_dense_tensors(coords, pad_v=np.nan, max_shape=coords_max_shape)
        confidence = self.collate_dense_tensors(confidence, pad_v=-1., max_shape=confidence_max_shape)
        if device is not None:
            coords = coords.to(device)
            confidence = confidence.to(device)
            tokens = tokens.to(device)
        padding_mask = torch.isnan(coords[:, :, 0, 0])
        coord_mask = torch.isfinite(coords.sum(-2).sum(-1))
        confidence = confidence * coord_mask + (-1.) * padding_mask
        return coords, coord_mask, padding_mask, confidence

    def from_lists(self, coords_list, confidence_list=None, seq_list=None, device=None):
        batch_size = len(coords_list)
        if confidence_list is None:
            confidence_list = [None] * batch_size
        if seq_list is None:
            seq_list = [None] * batch_size
        raw_batch = zip(coords_list, confidence_list, seq_list)
        return self.__call__(raw_batch, device)

    @staticmethod
    def collate_dense_tensors(samples, pad_v, max_shape=None):
        if len(samples) == 0:
            return torch.Tensor()
        if len(set(x.dim() for x in samples)) != 1:
            raise RuntimeError(
                f"Samples has varying dimensions: {[x.dim() for x in samples]}"
            )
        (device,) = tuple(set(x.device for x in samples))  # assumes all on same device
        result = torch.empty(
            len(samples), *max_shape, dtype=samples[0].dtype, device=device
        )
        result.fill_(pad_v)
        for i in range(len(samples)):
            result_i = result[i]
            t = samples[i]
            slices = tuple(slice(0, min(dim, max_dim)) for dim, max_dim in zip(t.shape, max_shape))
            result_i[slices] = t[slices]

        return result
    
def load_data(args, run_mode='train'):
    
    def load_structure(pdb_path, pdbs, max_shape):
        
        pdb_suffix = '.ef.pdb' if 'esmfold' in pdb_path else '.pdb'
        if not os.path.exists(pdb_path):
            raise FileNotFoundError(f'Path {pdb_path} not found.')
        raw_batch = []
        with tqdm(total=len(pdbs)) as pbar:
            pbar.set_description(f'Loading proteins from {pdb_path}')
            for pdb in pdbs:
                if 'Solubility' in pdb_path:
                    coords, seqs = load_coords(os.path.join(pdb_path, pdb), ['A'])
                else:
                    coords, seqs = load_coords(os.path.join(pdb_path, pdb + pdb_suffix), ['A'])
                raw_batch.append((coords, None, seqs))
                pbar.update(1)
            alphabet = Alphabet.from_architecture("invariant_gvp")
            converter = CoordBatchConverter(alphabet)
            coords, coord_mask, padding_mask, confidence = converter(
                    raw_batch=raw_batch,
                    coords_max_shape=[max_shape, 3, 3],
                    confidence_max_shape=[max_shape])
        return (coords, coord_mask, padding_mask, confidence)
    
    def load_struc_seq(foldseek, path, chains=["A"], process_id=0):

        assert os.path.exists(foldseek), f"Foldseek not found: {foldseek}"
        assert os.path.exists(path), f"Pdb file not found: {path}"

        tmp_save_path = f"get_struc_seq_{process_id}.tsv"
        cmd = f"{foldseek} structureto3didescriptor -v 0 --threads 1 --chain-name-mode 1 {path} {tmp_save_path}"
        os.system(cmd)

        seq_dict = {}
        name = os.path.basename(path)
        with open(tmp_save_path, "r") as r:
            for line in r:
                desc, seq, struc_seq = line.split("\t")[:3]
                name_chain = desc.split(" ")[0]
                chain = name_chain.replace(name, "").split("_")[-1]
                if chains is None or chain in chains:
                    if chain not in seq_dict:
                        combined_seq = "".join([a + b.lower() for a, b in zip(seq, struc_seq)])
                        seq_dict[chain] = [seq, struc_seq, combined_seq]

        os.remove(tmp_save_path)
        os.remove(tmp_save_path + ".dbtype")
        return seq_dict
        
    def read_seq_labels(file_path):

        if not os.path.exists(file_path):
            raise FileNotFoundError(f'File {file_path} not found.')
        
        protein_data = []
        
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                protein_data.append({
                    'name': row['name'],
                    'seq': row['aa_seq'],
                    'ss8_seq': row['ss8_seq'],
                    'label': row['label']
                })


        keywords = ["MetalIonBinding", "Thermostability", "EC", "GO"]

        if any(keyword in args.dataset_file for keyword in keywords) and 'alphafold' in args.structure_file:
            
            for item in protein_data:
                item['name'] = item['name'].split('-')[-1] if '-' in item['name'] else item['name']
        
        if args.task in ['binaryclass', 'multiclass']:
            
            for item in protein_data:
                item['label'] = int(item['label'])
            
        elif args.task == 'regression':
            for item in protein_data:
                item['label'] = float(item['label'])

        elif args.task == 'multilabel':
            for item in protein_data:
                item['label'] = [1 if i in map(int, item['label'].split(',')) else 0 for i in range(args.num_labels)]
        
        return protein_data
    
    def process_dssp_seqs(dssp_seqs):
        dssp_vocab = {'STA': 0, 'H': 1, 'G': 2, 'I': 3, 'E': 4, 'B': 5, 'T': 6, 'S': 7, 'L': 8, 'END': 9}
        for index, dssp_seq in enumerate(dssp_seqs):
            if not all([item in dssp_vocab for item in dssp_seq]):
                raise ValueError('Invalid secondary structure sequence.')
            dssp_seqs[index] = [dssp_vocab['STA']] + [dssp_vocab[item] for item in dssp_seq] + [dssp_vocab['END']]
            if len(dssp_seqs[index]) > args.max_length:
                dssp_seqs[index] = dssp_seqs[index][:args.max_length]
            else: 
                dssp_seqs[index] = dssp_seqs[index] + [dssp_vocab['L']] * (args.max_length - len(dssp_seqs[index]))
        return torch.tensor(dssp_seqs)
    
    def process_foldseek_seqs(foldseek_seqs):
        foldseek_vocab = {'STA': 0, '<unk>': 1, 'A': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9, 
                          'K': 10, 'L': 11, 'M': 12, 'N': 13, 'P': 14, 'Q': 15, 'R': 16, 'S': 17, 'T': 18, 'V': 19, 
                          'W': 20, 'Y': 21, 'X': 22, 'END': 23}
        
        for index, foldseek_seq in enumerate(foldseek_seqs):
            foldseek_seqs[index] = [foldseek_vocab['STA']] + [foldseek_vocab.get(item, foldseek_vocab['<unk>']) for item in foldseek_seq] + [foldseek_vocab['END']]
            
            if len(foldseek_seqs[index]) > args.max_length:
                foldseek_seqs[index] = foldseek_seqs[index][:args.max_length]
            else:
                foldseek_seqs[index] = foldseek_seqs[index] + [foldseek_vocab['<unk>']] * (args.max_length - len(foldseek_seqs[index]))
        
        return torch.tensor(foldseek_seqs)

    protein_data = read_seq_labels(args.dataset_file+run_mode+'.csv')

    
    if args.dssp_token:
        dssp_seqs = [item['ss8_seq'] for item in protein_data]
        dssp_seqs = process_dssp_seqs(dssp_seqs)
        for i, item in enumerate(protein_data):
            item['struc_seq'] = dssp_seqs[i]
    if args.foldseek_seq:
        # real time loading
        pdb_suffix = '.ef.pdb' if 'esmfold' in args.structure_file else '.pdb'
        if 'Solubility' in args.structure_file: pdb_suffix = ''
        foldseek_seqs = [load_struc_seq(
            args.foldseek, os.path.join(args.structure_file, pdb + pdb_suffix), ["A"], 
            args.foldseek_process_id)["A"][1] for pdb in pdbs]
        foldseek_seqs = process_foldseek_seqs(foldseek_seqs)
        for i, item in enumerate(protein_data):
            item['struc_seq'] = foldseek_seqs[i]
    
    if args.plm_type == 'saprot':
        pdb_suffix = '.ef.pdb' if 'ESMFold' in args.dataset_file else '.pdb'
        if 'Solubility' in args.dataset_file: pdb_suffix = ''
        combined_seqs = [load_struc_seq(
            args.foldseek, os.path.join(args.structure_file, pdb + pdb_suffix), ["A"], 
            args.foldseek_process_id)["A"][2] for pdb in pdbs]
        for i, item in enumerate(protein_data):
            item['seq'] = combined_seqs[i]

    if args.structgwr:
        pdbs = [item['name'] for item in protein_data]
        structures = load_structure(args.structure_file, pdbs, args.max_length)
        for i, item in enumerate(protein_data):
            item['coords'] = structures[0][i]
            item['coord_mask'] = structures[1][i]
            item['padding_mask'] = structures[2][i]
            item['confidence'] = structures[3][i]
    
    return protein_data

def str2dict(value):
    """Convert string to dictionary."""
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError("Dictionary value is not valid.")

class Seed():
    
    def __init__(self, seed_value):
        super(Seed, self).__init__()
        self.seed_value = seed_value
    
    def set(self):
        
        random.seed(self.seed_value)
        np.random.seed(self.seed_value)
        torch.manual_seed(self.seed_value)
        os.environ['PYTHONHASHSEED'] = str(self.seed_value)
        if torch.cuda.is_available(): 
            torch.cuda.manual_seed(self.seed_value)
            torch.cuda.manual_seed_all(self.seed_value)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True
            
        return f'Seed has been set with value {self.seed_value}.'
    
class EarlyStopping:
    
    def __init__(self, 
                 patience=10,
                 indicator_larger_better=True):
        
        self.patience = patience
        self.counter = 0
        self.larger_better = indicator_larger_better
        if indicator_larger_better:
            self.best = -np.inf
        else:
            self.best = np.inf

    def early_stopping(self, current_indicator):
        
        update = (current_indicator > self.best) if self.larger_better else (current_indicator < self.best)
        
        if update:
            self.best = current_indicator
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience