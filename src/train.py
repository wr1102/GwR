import gc
from argparse import ArgumentParser
import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import csv
from accelerate import Accelerator
from copy import deepcopy
import os
from utils import Seed, EarlyStopping, str2dict
from dataset import DataModule
from model.gwr import GwR
from metrics import Metrics

import time
import pynvml
import psutil
from fvcore.nn import FlopCountAnalysis, parameter_count_table
import torch.multiprocessing as mp

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

def get_gpu_stats():
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    return {
        "gpu_mem_used_MB": mem_info.used / 1024 ** 2,
        "gpu_util_percent": util.gpu
    }

def get_cpu_stats():
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.1)
    }

def log_metrics_to_csv(
    csv_path,
    epoch,
    epoch_time,
    gpu_mem_mb,
    gpu_util_percent,
    max_cpu_percent
):
    if not os.path.exists(csv_path):
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "epoch_time_s", "gpu_mem_MB",
                "gpu_util_percent", "max_cpu_percent"
            ])

    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            round(epoch_time, 2),
            round(gpu_mem_mb, 2),
            round(gpu_util_percent, 1),
            round(max_cpu_percent, 1)
        ])

class Trainer(nn.Module):

    def __init__(self, model, args):
        
        super(Trainer, self).__init__()

        Seed(args.seed).set()
        
        self.args = args
        self.model = model
        self.epoch = args.epoch
        self.metrics = Metrics(
            args.num_labels, args.task, args.device
        )
        self.es = EarlyStopping(args.patience)
        self.accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
        self.optimizer =  torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=args.init_lr)
        if args.lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=args.epoch, eta_min=args.min_lr)
        
        train_params = sum(p.numel() for p in model.parameters() if p.requires_grad) // 1e3
        all_params = sum(p.numel() for p in model.parameters()) // 1e6
        print(f"Trainable parameters: {train_params}k; All parameters: {all_params}M")

        self.info = f"Gwr({args.plm_dir.split('/')[-1]})_task_{args.dataset_file.split('/')[-2]}_train{train_params}K_all{all_params}M"

        self.save_model = args.save_model    
        self.best_state = None
        self.log = self.init_log(args.task)
        
        self.best_epoch = 0
        
    def init_log(self, task):
        if task == 'regression':
            return {
                'train_loss': [], 
                'val_loss': [], 'val_spearman': [], 'val_mse': [], 'val_mae': [], 'val_r2': [],
                'test_spearman': [], 'test_mse': [], 'test_mae': [], 'test_r2': []
            }
        elif task == 'multilabel':
            return {
                'train_loss': [], 
                'val_loss': [], 'val_aupr': [], 'val_f1max': [],
                'test_aupr': [], 'test_f1max': []
            }
        else:
            return {
                'train_loss': [], 
                'val_loss': [], 'val_acc': [], 'val_precision': [], 'val_recall': [], 'val_f1': [], 'val_mcc': [],
                'test_acc': [], 'test_precision': [], 'test_recall': [], 'test_f1': [], 'test_mcc': []
            }

    def update_log(self, phase, loss=None, metrics=None):

        if phase in ['train', 'val'] and loss is not None:
            self.log[f'{phase}_loss'].extend(loss)

        if metrics is not None:
            if self.args.task == 'regression':
                self.log[f'{phase}_spearman'].append(metrics['spearman'].item())
                self.log[f'{phase}_mse'].append(metrics['mse'].item())
                self.log[f'{phase}_mae'].append(metrics['mae'].item())
                self.log[f'{phase}_r2'].append(metrics['r2'].item())
            elif self.args.task == 'multilabel':
                self.log[f'{phase}_aupr'].append(metrics['aupr'].item())
                self.log[f'{phase}_f1max'].append(metrics['f1max'].item())
            else:
                self.log[f'{phase}_acc'].append(metrics['acc'].item())
                self.log[f'{phase}_precision'].append(metrics['precision'].item())
                self.log[f'{phase}_recall'].append(metrics['recall'].item())
                self.log[f'{phase}_f1'].append(metrics['f1'].item())
                self.log[f'{phase}_mcc'].append(metrics['mcc'].item())

    
    def print_metrics(self, metrics):

        if self.args.task == 'regression':
            print(f"== Spearman: {round(metrics['spearman'].item(), 4)}")
            print(f"== MSE: {round(metrics['mse'].item(), 4)}")
            print(f"== MAE: {round(metrics['mae'].item(), 4)}")
            print(f"== R2: {round(metrics['r2'].item(), 4)}")

        elif self.args.task == 'multilabel':
            print(f"== AUPR: {round(metrics['aupr'].item(), 4)}")  
            print(f"== F1-max: {round(metrics['f1max'].item(), 4)}")

        else:
            print(f"== Accuracy: {round(metrics['acc'].item(), 4)}")
            print(f"== Precision: {round(metrics['precision'].item(), 4)}")
            print(f"== Recall: {round(metrics['recall'].item(), 4)}")
            print(f"== F1-score: {round(metrics['f1'].item(), 4)}")
            print(f"== MCC: {round(metrics['mcc'].item(), 4)}")


    def forward(self, data_module):


        print(f'>> Start training {self.info}...')
        train_dl, val_dl, test_dl = data_module()
        self.model, self.optimizer, train_dl = self.accelerator.prepare(
            self.model, self.optimizer, train_dl
        )
        print('Start training...')
        for epoch in range(self.epoch):

            gpu_mem_list = []
            gpu_util_list = []
            cpu_percent_list = []
            tl = []; vl = []

            self.model.train()
            torch.cuda.reset_peak_memory_stats()
            start_time = time.time()

            with tqdm(total=len(train_dl)) as pbar:
                pbar.set_description(f'Training Epoch {epoch+1}/{self.epoch}')
                for batch_idx, batch in enumerate(train_dl):
                    tl.append(self.training_step(batch))
                    if self.args.record_source and batch_idx % 5 == 0:
                        torch.cuda.synchronize()
                        gpu_stats = get_gpu_stats()
                        cpu_stats = get_cpu_stats()
                        gpu_mem_list.append(gpu_stats["gpu_mem_used_MB"])
                        gpu_util_list.append(gpu_stats["gpu_util_percent"])
                        cpu_percent_list.append(cpu_stats["cpu_percent"])
                    pbar.set_postfix({'current loss': sum(tl)/len(tl)})
                    pbar.update(1)
            self.update_log('train', tl)
            print(f">> Epoch {epoch+1} Loss: {sum(tl)/len(tl)}")
            if self.args.record_source:
                epoch_time = time.time() - start_time
                avg_gpu_mem = sum(gpu_mem_list) / len(gpu_mem_list)
                avg_gpu_util = sum(gpu_util_list) / len(gpu_util_list)
                max_cpu_percent = max(cpu_percent_list)
                log_metrics_to_csv(
                    self.args.source_log,
                    epoch + 1,
                    epoch_time,
                    avg_gpu_mem,
                    avg_gpu_util,
                    max_cpu_percent
                )

            self.model.eval()
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_dl):
                    vl.append(self.validation_step(batch))
                metrics_res = self.metrics.compute()
                self.metrics.reset()
                self.update_log('val', vl, metrics_res)
                print(f">> Valid loss: {sum(vl)/len(vl)}")
                self.print_metrics(metrics_res)
                if self.args.task in ['multiclass', 'binaryclass']:
                    key_metric = 'acc'
                elif self.args.task == 'multilabel':
                    key_metric = 'f1max'
                elif self.args.task == 'regression':
                    key_metric = 'spearman'
                if metrics_res[key_metric] >= max(self.log[f'val_{key_metric}']):
                    print(f'>> Save best model at epoch {epoch+1}')
                    self.best_state = deepcopy(self.model.state_dict())
                    self.best_epoch = epoch + 1
            if self.es.early_stopping(metrics_res[key_metric]): 
                print(f'>> Early stop at epoch {epoch+1}'); break 
            
        print(f'>> Training finished.\n>> Testing...')
        print(f'>> Best epoch: {self.best_epoch}')
        self.model.load_state_dict(self.best_state)
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_dl):
                self.test_step(batch)
            metrics_res = self.metrics.compute()
            self.print_metrics(metrics_res)
            self.update_log('test', metrics=metrics_res)
        self.save_log()
        if self.args.save_model:
            save_path=f'/checkpoint/{self.info}_{self.args.seed}.pth'
            torch.save(self.best_state, save_path)
            print(f'model saved at {save_path}')
        gc.collect()
      
    def training_step(self, batch):
        self.optimizer.zero_grad()
        with self.accelerator.accumulate(self.model):
            loss = self.model(batch).loss
            self.accelerator.backward(loss)
            self.optimizer.step()
            if self.args.lr_scheduler:
                self.scheduler.step()
            return round(loss.item(), 4)    

    def validation_step(self, batch):
        output = self.model(batch)
        self.metrics.update(output.logits, batch['target'])
        return round(self.model(batch).loss.item(), 4)

    def test_step(self, batch):
        self.metrics.update(self.model(batch).logits, batch['target'].to(self.args.device))

        
    def save_log(self):

        file_name = f'/csv/{self.info}_{self.args.seed}_log.csv'
        
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)

            task_headers = {
                'regression': ['val_spearman', 'val_mse', 'val_mae', 'val_r2',
                            'test_spearman', 'test_mse', 'test_mae', 'test_r2'],
                'multilabel': ['val_aupr', 'val_f1max', 'test_aupr', 'test_f1max'],
                'classification': ['val_acc', 'val_precision', 'val_recall', 'val_f1', 'val_mcc',
                                'test_acc', 'test_precision', 'test_recall', 'test_f1', 'test_mcc']
            }

            task_type = self.args.task if self.args.task in task_headers else 'classification'
            metric_headers = task_headers[task_type]

            header = ['step/epoch', 'train_loss', 'val_loss'] + metric_headers
            writer.writerow(header)

            max_length = max(len(self.log['train_loss']), len(self.log['val_loss']), 
                            *[len(self.log[m]) for m in metric_headers])
            for i in range(max_length):
                row = [i + 1, 
                    self.log['train_loss'][i] if i < len(self.log['train_loss']) else '',
                    self.log['val_loss'][i] if i < len(self.log['val_loss']) else '']

                for metric in metric_headers:
                    row.append(self.log[metric][i] if i < len(self.log[metric]) else '')

                writer.writerow(row)

def create_parser():

    parser = ArgumentParser()
    parser.add_argument('--task', type=str, 
                        choices=['multilabel', 'multiclass', 'binaryclass', 'regression'])
    parser.add_argument('--structure_file', type=str)
    parser.add_argument('--dataset_file', type=str)
    parser.add_argument('--foldseek', type=str, default=None)
    parser.add_argument('--foldseek_process_id', type=int, default=0)
    parser.add_argument('--max_length', type=int, default=1024) 
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr_scheduler', action="store_true", default=True)
    parser.add_argument('--init_lr', type=float, default=0.001)
    parser.add_argument('--min_lr', type=float, default=0.0001)
    parser.add_argument('--gradient_accumulation_steps', type=int)
    parser.add_argument('--early_stopping', action="store_true", default=True)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--save_model', action="store_true", default=False)
    
    parser.add_argument('--plm_type', type=str, 
                        choices=['esm', 'bert', 'ankh', 'saprot', 't5'])
    parser.add_argument('--plm_dir', type=str)
    parser.add_argument('--plm_freeze', action="store_true", default=True)
    
    # gwr_plm
    parser.add_argument('--gwr_plm', action="store_true", default=True)
    parser.add_argument('--g_layers', type=int, default=3)
    parser.add_argument('--gh_dim', type=int, default=128)
    parser.add_argument('--go_dim', type=int, default=128)
    parser.add_argument('--rpa_threshold', type=float, default=0.2)
    parser.add_argument('--atten_map_layers', type=int, default=1)
    
    # GwR_Struc
    parser.add_argument('--gwr_struc', action="store_true", default=True)
    parser.add_argument('--dssp_token', action="store_true", default=True)
    parser.add_argument('--dssp_dim', type=int, default=10)
    parser.add_argument('--foldseek_seq', action="store_true", default=False)
    parser.add_argument('--foldseek_seq_dim', type=int, default=24)
    parser.add_argument('--top_k_neighbors', type=int, default=10)
    parser.add_argument('--node_hidden_dim_scalar', type=int, default=128)
    parser.add_argument('--node_hidden_dim_vector', type=int, default=128)
    parser.add_argument('--edge_hidden_dim_scalar', type=int, default=128)
    parser.add_argument('--edge_hidden_dim_vector', type=int, default=128)
    parser.add_argument('--num_encoder_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embed_gvp_output_dim', type=int, default=128)
    
    # layer_norm 
    parser.add_argument('--hidden_dim', type=int, default=512)
    
    # intergrate method
    parser.add_argument('--intergrate_method', type=str, choices=['concat', 'cross_atten'])
    parser.add_argument('--multihead_heads', type=int, default=4)
    parser.add_argument('--query', type=str, choices=['gwr_plm', 'gwr_struc'])
    
    # pooling head
    parser.add_argument('--linear_dropout', type=float, default=0.15)
    parser.add_argument('--attention_pooling', action="store_true", default=True)
    parser.add_argument('--num_labels', type=int, default=1)
    
    # tuning
    parser.add_argument('--tuning', action="store_true", default=False)
    parser.add_argument('--lora', action="store_true", default=False)
    parser.add_argument('--lora_alpha', type=int, default=1)
    parser.add_argument('--lora_r', type=int, default=4)
    parser.add_argument('--fft', action="store_true", default=False)
    
    # computing source log
    parser.add_argument('--record_source', action="store_true", default=False)
    parser.add_argument('--source_log', type=str)
     
    return parser.parse_args()
    

if __name__ == '__main__':
    
    mp.set_start_method("spawn")
    args = create_parser()

    for key, value in vars(args).items():
        print(f'{key}: {value}')
    model = GwR(args)

    data_module = DataModule(args)
    
    Trainer(model.to(args.device), args)(data_module)
  