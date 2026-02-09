# -*- coding: utf-8 -*-
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import shutil
import argparse
import configparser
from model.AmTGBiM import make_AmTGBiM
from lib.utils import get_adjacency_matrix, compute_val_loss_mgcn, predict_and_save_results_mgcn
from tensorboardX import SummaryWriter
from lib.metrics import masked_mape_np, masked_mae, masked_mse, masked_rmse
from data_provider.data_factory import data_provider
import random
from lib.utils import EarlyStopping, adjust_learning_rate
import json
import pandas as pd
from datetime import datetime
import psutil
import gc
import warnings

warnings.filterwarnings('ignore')


def parse_arguments():
    parser = argparse.ArgumentParser(description='AmTGBiM Training Script')
    parser.add_argument("--config", default='configurations/PEMS08_mgcn.conf', type=str,
                        help="configuration file path")
    parser.add_argument("--dataset", default=None, type=str,
                        choices=['PEMS03', 'PEMS04', 'PEMS07', 'PEMS08'],
                        help="dataset name (overrides config)")
    parser.add_argument("--save_results", action='store_true',
                        help="save experiment results")
    parser.add_argument("--test_only", action='store_true',
                        help="only run testing on pre-trained model")
    parser.add_argument("--model_path", default=None, type=str,
                        help="path to pre-trained model for testing")

    # Parameter tuning options
    parser.add_argument("--k_hop", default=None, type=int, choices=[2, 3, 4],
                        help="k-hop neighborhood range")
    parser.add_argument("--mask_threshold", default=None, type=float,
                        help="adaptive mask threshold")
    parser.add_argument("--feature_weight", default=None, type=float,
                        help="feature mask weight")
    parser.add_argument("--enhanced_pe", default=None, type=str, choices=['true', 'false'],
                        help="use enhanced positional encoding")
    parser.add_argument("--fusion_gate_dropout", default=None, type=float,
                        help="fusion gate dropout rate")
    parser.add_argument("--mamba_d_state_override", default=None, type=int,
                        help="override mamba d_state")
    parser.add_argument("--experiment_name", default=None, type=str,
                        help="custom experiment name suffix")

    return parser.parse_args()


def load_config(args):
    if args.dataset is not None:
        config_file = f'configurations/{args.dataset}_mgcn.conf'
        if os.path.exists(config_file):
            args.config = config_file
        else:
            raise FileNotFoundError(f"Config file not found: {config_file}")

    config = configparser.ConfigParser()
    config.read(args.config)

    data_cfg = config['Data']
    train_cfg = config['Training']

    adj_filename = data_cfg['adj_filename']
    graph_signal_matrix_filename = data_cfg['graph_signal_matrix_filename']
    id_filename = data_cfg.get('id_filename', None)

    global num_of_vertices, points_per_hour, num_for_predict, len_input, dataset_name
    num_of_vertices = int(data_cfg['num_of_vertices'])
    points_per_hour = int(data_cfg['points_per_hour'])
    num_for_predict = int(data_cfg['num_for_predict'])
    len_input = int(data_cfg['len_input'])
    dataset_name = data_cfg['dataset_name']

    global in_channels, K, nb_chev_filter, nb_time_filter, learning_rate, epochs, batch_size
    in_channels = int(train_cfg['in_channels'])
    K = int(train_cfg['K'])
    nb_chev_filter = int(train_cfg['nb_chev_filter'])
    nb_time_filter = int(train_cfg['nb_time_filter'])
    learning_rate = float(train_cfg['learning_rate'])
    epochs = int(train_cfg.get('epochs', 50))
    batch_size = int(train_cfg['batch_size'])

    global attention_heads, mamba_d_state, mamba_d_conv, mamba_expand
    attention_heads = int(train_cfg.get('attention_heads', 4))
    mamba_d_state = train_cfg.get('mamba_d_state', 'auto')
    if mamba_d_state.lower() != 'auto':
        mamba_d_state = int(mamba_d_state)
    mamba_d_conv = int(train_cfg.get('mamba_d_conv', 2))
    mamba_expand = int(train_cfg.get('mamba_expand', 1))

    global k_hop, mask_threshold, feature_weight, enhanced_pe, fusion_gate_dropout
    k_hop = int(train_cfg.get('k_hop', 2))
    mask_threshold = float(train_cfg.get('mask_threshold', 0.5))
    feature_weight = float(train_cfg.get('feature_weight', 0.5))
    enhanced_pe = train_cfg.get('enhanced_pe', 'true').lower() == 'true'
    fusion_gate_dropout = float(train_cfg.get('fusion_gate_dropout', 0.05))

    # Command-line overrides
    if args.k_hop is not None:
        k_hop = args.k_hop
    if args.mask_threshold is not None:
        mask_threshold = args.mask_threshold
    if args.feature_weight is not None:
        feature_weight = args.feature_weight
    if args.enhanced_pe is not None:
        enhanced_pe = args.enhanced_pe.lower() == 'true'
    if args.fusion_gate_dropout is not None:
        fusion_gate_dropout = args.fusion_gate_dropout
    if args.mamba_d_state_override is not None:
        mamba_d_state = args.mamba_d_state_override

    return adj_filename, graph_signal_matrix_filename, id_filename


def get_experiment_dir():
    if args.experiment_name:
        suffix = f"_{args.experiment_name}"
    else:
        suffix = f"_k{k_hop}_t{mask_threshold:.1f}_w{feature_weight:.1f}_pe{int(enhanced_pe)}_gd{fusion_gate_dropout:.2f}"

    folder = f'predict{num_for_predict}_AmTGBiM_h{attention_heads}_s{mamba_d_state}_c{mamba_d_conv}_e{mamba_expand}{suffix}'
    return os.path.join('experiments', dataset_name, folder)


def get_memory_usage():
    mem = {}
    if torch.cuda.is_available():
        mem['gpu_allocated'] = torch.cuda.memory_allocated() / 1e9
        mem['gpu_cached'] = torch.cuda.memory_reserved() / 1e9
        mem['gpu_total'] = torch.cuda.get_device_properties(0).total_memory / 1e9
    sys_mem = psutil.virtual_memory()
    mem['system_used'] = sys_mem.used / 1e9
    mem['system_total'] = sys_mem.total / 1e9
    mem['system_percent'] = sys_mem.percent
    return mem


def get_data_loaders(root_path, batch_size):
    train_set, train_loader = data_provider(root_path, 'train', points_per_hour, 0, num_for_predict, batch_size)
    val_set, val_loader = data_provider(root_path, 'val', points_per_hour, 0, num_for_predict, batch_size)
    test_set, test_loader = data_provider(root_path, 'test', points_per_hour, 0, num_for_predict, batch_size)
    return train_loader, val_loader, test_loader, train_set, val_set, test_set


def create_model(adj_mx):
    if nb_chev_filter % attention_heads != 0:
        adjusted = ((nb_chev_filter + attention_heads - 1) // attention_heads) * attention_heads
        print(f"Adjusting nb_chev_filter: {nb_chev_filter} -> {adjusted}")
        nb_chev_filter_adjusted = adjusted
    else:
        nb_chev_filter_adjusted = nb_chev_filter

    model = make_AmTGBiM(
        DEVICE=DEVICE,
        in_channels=in_channels,
        K=K,
        nb_chev_filter=nb_chev_filter_adjusted,
        nb_time_filter=nb_time_filter,
        time_strides=1,
        adj_mx=adj_mx,
        num_for_predict=num_for_predict,
        len_input=len_input,
        attention_heads=attention_heads,
        mamba_d_state=mamba_d_state,
        mamba_d_conv=mamba_d_conv,
        mamba_expand=mamba_expand,
        k_hop=k_hop,
        mask_threshold=mask_threshold,
        feature_weight=feature_weight,
        enhanced_pe=enhanced_pe,
        fusion_gate_dropout=fusion_gate_dropout
    )
    return model


def set_seed(seed=2024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    args = parse_arguments()
    set_seed(2024)

    adj_filename, graph_signal_matrix_filename, id_filename = load_config(args)
    params_path = get_experiment_dir()

    adj_mx, _ = get_adjacency_matrix(adj_filename, num_of_vertices, id_filename)

    train_loader, val_loader, test_loader, train_data, val_data, test_data = get_data_loaders(
        graph_signal_matrix_filename, batch_size
    )

    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    if args.test_only:
        if not args.model_path or not os.path.exists(args.model_path):
            print("Test mode requires a valid model_path")
            exit(1)
        print(f"Test mode: loading model from {args.model_path}")
        model = create_model(adj_mx)
        model.load_state_dict(torch.load(args.model_path))
        predict_and_save_results_mgcn(model, test_loader, test_data, num_for_predict, 'MAE_RMSE_MAPE', params_path, 'test')
        exit(0)

    # Training mode
    model = create_model(adj_mx)
    model.to(DEVICE)

    criterion = nn.L1Loss().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate * 0.8, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30, 40], gamma=0.7)
    early_stopping = EarlyStopping(patience=12)

    writer = SummaryWriter(logdir=params_path)

    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.float().to(DEVICE)
            targets = targets.float().to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs, batch_idx=batch_idx, total_batches=len(train_loader), epoch=epoch)

            loss = criterion(outputs, targets)
            l1_reg = model.get_l1_regularization()
            total_loss = loss + 0.05 * l1_reg

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        val_loss = compute_val_loss_silent(model, val_loader, criterion, False, 0.0, DEVICE)
        print(f"Epoch {epoch+1:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")

        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(params_path, 'best_model.pth'))

        early_stopping(val_loss, model, os.path.join(params_path, f'epoch_{epoch+1}.pth'))
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

        torch.cuda.empty_cache()
        gc.collect()

    writer.close()

    print(f"Training finished. Best epoch: {best_epoch+1}, Best val loss: {best_val_loss:.6f}")

    # Final test
    model.load_state_dict(torch.load(os.path.join(params_path, 'best_model.pth')))
    predict_and_save_results_mgcn(model, test_loader, test_data, num_for_predict, 'MAE_RMSE_MAPE', params_path, 'test')