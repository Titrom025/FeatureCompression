#!/usr/bin/env python3
"""
train_compression.py — learn low‑dimensional embeddings (16/32/64‑D)
for CLIP (1024‑D) and DINOv2 (768‑D) patch features.

❱❱ NEW — supports keyword **all** for `--backbone` and/or `--embed_dim`.
Runs every combination sequentially and stores results under:
    <out_dir>/<backbone>/<method>/<dim>/
Each run saves:
    • model.pt / components.npy / components.npz
    • metrics.json
"""
from __future__ import annotations

import argparse, json, math, os, random, shutil, traceback
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.decomposition import IncrementalPCA
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# -----------------------------------------------------------------------------
#  Repro
# -----------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# -----------------------------------------------------------------------------
#  Dataset streaming patch‑vectors from .npy
# -----------------------------------------------------------------------------

class FeaturePatchDataset(Dataset):
    """
    Загружает патч‑векторы из .npy независимо от порядка осей.
    Поддерживает (H, W, C)  и  (C, H, W).
    """
    def __init__(self, root: Path, backbone: str, k: int = 128):
        self.files = sorted(root.rglob(f"*_{backbone}.npy"))
        if not self.files:
            raise FileNotFoundError(f"No *_{backbone}.npy under {root}")

        # определяем формат по первому примеру
        sample = np.load(self.files[0], mmap_mode="r")
        if sample.shape[0] in {512, 768}:          # (C, H, W)
            self.channels_first = True
            self.C, self.H, self.W = sample.shape
        else:                                      # (H, W, C)
            self.channels_first = False
            self.H, self.W, self.C = sample.shape
        self.k = k

    def __len__(self):
        return len(self.files) * self.k

    def __getitem__(self, idx):
        img_idx, _ = divmod(idx, self.k)
        fmap = np.load(self.files[img_idx], mmap_mode="r")
        y = np.random.randint(0, self.H)
        x = np.random.randint(0, self.W)
        if self.channels_first:                    # (C,H,W) → (C)
            vec = fmap[:, y, x]
        else:                                      # (H,W,C) → (C)
            vec = fmap[y, x]
        return torch.from_numpy(vec.copy())

# -----------------------------------------------------------------------------
#  Models
# -----------------------------------------------------------------------------
from models_to_train import LinearCompressor, ShallowAE, DeepAE
BACKBONE_DIM = {"openseg": 768, "dino": 768}

# -----------------------------------------------------------------------------
#  Loss helpers
# -----------------------------------------------------------------------------

def cosine_distance(a,b):
    return 1-(F.normalize(a,dim=-1)*F.normalize(b,dim=-1)).sum(-1)

def reconstruction_loss(recon, target):
    """Combined MSE and cosine distance loss for reconstruction"""
    return F.mse_loss(recon, target) + 0.1 * cosine_distance(recon, target).mean()

def similarity_loss(compressed, original):
    """Similarity matrix matching loss for knowledge distillation"""
    t = F.normalize(original, dim=1)
    s = F.normalize(compressed, dim=1)
    return F.mse_loss(s@s.T, t@t.T)

def combined_loss(z, recon, target, alpha=0.1):
    """Combined loss focusing on similarity preservation with reconstruction as regularization"""
    sim_loss = similarity_loss(z, target)
    recon_loss = reconstruction_loss(recon, target)
    return sim_loss + alpha * recon_loss

# -----------------------------------------------------------------------------
#  Train / validate one epoch
# -----------------------------------------------------------------------------

def train_epoch(model, loader, optim, device, method):
    model.train()
    total = 0
    for batch in loader:
        batch = batch.to(device).float()
        
        if method in {"ae_shallow", "ae_deep", "lin_proj"}:
            z, recon = model(batch)
            # Focus on similarity preservation in compressed space
            loss = combined_loss(z, recon, batch)
        elif method == "sim_kd":
            z = model.get_compressed(batch)
            loss = similarity_loss(z, batch)
        else:
            raise ValueError(f"Unknown method: {method}")
            
        optim.zero_grad()
        loss.backward()
        optim.step()
        total += loss.item() * batch.size(0)
    return total / len(loader.dataset)

def validate(model, loader, device):
    model.eval()
    cds, mses, l2s, sim_dists = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device).float()
            z = model.get_compressed(batch)
            recon = model.get_reconstructed(batch)
            
            # Original space metrics
            cds.append(cosine_distance(recon, batch).cpu())
            mses.append(F.mse_loss(recon, batch, reduction='none').mean(-1).cpu())
            l2s.append(torch.norm(recon-batch, dim=-1).cpu())
            
            # Compressed space similarity
            t = F.normalize(batch, dim=1)
            s = F.normalize(z, dim=1)
            # Calculate similarity matrix for the batch
            sim_matrix = s @ s.T
            target_matrix = t @ t.T
            # Calculate MSE between similarity matrices
            sim_dist = F.mse_loss(sim_matrix, target_matrix, reduction='none').mean().cpu()
            sim_dists.append(sim_dist)
            
    cd_mean = torch.cat(cds).mean().item()
    mse_mean = torch.cat(mses).mean().item()
    l2_mean = torch.cat(l2s).mean().item()
    sim_dist_mean = torch.stack(sim_dists).mean().item()
    
    return {
        'cosine_dist': cd_mean, 
        'mse': mse_mean, 
        'l2_norm': l2_mean,
        'similarity_dist': sim_dist_mean
    }

# -----------------------------------------------------------------------------
#  PCA & random projection utilities
# -----------------------------------------------------------------------------

def train_pca(ds: FeaturePatchDataset, out_dim:int, save_path:Path):
    ipca = IncrementalPCA(n_components=out_dim, batch_size=16384)
    dl = DataLoader(ds, batch_size=16384, shuffle=True, num_workers=4)
    for b in tqdm(dl, desc="IPCA"): 
        ipca.partial_fit(b.numpy())
    np.savez(save_path, mean_=ipca.mean_, components_=ipca.components_)
    
    # Calculate explained variance ratio
    explained_var_ratio = ipca.explained_variance_ratio_.sum()
    return {'explained_variance_ratio': float(explained_var_ratio)}

def train_rand(in_dim:int, out_dim:int, save_path:Path):
    q,_ = torch.linalg.qr(torch.randn(in_dim,out_dim))
    np.save(save_path, q.numpy().astype(np.float32))
    return {}

# -----------------------------------------------------------------------------
#  Core routine for single combo
# -----------------------------------------------------------------------------

def run_single(features_dir:Path, backbone:str, method:str, dim:int, args):
    print(f"=== {backbone}|{method}|{dim} ===")
    out_dir = Path(args.out)/backbone/method/str(dim)
    if out_dir.exists():
        print(f"Skipping {out_dir} because it already exists")
        return
    out_dir.mkdir(parents=True,exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = FeaturePatchDataset(features_dir, backbone, k=args.patches_per_img)
    val_ds = FeaturePatchDataset(features_dir, backbone, k=32)

    in_dim_required = BACKBONE_DIM[backbone]
    if train_ds.C != in_dim_required:
        raise ValueError(
            f"Expected {in_dim_required}‑D features for {backbone}, "
            f"but .npy contains {train_ds.C}‑D. "
            "Пересчитайте признаки или проверьте extraction‑скрипт.")

    # Common metrics for all methods
    base_metrics = {
        'method': method,
        'backbone': backbone,
        'input_dim': train_ds.C,
        'output_dim': dim,
        'compression_ratio': train_ds.C / dim,
        'patches_per_img': args.patches_per_img,
        'total_samples': len(train_ds)
    }

    if method == "pca": 
        method_metrics = train_pca(train_ds, dim, out_dir/"pca.npz")
    elif method == "rand_proj": 
        method_metrics = train_rand(train_ds.C, dim, out_dir/"rand.npy")
    else:
        # Initialize model
        if method == "lin_proj": 
            model = LinearCompressor(train_ds.C, dim)
        elif method == "ae_shallow": 
            model = ShallowAE(train_ds.C, dim)
        elif method == "ae_deep": 
            model = DeepAE(train_ds.C, dim)
        elif method == "sim_kd": 
            model = LinearCompressor(train_ds.C, dim)
        else: 
            raise NotImplementedError
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        model.to(device)
        dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        vdl = DataLoader(val_ds, batch_size=4096, shuffle=False, num_workers=2)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        # Add learning rate scheduler with more aggressive settings
        scheduler = ReduceLROnPlateau(
            opt, mode='min', factor=0.2,  # More aggressive LR reduction
            patience=3,  # Shorter patience for LR reduction
            verbose=True, min_lr=1e-5,
            threshold=1e-4  # Smaller threshold for improvement
        )
        
        best_metrics = None
        best_score = float('inf')
        train_losses, val_metrics_history = [], []
        patience = args.patience
        patience_counter = 0
        min_delta = 1e-4  # Minimum change to be considered improvement
        
        # Create progress bar
        pbar = tqdm(range(1, args.max_epochs+1))
        
        for ep in pbar:
            tl = train_epoch(model, dl, opt, device, method)
            vm = validate(model, vdl, device)
            train_losses.append(tl)
            val_metrics_history.append(vm)
            
            # Update learning rate based on validation loss
            scheduler.step(vm['cosine_dist'])
            
            # Update progress bar description
            pbar.set_description(
                f"train={tl:.4f} val_cos={vm['cosine_dist']:.4f} "
                f"val_mse={vm['mse']:.4f} val_l2={vm['l2_norm']:.4f} "
                f"lr={opt.param_groups[0]['lr']:.2e} patience={patience_counter}/{patience}"
            )
            
            # Check for improvement with minimum delta
            if vm['cosine_dist'] < best_score - min_delta: 
                best_score = vm['cosine_dist']
                best_metrics = vm.copy()
                best_metrics['epoch'] = ep
                torch.save({'model': model.state_dict(), 'epoch': ep}, out_dir/"model.pt")
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {ep}")
                break
        
        method_metrics = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'epochs_trained': ep,
            'learning_rate': args.lr,
            'final_learning_rate': opt.param_groups[0]['lr'],
            'batch_size': args.batch_size,
            'best_epoch': best_metrics['epoch'],
            'best_val_cosine_dist': best_metrics['cosine_dist'],
            'best_val_mse': best_metrics['mse'],
            'best_val_l2_norm': best_metrics['l2_norm'],
            'final_train_loss': train_losses[-1],
            'final_val_cosine_dist': val_metrics_history[-1]['cosine_dist'],
            'final_val_mse': val_metrics_history[-1]['mse'],
            'final_val_l2_norm': val_metrics_history[-1]['l2_norm']
        }
    
    # Combine all metrics
    metrics = {**base_metrics, **method_metrics}
    with open(out_dir/"metrics.json", "w") as fp: 
        json.dump(metrics, fp, indent=2)

# -----------------------------------------------------------------------------
#  Entry point with all‑loop support
# -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features_dir", required=True)
    p.add_argument("--out", required=True, help="Root output directory")
    p.add_argument("--backbone", choices=["openseg","dino","all"], required=True)
    p.add_argument("--method", choices=["pca","rand_proj","lin_proj","ae_shallow","ae_deep","sim_kd", "all"], required=True)
    p.add_argument("--embed_dim", choices=["16","32","64","all"], required=True)
    p.add_argument("--patches_per_img", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=8192)
    p.add_argument("--max_epochs", type=int, default=200, help="Maximum number of epochs")
    p.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    p.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    feats_dir = Path(args.features_dir)

    bk_list = ["openseg","dino"] if args.backbone=="all" else [args.backbone]
    dim_list = [16,32,64] if args.embed_dim=="all" else [int(args.embed_dim)]
    method_list = ["pca","rand_proj","lin_proj","ae_shallow","ae_deep","sim_kd"] if args.method=="all" else [args.method]

    for bk in bk_list:
        for method in method_list:
            for dim in dim_list:
                try:
                    run_single(feats_dir, bk, method, dim, args)
                except Exception as e:
                    out_dir = Path(args.out)/bk/method/str(dim)
                    with open(out_dir/"error.txt", "w") as f:
                        f.write(traceback.format_exc())
                    print(f"Error running {bk}|{method}|{dim}: {traceback.format_exc()}")
                    raise e

if __name__=="__main__":
    main()
