#!/usr/bin/env python3
"""
evaluate_compressors.py
=======================
Self‑contained evaluator: walks through a checkpoint tree created by
`train_compression.py`, loads every compressor (PCA, random projection, linear
projection, autoencoders), **without importing other local modules**, and
computes zero‑shot semantic‑segmentation scores (mIoU, mAcc) using the
Talk2DINO pipeline.  Results are saved into `<leaf>/seg_metrics.json`; a global
CSV `summary.csv` is written at the root.
"""
from __future__ import annotations

import argparse, csv, json, sys, random
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import metric_utils, cv2, os
from PIL import Image
from torchvision import transforms as T
from utils import build_dino_text_embedding
    
from models_to_train import LinearEncoder, ShallowAE, DeepAE
# ----------------------------------------------------------------------------
#  Minimal model stubs for learned compressors
# ----------------------------------------------------------------------------

def _build_model_stub(method: str, d_in: int, d_out: int) -> nn.Module:
    if method in {"lin_proj", "sim_kd"}: return LinearEncoder(d_in, d_out)
    if method == "ae_shallow": return ShallowAE(d_in, d_out)
    if method == "ae_deep": return DeepAE(d_in, d_out)
    raise ValueError(method)

# ----------------------------------------------------------------------------
#  Compressor class (PCA / RandProj / learned)
# ----------------------------------------------------------------------------

class Compressor:
    def __init__(self, method: str, ckpt_path: Path, d_in: int, d_out: int):
        self.method, self.d_in, self.d_out = method, d_in, d_out
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder, self.W, self.mean = None, None, None

        if method == "pca":
            npz = np.load(ckpt_path)
            self.mean = torch.from_numpy(npz["mean_"]).float().to(self.device)
            self.W = torch.from_numpy(npz["components_"]).float().to(self.device)  # (d,C)
        elif method == "rand_proj":
            W = torch.from_numpy(np.load(ckpt_path)).float()
            self.W = W.t() if W.shape[0] == d_in else W  # ensure (d,C)
            self.W = self.W.to(self.device)
        else:
            self.encoder = _build_model_stub(method, d_in, d_out).to(self.device)
            ckpt = torch.load(ckpt_path, map_location=self.device)
            self.encoder.load_state_dict(ckpt["model"], strict=False)
            self.encoder.eval().requires_grad_(False)

    # --------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:  # (N,C)->(N,d)
        if self.encoder is not None:
            z = self.encoder.encode(x) if hasattr(self.encoder, "encode") else self.encoder(x)
        elif self.method == "pca":
            z = (x - self.mean) @ self.W.t()
        else:  # rand_proj
            z = x @ self.W.t()
        return F.normalize(z, dim=-1)

    def encode_map(self, fmap: torch.Tensor) -> torch.Tensor:  # (B,C,H,W)->(B,d,H,W)
        B, C, H, W = fmap.shape
        flat = fmap.permute(0, 2, 3, 1).reshape(-1, C)
        with torch.no_grad():
            z = self.encode(flat)
        return z.reshape(B, H, W, self.d_out).permute(0, 3, 1, 2).contiguous()

# ----------------------------------------------------------------------------
#  Filesystem helpers
# ----------------------------------------------------------------------------

_CKPT_FILES = {"pca": "pca.npz", "rand_proj": "rand.npy", "default": "model.pt"}

def _find_ckpt(path: Path) -> Path:
    for f in _CKPT_FILES.values():
        fp = path / f
        if fp.exists():
            return fp
    raise FileNotFoundError(f"No checkpoint file in {path}")

def discover_leaves(root: Path) -> List[Path]:
    leaves = []
    for back in root.iterdir():
        if not back.is_dir():
            continue
        for method in back.iterdir():
            if not method.is_dir():
                continue
            for dim in method.iterdir():
                if any((dim / f).exists() for f in _CKPT_FILES.values()):
                    leaves.append(dim)
    return sorted(leaves)

# ----------------------------------------------------------------------------
#  DINO backbone utilities
# ----------------------------------------------------------------------------

def _load_dino_backbone(cfg_path: str, device):
    from omegaconf import OmegaConf
    sys.path.insert(0, "src/open_vocabulary_segmentation")
    from models import build_model
    cfg = OmegaConf.load(cfg_path)
    model = build_model(cfg.model).to(device).eval()

    from math import sqrt
    def extract(img_bchw):
        img_rgb = img_bchw[:, [2,1,0], :, :]
        prep = model.image_transforms(img_rgb).to(device)
        feats = model.model.forward_features(prep)
        patch = feats['x_norm_patchtokens']  # (B,N,C)
        g = int(sqrt(patch.shape[1]))
        patch = patch.reshape(img_bchw.shape[0], g, g, -1).permute(0,3,1,2)
        return F.normalize(patch, dim=1)
    return model, extract

# ----------------------------------------------------------------------------
#  Evaluation per compressor (Talk2DINO path)
# ----------------------------------------------------------------------------

def _infer_C(backbone: str) -> int:
    return 1024 if backbone == "clip" else 768

def eval_compressor(leaf: Path, scenes_dir: Path, label_file: Path, dino_cfg: str, device, n_frames_per_scene: int = None, frame_seed: int = 42) -> Dict:
    backbone, method, dim = leaf.parents[1].name, leaf.parents[0].name, int(leaf.name)
    ckpt_path = _find_ckpt(leaf)
    compressor = Compressor(method, ckpt_path, _infer_C(backbone), dim)

    # -------- load DINO model & text embeddings
    dino_model, extractor = _load_dino_backbone(dino_cfg, device)
    palette, labelset = metric_utils.get_text_requests("cocomap")
    label_map = metric_utils.read_label_mapping(str(label_file), label_to="cocomapid")
    text_orig = build_dino_text_embedding(labelset, dino_model, device).squeeze().to(device)
    text_emb = compressor.encode(text_orig).cpu()

    # Create output directory for predicted semantic images
    pred_dir = leaf / "pred_semantics"
    pred_dir.mkdir(exist_ok=True)

    scenes = [s for s in scenes_dir.iterdir() if (s/"color").exists()]
    per_scene = {}
    for scene in tqdm(scenes, desc=f"{backbone}-{method}-{dim}"):
        color_dir = scene/"color"
        imgs = sorted(p for p in color_dir.iterdir() if p.suffix.lower() in {'.jpg','.png'})
        if n_frames_per_scene is not None and n_frames_per_scene > 0 and len(imgs) > n_frames_per_scene:
            random.seed(frame_seed)
            imgs = random.sample(imgs, n_frames_per_scene)
            imgs = sorted(imgs, key=lambda x: x.name)  # sort for reproducibility
        conf = np.zeros((len(labelset)+1, len(labelset)), dtype=np.ulonglong)

        # Create subfolder for this scene's predictions
        scene_pred_dir = pred_dir / scene.name
        scene_pred_dir.mkdir(exist_ok=True)

        for img_p in tqdm(imgs, desc=f"{scene.name}"):
            img = Image.open(img_p).convert("RGB")
            img_t = T.ToTensor()(img).unsqueeze(0).to(device) * 255.0
            feats = extractor(img_t)
            feats_c = compressor.encode_map(feats)
            sim = torch.einsum('b c h w, n c -> b n h w', feats_c, text_emb.to(device))
            sim = F.interpolate(sim, img_t.shape[2:], mode='bilinear', align_corners=True)
            pred = torch.argmax(sim.squeeze(0), 0).cpu().numpy()
            label_img = metric_utils.get_mapped_label(img.height, img.width, str(img_p), label_map)
            if label_img is None:
                continue
            conf += metric_utils.confusion_matrix(pred.reshape(-1), label_img.reshape(-1), len(labelset))

            # Save predicted semantic image
            pred_img = Image.fromarray(pred.astype(np.uint8), mode="P")
            pred_img.putpalette(np.array(palette.cpu(), dtype=np.uint8).flatten())
            pred_img_path = scene_pred_dir / f"{img_p.stem}_pred.png"
            pred_img.save(pred_img_path)

        miou, macc = metric_utils.evaluate_confusion(scene.name, conf, stdout=False, dataset="cocomap")
        per_scene[scene.name] = {"mIoU": miou, "mAcc": macc}
    overall = {
        "overall_mIoU": float(np.mean([v['mIoU'] for v in per_scene.values()])),
        "overall_mAcc": float(np.mean([v['mAcc'] for v in per_scene.values()])),
        "scene_metrics": per_scene
    }
    return overall

# ----------------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser("Evaluate all compressors on Talk2DINO zero‑shot segmentation")
    p.add_argument("--ckpt_root", default="ckpts")
    p.add_argument("--scenes_dir", required=True)
    p.add_argument("--label_file", default="scannetv2-labels.modified.tsv")
    p.add_argument("--dino_cfg", default="talk2dino.yml")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n_frames_per_scene", type=int, default=None, help="Number of frames to evaluate per scene (randomly selected, deterministic)")
    p.add_argument("--frame_seed", type=int, default=42, help="Random seed for frame selection per scene")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt_root = Path(args.ckpt_root)
    rows = []

    for leaf in discover_leaves(ckpt_root):
        seg_file = leaf/"seg_metrics.json"
        if seg_file.exists():
            with open(seg_file) as fp: seg = json.load(fp)
        else:
            try:
                seg = eval_compressor(
                    leaf,
                    Path(args.scenes_dir),
                    Path(args.label_file),
                    args.dino_cfg,
                    device,
                    n_frames_per_scene=args.n_frames_per_scene,
                    frame_seed=args.frame_seed
                )
                with open(seg_file,"w") as fp: json.dump(seg, fp, indent=2)
            except Exception as e:
                (leaf/"seg_error.txt").write_text(str(e))
                print(f"[ERROR] {leaf} {e}")
                raise e
        rows.append({
            "backbone": leaf.parents[1].name,
            "method": leaf.parents[0].name,
            "dim": int(leaf.name),
            "overall_mIoU": seg["overall_mIoU"],
            "overall_mAcc": seg["overall_mAcc"]})

    # summary CSV
    csv_p = ckpt_root/"summary.csv"
    with open(csv_p, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f"✓ Done. CSV at {csv_p}")

if __name__ == "__main__":
    main()
