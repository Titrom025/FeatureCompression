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
from typing import List, Dict, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import metric_utils, cv2, os
from PIL import Image
from torchvision import transforms as T
from utils import build_dino_text_embedding
import tensorflow as tf2
import tensorflow.compat.v1 as tf
import clip
from models_to_train import LinearCompressor, ShallowAE, DeepAE

# ----------------------------------------------------------------------------
#  OpenSeg utilities
# ----------------------------------------------------------------------------

def load_openseg_model(model_path: str):
    """Load OpenSeg model and configure GPU settings"""
    # Reset TensorFlow's GPU configuration
    tf.keras.backend.clear_session()
    
    # Get available GPUs
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.experimental.set_synchronous_execution(False)
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=1024 * 6)])
    
    return tf2.saved_model.load(model_path, tags=[tf.saved_model.tag_constants.SERVING])

def extract_openseg_features(model, img_path: str, text_embedding_tf: np.ndarray, device: torch.device) -> torch.Tensor:
    """Extract features from OpenSeg model"""
    with tf.io.gfile.GFile(img_path, 'rb') as f:
        np_image_string = np.array([f.read()])
    
    with torch.no_grad():
        output = model.signatures['serving_default'](
            inp_image_bytes=tf.convert_to_tensor(np_image_string[0]),
            inp_text_emb=text_embedding_tf)
    
    # Get segmentation mask and crop features
    image = output['image'].numpy()
    non_zero_mask = np.any(image != 0, axis=2)
    y_indices, x_indices = np.nonzero(non_zero_mask)
    min_y, max_y = y_indices.min(), y_indices.max()
    min_x, max_x = x_indices.min(), x_indices.max()
    
    # Get features and convert to torch tensor
    embedding_feat_square = output['ppixel_ave_feat'].numpy()
    embedding_feat = embedding_feat_square[:, min_y:max_y+1, min_x:max_x+1, :]
    embedding_feat = torch.from_numpy(embedding_feat).to(device)
    
    # Convert to BCHW format
    return embedding_feat.permute(0, 3, 1, 2)

def build_clip_text_embedding(categories):
    """Build CLIP text embeddings for given categories."""
    model, _ = clip.load("ViT-L/14@336px")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        all_text_embeddings = []
        print(f"Building text embeddings...")
        for category in tqdm(categories):
            texts = clip.tokenize(category).to(device)
            text_embeddings = model.encode_text(texts)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            text_embedding = text_embeddings.mean(dim=0)
            text_embedding /= text_embedding.norm()
            all_text_embeddings.append(text_embedding)

        all_text_embeddings = torch.stack(all_text_embeddings, dim=1)
    return all_text_embeddings.T.float()

# ----------------------------------------------------------------------------
#  Model loading and feature extraction
# ----------------------------------------------------------------------------

def load_model(method: str, ckpt_path: Path, d_in: int, d_out: int, device: torch.device) -> Tuple[Union[nn.Module, Tuple[torch.Tensor, torch.Tensor], torch.Tensor, None], str]:
    """Load model or projection matrix based on method"""
    if method == "raw":
        # No model, no compression
        return None, "raw"
    if method == "pca":
        npz = np.load(ckpt_path)
        mean = torch.from_numpy(npz["mean_"]).float().to(device)
        W = torch.from_numpy(npz["components_"]).float().to(device)  # (d,C)
        return (mean, W), "pca"
    elif method == "rand_proj":
        W = torch.from_numpy(np.load(ckpt_path)).float()
        W = W.t() if W.shape[0] == d_in else W  # ensure (d,C)
        W = W.to(device)
        return W, "rand_proj"
    else:
        if method in {"lin_proj", "sim_kd"}:
            model = LinearCompressor(d_in, d_out)
        elif method == "ae_shallow":
            model = ShallowAE(d_in, d_out)
        elif method == "ae_deep":
            model = DeepAE(d_in, d_out)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        model.to(device).eval().requires_grad_(False)
        return model, "learned"

def encode_features(x: torch.Tensor, model_or_W: Union[nn.Module, Tuple[torch.Tensor, torch.Tensor], torch.Tensor, None], method_type: str) -> torch.Tensor:
    """Encode features using the loaded model or projection matrix"""
    if method_type == "raw":
        z = x
    elif method_type == "pca":
        mean, W = model_or_W
        z = (x - mean) @ W.t()
    elif method_type == "rand_proj":
        W = model_or_W
        z = x @ W.t()
    else:  # learned
        model = model_or_W
        z = model.encode(x) if hasattr(model, "encode") else model(x)
    return F.normalize(z, dim=-1)

def encode_feature_map(fmap: torch.Tensor, model_or_W: Union[nn.Module, Tuple[torch.Tensor, torch.Tensor], torch.Tensor, None], method_type: str) -> torch.Tensor:
    """Encode feature map using the loaded model or projection matrix"""
    B, C, H, W = fmap.shape
    flat = fmap.permute(0, 2, 3, 1).reshape(-1, C)
    with torch.no_grad():
        z = encode_features(flat, model_or_W, method_type)
    return z.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

# ----------------------------------------------------------------------------
#  Filesystem helpers
# ----------------------------------------------------------------------------

_CKPT_FILES = {"pca": "pca.npz", "rand_proj": "rand.npy", "default": "model.pt"}

def _find_ckpt(path: Path, method: str = None) -> Path:
    if method == "raw":
        return None
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
    return 768 if backbone == "openseg" else 768

def eval_compressor(leaf: Union[Path, str], scenes_dir: Path, label_file: Path, dino_cfg: str, device, openseg_model=None, text_embedding_tf=None, n_frames_per_scene: int = None, frame_seed: int = 42) -> Dict:
    # If method_override is set, use it instead of inferring from leaf
    backbone, method, dim = leaf.parents[1].name, leaf.parents[0].name, int(leaf.name)
    
    # Load appropriate model based on backbone
    if backbone == "openseg":
        if openseg_model is None:
            raise ValueError("OpenSeg model must be provided for openseg backbone")
        extractor = lambda x: extract_openseg_features(openseg_model, x, text_embedding_tf, device)
    else:
        dino_model, extractor = _load_dino_backbone(dino_cfg, device)
    
    if method == "raw":
        model_or_W, method_type = load_model("raw", None, dim, dim, device)
    else:
        ckpt_path = _find_ckpt(leaf, method=method)
        model_or_W, method_type = load_model(method, ckpt_path, _infer_C(backbone), dim, device)

    # -------- load DINO model & text embeddings
    palette, labelset = metric_utils.get_text_requests("cocomap")
    label_map = metric_utils.read_label_mapping(str(label_file), label_to="cocomapid")
    
    if backbone == "openseg":
        # Use OpenSeg text embeddings
        text_orig = build_clip_text_embedding(labelset)
        if text_embedding_tf is None:
            text_embedding_tf = tf.reshape(text_orig.cpu().numpy(), [-1, 1, text_orig.shape[-1]])
            text_embedding_tf = tf.cast(text_embedding_tf, tf.float32)
    else:
        # Use DINO text embeddings
        text_orig = build_dino_text_embedding(labelset, dino_model, device).squeeze().to(device)
    
    text_emb = encode_features(text_orig, model_or_W, method_type).cpu()

    # Create output directory for predicted semantic images
    pred_dir = leaf / f"pred_semantics_frames{n_frames_per_scene}"
    pred_dir.mkdir(exist_ok=True, parents=True)

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
            
            if backbone == "openseg":
                feats = extractor(str(img_p))
            else:
                feats = extractor(img_t)
                
            feats_c = encode_feature_map(feats, model_or_W, method_type)
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
    
    csv_p = ckpt_root/f"summary_frames{args.n_frames_per_scene}.csv"
    if csv_p.exists():
        print(f"✓ Exists. CSV at {csv_p}")
        return

    # Initialize OpenSeg model and text embeddings once
    print("Initializing OpenSeg model...")
    openseg_model = load_openseg_model("./openseg_model")
    palette, labelset = metric_utils.get_text_requests("cocomap")
    text_orig = build_clip_text_embedding(labelset)
    text_embedding_tf = tf.reshape(text_orig.cpu().numpy(), [-1, 1, text_orig.shape[-1]])
    text_embedding_tf = tf.cast(text_embedding_tf, tf.float32)

    # Evaluate all discovered compressors
    for leaf in discover_leaves(ckpt_root):
        seg_file = leaf/f"seg_metrics_frames{args.n_frames_per_scene}.json"
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
                    openseg_model=openseg_model if leaf.parents[1].name == "openseg" else None,
                    text_embedding_tf=text_embedding_tf if leaf.parents[1].name == "openseg" else None,
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

    # Add "raw" method (no compression, original dim) for both dino and clip
    for backbone, dim in [("openseg", 768), ("dino", 768)]:
        print(f"Evaluating raw (no compression) method for {backbone}...")
        leaf = Path(f"{ckpt_root}/{backbone}/raw/{dim}")
        try:
            seg = eval_compressor(
                leaf,  # dummy, not a Path
                Path(args.scenes_dir),
                Path(args.label_file),
                args.dino_cfg,
                device,
                openseg_model=openseg_model if backbone == "openseg" else None,
                text_embedding_tf=text_embedding_tf if backbone == "openseg" else None,
                n_frames_per_scene=args.n_frames_per_scene,
                frame_seed=args.frame_seed
            )
            rows.append({
                "backbone": backbone,
                "method": "raw",
                "dim": dim,
                "overall_mIoU": seg["overall_mIoU"],
                "overall_mAcc": seg["overall_mAcc"]
            })
        except Exception as e:
            error_path = Path(args.scenes_dir).parent / f"seg_error_raw_{backbone}.txt"
            error_path.write_text(str(e))
            print(f"[ERROR] raw {backbone} {e}")
            raise e

    # summary CSV
    with open(csv_p, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f"✓ Done. CSV at {csv_p}")

if __name__ == "__main__":
    main()
