#!/usr/bin/env python3
"""
feature_extraction.py – Stage‑1 script to pre‑compute and store CLIP/DINOv2
patch‑level features for Replica/ScanNet‑style datasets.

For each RGB image it produces a H×W×C NumPy array (float32, L2‑normalised
along C) and saves it next to the image under <output_dir>/<scene>/<backbone>/.
The script deliberately avoids any evaluation logic – it only harvests the raw
features that later compression models will consume.
"""

import argparse
import os
import sys
import math
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms as T
import clip  # OpenAI CLIP
from omegaconf import OmegaConf
import cv2
import torch.nn.functional as F

# TensorFlow imports for OpenSeg
import tensorflow as tf2
import tensorflow.compat.v1 as tf

# ----------------------------------------------------------------------------
#  Utility helpers
# ----------------------------------------------------------------------------

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------
#  CLIP helpers – patch feature extraction from ViT‑L/14@336
# ----------------------------------------------------------------------------

class CLIPPatchExtractor:
    """Wraps a CLIP vision tower and exposes patch‑level feature maps."""

    def __init__(self, device: torch.device):
        self.model, self.preprocess = clip.load("ViT-L/14@336px", device=device)
        self.model.float()  # Ensure model is in float32
        self.model.eval().requires_grad_(False)
        # CLIP's own preprocessing (resize‑>center crop‑>norm)
        # Channel‑first, float32 0‑1 tensor expected.

    @torch.no_grad()
    def __call__(self, img_bchw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_bchw: RGB uint8/float32 tensor in [0,1] range.
        Returns:
            Tensor – (B, C=1024, H_p, W_p) L2‑normalised along C.
        """
        device = next(self.model.parameters()).device
        img = (img_bchw * 255.0 if img_bchw.max() <= 1.0 else img_bchw)
        img = img.to(torch.uint8).cpu()
        img_list = [T.ToPILImage()(x) for x in img]  # CLIP expects PIL inputs
        img_proc = torch.stack([self.preprocess(im) for im in img_list]).to(device)

        # Forward through visual transformer, capturing patch tokens
        visual = self.model.visual
        x = visual.conv1(img_proc)  # (B, width, grid, grid)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # (B, N, C)
        cls_emb = visual.class_embedding.to(x.dtype)
        cls_emb = cls_emb.expand(x.shape[0], 1, -1)
        x = torch.cat([cls_emb, x], dim=1)  # prepend CLS
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)  # (seq, B, C)
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)
        patch_tokens = x[:, 1:, :]  # drop CLS

        n_patches = patch_tokens.shape[1]
        grid = int(math.sqrt(n_patches))
        assert grid * grid == n_patches, "Image not square‑resized as expected."
        patch_tokens = patch_tokens.reshape(img_bchw.shape[0], grid, grid, -1)
        patch_tokens = patch_tokens.permute(0, 3, 1, 2).contiguous()  # B C H W
        patch_tokens = torch.nn.functional.normalize(patch_tokens, dim=1)
        return patch_tokens  # (B, 1024, H_p, W_p)


# ----------------------------------------------------------------------------
#  OpenSeg helpers – CLIP features via OpenSeg model
# ----------------------------------------------------------------------------

class OpenSegPatchExtractor:
    """Wraps OpenSeg model for CLIP-based patch feature extraction."""

    def __init__(self, model_path: str):
        # Setup TensorFlow GPU configuration
        gpus = tf2.config.experimental.list_physical_devices('GPU')
        if gpus:
            tf2.config.experimental.set_memory_growth(gpus[0], True)
            tf2.config.experimental.set_synchronous_execution(False)
            tf2.config.set_logical_device_configuration(
                gpus[0],
                [tf2.config.LogicalDeviceConfiguration(memory_limit=1024 * 10)])
        
        # Load OpenSeg model
        self.model = tf2.saved_model.load(model_path, tags=[tf.saved_model.tag_constants.SERVING])
        print(f"Loaded OpenSeg model from {model_path}")

    def __call__(self, img_path: Path) -> np.ndarray:
        """
        Args:
            img_path: Path to image file
        Returns:
            ndarray – (H_p, W_p, C=768) L2‑normalised features
        """
        # Read image as bytes (OpenSeg expects byte input)
        with tf2.io.gfile.GFile(str(img_path), 'rb') as f:
            np_image_string = np.array([f.read()])
        
        # Create dummy text embedding (we only want image features)
        dummy_text_emb = tf.zeros([1, 1, 768], dtype=tf.float32)
        
        # Run OpenSeg inference
        with torch.no_grad():
            output = self.model.signatures['serving_default'](
                inp_image_bytes=tf.convert_to_tensor(np_image_string[0]),
                inp_text_emb=dummy_text_emb)
        
        # Extract pixel-level features
        embedding_feat_square = output['ppixel_ave_feat'].numpy()  # (1, H, W, C)
        
        # Find non-zero region (OpenSeg pads with zeros)
        image_output = output['image'].numpy()
        non_zero_mask = np.any(image_output != 0, axis=2)
        if np.any(non_zero_mask):
            y_indices, x_indices = np.nonzero(non_zero_mask)
            min_y, max_y = y_indices.min(), y_indices.max()
            min_x, max_x = x_indices.min(), x_indices.max()
            
            # Crop to non-zero region
            embedding_feat = embedding_feat_square[0, min_y:max_y+1, min_x:max_x+1, :]
        else:
            # Fallback if no non-zero region found
            embedding_feat = embedding_feat_square[0]
        
        # L2 normalize along feature dimension
        norm = np.linalg.norm(embedding_feat, axis=-1, keepdims=True)
        embedding_feat = embedding_feat / (norm + 1e-7)
        
        return embedding_feat.astype(np.float32)  # (H_p, W_p, C=768)


# ----------------------------------------------------------------------------
#  DINOv2 helpers – reuse Talk2DINO pipeline from previous script
# ----------------------------------------------------------------------------

sys.path.insert(0, "src/open_vocabulary_segmentation")
from models import build_model  # noqa: E402, pylint: disable=wrong-import-position


def build_dino_feature_extractor(cfg_path: str, device: torch.device):
    cfg = OmegaConf.load(cfg_path)
    model = build_model(cfg.model)
    model.to(device).eval().requires_grad_(False)

    def _extract(img_bchw: torch.Tensor):
        with torch.no_grad():
            # Convert from RGB float [0,1] to RGB float [0,255] as expected by model.image_transforms
            img_rgb = (img_bchw * 255.0).to(torch.uint8).float()
            img_preprocessed = model.image_transforms(img_rgb).to(device)
            feats = model.model.forward_features(img_preprocessed)
            patch = feats["x_norm_patchtokens"]  # (B, N, C)
            n_patches = patch.shape[1]
            grid = int(math.sqrt(n_patches))
            patch = patch.reshape(img_bchw.shape[0], grid, grid, -1).permute(0, 3, 1, 2)
            patch = torch.nn.functional.normalize(patch, dim=1)
            return patch  # (B, 768, H_p, W_p)

    return _extract


# ----------------------------------------------------------------------------
#  I/O utils
# ----------------------------------------------------------------------------

def find_images(scene_dir: Path) -> List[Path]:
    return sorted([p for p in scene_dir.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])


def save_feature(feat: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, feat.astype(np.float32))


# ----------------------------------------------------------------------------
#  Main routine
# ----------------------------------------------------------------------------

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build extractors as needed
    extractors: Dict[str, callable] = {}
    # if args.backbone in {"clip", "both"}:
    #     extractors["clip"] = CLIPPatchExtractor(device)
    if args.backbone in {"openseg", "both"}:
        extractors["openseg"] = OpenSegPatchExtractor(args.openseg_model_path)
    if args.backbone in {"dino", "both"}:
        extractors["dino"] = build_dino_feature_extractor(args.dino_cfg, device)

    scenes_root = Path(args.scenes_dir)
    scene_dirs = [p for p in scenes_root.iterdir() if p.is_dir()]
    print(f"Found {len(scene_dirs)} scenes under {scenes_root}.")

    for scene_path in tqdm(scene_dirs, desc="Scenes"):
        image_files = find_images(scene_path / "color") if (scene_path / "color").exists() else find_images(scene_path)
        if not image_files:
            print(f"[!] No images in {scene_path}")
            continue

        # Limit number of images per scene if specified
        if args.max_images_per_scene > 0:
            image_files = image_files[:args.max_images_per_scene]
            print(f"Processing {len(image_files)} images from {scene_path.name} (limited by --max-images-per-scene)")

        for img_path in tqdm(image_files, desc=f"{scene_path.name}", leave=False):
            for name, extractor in extractors.items():
                if name == "openseg":
                    # OpenSeg extractor takes image path directly
                    feat_full = extractor(img_path)  # (H_p, W_p, C)
                    H_p, W_p, C = feat_full.shape
                    # Randomly select 37 unique rows and 37 unique columns
                    rows = np.random.choice(H_p, 37, replace=False)
                    cols = np.random.choice(W_p, 37, replace=False)
                    rows.sort()
                    cols.sort()
                    feat = feat_full[np.ix_(rows, cols, np.arange(C))]  # (37, 37, C)
                else:
                    # Other extractors take tensor input
                    img = Image.open(img_path).convert("RGB")
                    tensor = T.ToTensor()(img).unsqueeze(0).to(device)  # (1,3,H,W) float 0‑1
                    feat = extractor(tensor)  # (1,C,Hp,Wp)
                    feat = feat.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H_p, W_p, C)

                out_rel = img_path.relative_to(scenes_root)
                out_file = Path(args.output_dir) / name / out_rel.parent / (out_rel.stem + f"_{name}.npy")
                save_feature(feat, out_file)
                
                del feat

            torch.cuda.empty_cache()


# ----------------------------------------------------------------------------
#  CLI
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract patch‑level CLIP & DINOv2 features for subsequent compression experiments.")
    parser.add_argument("--scenes_dir", required=True, help="Root folder with scene sub‑directories (e.g. Replica/ScanNet).")
    parser.add_argument("--output_dir", required=True, help="Folder where .npy feature files will be written.")
    parser.add_argument("--backbone", default="both", choices=["clip", "openseg", "dino", "both"], help="Which backbone(s) to process.")
    parser.add_argument("--dino_cfg", default="talk2dino.yml", help="Path to Talk2DINO YAML or URL.")
    parser.add_argument("--openseg_model_path", default="./openseg_model", help="Path to OpenSeg model directory.")
    parser.add_argument("--max_images_per_scene", type=int, default=0, help="Maximum number of images to process per scene (0 = no limit).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with torch.no_grad():
        main(args)
