import os
import argparse
import random
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import sys
from utils import build_text_embedding
from omegaconf import OmegaConf
from math import sqrt
from torchvision import transforms
import torch.nn.functional as F
import cv2

# For DINO
sys.path.insert(0, "src/open_vocabulary_segmentation")
from models import build_model  # assumes models.py is available
# For CLIP (optional)
try:
    import clip
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False

from sklearn.decomposition import PCA
import torch.nn as nn
import metric_utils

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_image_features(model, img, device):
    # img: torch tensor (B, 3, H, W), values in [0,1] or [0,255]
    H, W = img.shape[2:]
    pH, pW = H, W
    # Preprocess image
    img_rgb = img[:, [2,1,0], :, :]  # BGR to RGB
    img_preprocessed = model.image_transforms(img_rgb).to(device)
    # Extract image features
    features = model.model.forward_features(img_preprocessed)
    image_feat = features['x_norm_patchtokens']
    b, np_, c = image_feat.shape
    np_h = np_w = int(sqrt(np_))
    image_feat = image_feat.reshape(b, np_h, np_w, c).permute(0, 3, 1, 2)
    norm = torch.norm(image_feat, p=2, dim=1, keepdim=True)
    image_feat /= (norm + 1e-7)
    return image_feat

def extract_dino_features(model, image_paths, device):
    from torchvision import transforms
    features = {}
    for img_path in tqdm(image_paths, desc="Extracting DINO features"):
        img = Image.open(img_path).convert("RGB")
        img_tensor = transforms.ToTensor()(img).unsqueeze(0) * 255.0  # (1, 3, H, W), BGR expected
        with torch.no_grad():
            image_feat = get_image_features(model, img_tensor, device)
            # Remove batch dim and permute to (H, W, C)
            patch_grid = image_feat[0].permute(1, 2, 0).cpu().numpy()
        features[img_path] = patch_grid
    return features

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

def fit_pca(all_feats, dim):
    pca = PCA(n_components=dim)
    pca.fit(all_feats)
    return pca

def compress_with_pca(pca, feats):
    H, W, D = feats.shape
    flat = feats.reshape(-1, D)
    proj = pca.transform(flat)
    return proj.reshape(H, W, -1)

def train_autoencoder(all_feats, latent_dim, device, epochs=20, lr=1e-3):
    input_dim = all_feats.shape[1]
    model = Autoencoder(input_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    data = torch.from_numpy(all_feats).float().to(device)
    for epoch in range(epochs):
        optimizer.zero_grad()
        recon, _ = model(data)
        loss = ((recon - data) ** 2).mean()
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0:
            print(f"AE Epoch {epoch}/{epochs}, MSE={loss.item():.4f}")
    return model

def generate_masks(image_feat, text_emb, target_dims):
    """Generate segmentation masks using interpolation.
    
    Args:
        image_feat: Feature map tensor (B, C, H, W)
        text_emb: Text embeddings tensor (num_classes, C)
        target_dims: Target dimensions (H, W)
    Returns:
        Interpolated mask tensor (B, num_classes, H, W)
    """
    B, C, H, W = image_feat.shape
    num_classes = text_emb.shape[0]
    
    # Compute similarity scores
    simmap = torch.einsum("b c h w, n c -> b n h w", image_feat, text_emb.float())
    
    # Interpolate to target dimensions
    mask = F.interpolate(simmap, target_dims, mode='bilinear', align_corners=True)
    
    return mask

def main(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_paths = [os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir)
                   if f.endswith((".jpg", ".png"))]
    print(f"Found {len(image_paths)} images in {args.image_dir}")

    # Only DINO for now
    cfg = OmegaConf.load(args.dino_cfg)
    dino_model = build_model(cfg.model)
    feats = extract_dino_features(dino_model, image_paths, device)

    # Prepare data for compression
    all_feats = np.concatenate([v.reshape(-1, v.shape[2]) for v in feats.values()], axis=0)
    print(f"Total feature vectors for compression: {all_feats.shape}")

    # Build compressors
    compressors = {}
    for method in args.methods:
        for dim in args.dims:
            key = f"{method}_{dim}"
            if method == 'pca':
                pca = fit_pca(all_feats, dim)
                compressors[key] = ('pca', pca)
                print(f"Fitted PCA for dim={dim}")
            elif method == 'ae':
                ae = train_autoencoder(all_feats, dim, device, epochs=args.ae_epochs)
                compressors[key] = ('ae', ae)
                print(f"Trained Autoencoder for dim={dim}")

    # Apply compression and save
    os.makedirs(args.output_dir, exist_ok=True)
    for img_path, feat_arr in feats.items():
        basename = os.path.splitext(os.path.basename(img_path))[0]
        for key, (method, comp) in compressors.items():
            if method == 'pca':
                compressed = compress_with_pca(comp, feat_arr)
            else:
                flat = torch.from_numpy(feat_arr.reshape(-1, feat_arr.shape[2])).float().to(device)
                with torch.no_grad(): _, z = comp(flat)
                compressed = z.cpu().numpy().reshape(feat_arr.shape[0], feat_arr.shape[1], -1)
            out_fname = f"{basename}_dino_{key}.npy"
            np.save(os.path.join(args.output_dir, out_fname), compressed)
            print(f"Saved {out_fname}")

    print(f"Feature extraction and compression complete. Results in {args.output_dir}")

def eval_mode(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scenes_dir = args.scenes_dir
    
    # Get all scene directories
    scene_dirs = [d for d in os.listdir(scenes_dir) if os.path.isdir(os.path.join(scenes_dir, d))]
    print(f"Found {len(scene_dirs)} scenes in {scenes_dir}")
    
    # Load model and setup once
    cfg = OmegaConf.load(args.dino_cfg)
    dino_model = build_model(cfg.model)
    dino_model.to(device).eval()
    
    # Get label mapping, palette, labelset
    label_mapping = metric_utils.read_label_mapping(args.label_file, label_to="cocomapid")
    palette, labelset = metric_utils.get_text_requests(args.dataset_type)
    
    # Build text embeddings once
    with torch.no_grad():
        text_emb = build_text_embedding(labelset, dino_model, device=device)
        text_emb = text_emb.squeeze(-1).squeeze(0).squeeze(0).to(device)
    
    # Initialize metrics storage
    all_scene_metrics = []
    
    # Process each scene
    for scene_name in tqdm(scene_dirs, desc="Processing scenes"):
        scene_dir = os.path.join(scenes_dir, scene_name)
        color_dir = os.path.join(scene_dir, "color")
        
        if not os.path.exists(color_dir):
            print(f"Skipping {scene_name} - no color directory found")
            continue
            
        # Setup scene-specific output directories
        scene_output_dir = os.path.join(args.output_dir, scene_name)
        color_gt_dir = os.path.join(scene_output_dir, 'color_gt')
        labelmap_gt_dir = os.path.join(scene_output_dir, 'labelmap_gt')
        labelmap_pred_dir = os.path.join(scene_output_dir, 'labelmap_pred')
        
        os.makedirs(color_gt_dir, exist_ok=True)
        os.makedirs(labelmap_gt_dir, exist_ok=True)
        os.makedirs(labelmap_pred_dir, exist_ok=True)
        
        # Get image files
        image_files = sorted([f for f in os.listdir(color_dir) if f.endswith(('.jpg', '.png'))])
        
        # Initialize confusion matrix for this scene
        confusion = np.zeros((len(labelset)+1, len(labelset)), dtype=np.ulonglong)
        
        # Process images in batches
        batch_size = 64
        for i in tqdm(range(0, len(image_files), batch_size), desc=f"Processing scene {scene_name}"):
            batch_files = image_files[i:i + batch_size]
            batch_images = []
            batch_labels = []
            
            # Load batch of images and labels
            for img_file in batch_files:
                img_path = os.path.join(color_dir, img_file)
                img = Image.open(img_path).convert("RGB")
                img_tensor = transforms.ToTensor()(img).unsqueeze(0) * 255.0
                batch_images.append(img_tensor)
                
                # Get GT label
                label_img = metric_utils.get_mapped_label(img_tensor.shape[2], img_tensor.shape[3], img_path, label_mapping)
                if label_img is not None:
                    batch_labels.append(label_img)
                else:
                    continue
            
            if not batch_images:
                continue
                
            # Stack batch
            batch_images = torch.cat(batch_images, dim=0).to(device)
            
            # Extract features for batch
            with torch.no_grad():
                image_feat = get_image_features(dino_model, batch_images, device)
                
                # Process each image in batch
                for idx, (img_file, label_img) in enumerate(zip(batch_files, batch_labels)):
                    if label_img is None:
                        continue
                        
                    # Get features for current image
                    feat = image_feat[idx:idx+1]
                    
                    # Generate masks and get predictions
                    mask = generate_masks(feat, text_emb, (batch_images.shape[2], batch_images.shape[3]))
                    pred_label = torch.argmax(mask.squeeze(0), dim=0)
                    
                    # Save predictions
                    colored_pred = metric_utils.render_palette(pred_label, palette).permute(1,2,0).cpu().numpy()
                    colored_pred = (colored_pred * 255).astype('uint8')
                    pred_path = os.path.join(labelmap_pred_dir, f"{os.path.splitext(img_file)[0]}.png")
                    cv2.imwrite(pred_path, cv2.cvtColor(colored_pred, cv2.COLOR_RGB2BGR))
                    
                    # Save GT
                    label_img_tensor = torch.from_numpy(label_img).long().to(device)
                    colored_gt = metric_utils.render_palette(label_img_tensor, palette).permute(1,2,0).cpu().numpy()
                    colored_gt = (colored_gt * 255).astype('uint8')
                    gt_label_path = os.path.join(labelmap_gt_dir, f"{os.path.splitext(img_file)[0]}.png")
                    cv2.imwrite(gt_label_path, cv2.cvtColor(colored_gt, cv2.COLOR_RGB2BGR))
                    
                    # Save color image
                    color_gt_path = os.path.join(color_gt_dir, img_file)
                    cv2.imwrite(color_gt_path, cv2.cvtColor(batch_images[idx].cpu().numpy().transpose(1,2,0).astype('uint8'), cv2.COLOR_RGB2BGR))
                    
                    # Update confusion matrix
                    confusion += metric_utils.confusion_matrix(
                        pred_label.cpu().numpy().reshape(-1), 
                        label_img.reshape(-1), 
                        len(labelset)
                    )
                    
                    # Clear cache periodically
                    if idx % 2 == 0:
                        torch.cuda.empty_cache()
        
        # Evaluate scene results
        mean_iou, mean_acc = metric_utils.evaluate_confusion(
            scene_name, 
            confusion, 
            stdout=True, 
            dataset=args.dataset_type
        )
        
        # Store metrics for this scene
        all_scene_metrics.append({
            'scene': scene_name,
            'mIoU': mean_iou,
            'mAcc': mean_acc
        })
        
        # Save scene metrics to file
        metrics_file = os.path.join(scene_output_dir, 'metrics.txt')
        with open(metrics_file, 'w') as f:
            f.write(f"Scene: {scene_name}\n")
            f.write(f"mIoU: {mean_iou:.4f}\n")
            f.write(f"mAcc: {mean_acc:.4f}\n")
    
    # Calculate and save overall metrics
    overall_miou = np.mean([m['mIoU'] for m in all_scene_metrics])
    overall_macc = np.mean([m['mAcc'] for m in all_scene_metrics])
    
    print("\nOverall Results:")
    print(f"Average mIoU across all scenes: {overall_miou:.4f}")
    print(f"Average mAcc across all scenes: {overall_macc:.4f}")
    
    # Save overall metrics
    overall_metrics_file = os.path.join(args.output_dir, 'overall_metrics.txt')
    with open(overall_metrics_file, 'w') as f:
        f.write("Overall Results:\n")
        f.write(f"Average mIoU across all scenes: {overall_miou:.4f}\n")
        f.write(f"Average mAcc across all scenes: {overall_macc:.4f}\n\n")
        f.write("Per-scene results:\n")
        for metrics in all_scene_metrics:
            f.write(f"{metrics['scene']}: mIoU={metrics['mIoU']:.4f}, mAcc={metrics['mAcc']:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["extract_compress", "eval"], default="eval", help="Mode: extract_compress or eval")
    parser.add_argument("--image_dir", type=str, default="/media/titrom/storage/mipt/datasets/ScanNet_12/val/scene0050_02/color/", help="Path to input images (for extract_compress)")
    parser.add_argument("--output_dir", type=str, default="results/", help="Directory to save results (for both modes)")
    parser.add_argument("--methods", type=str, default=["pca"], nargs='+', choices=["pca", "ae"], help="Compression methods to apply (for extract_compress)")
    parser.add_argument("--dims", type=int, nargs='+', default=[16], help="Target dimensions for compression (for extract_compress)")
    parser.add_argument("--ae_epochs", type=int, default=20, help="Epochs for autoencoder training (for extract_compress)")
    parser.add_argument("--scenes_dir", type=str, default="/media/titrom/storage/mipt/datasets/ScanNet_12/val/", help="Path to directory containing ScanNet scenes (for eval)")
    parser.add_argument("--label_file", type=str, default="scannetv2-labels.modified.tsv", help="Label mapping file (e.g. scannetv2-labels.modified.tsv) (for eval)")
    parser.add_argument("--dataset_type", type=str, default="cocomap", help="Dataset type (cocomap or scannet20) (for eval)")
    parser.add_argument("--dino_cfg", type=str, default="talk2dino.yml", help="Path to DINO model config")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.mode == "extract_compress":
        if not args.image_dir or not args.output_dir or not args.methods:
            print(f"Please provide --image_dir, --output_dir, and --methods for extract_compress mode.")
            exit(1)
        main(args)
    elif args.mode == "eval":
        if not args.scenes_dir or not args.label_file:
            print(f"Please provide --scenes_dir and --label_file for eval mode.")
            exit(1)
        eval_mode(args) 