import os
import argparse
import random
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import sys
from utils import build_dino_text_embedding
from omegaconf import OmegaConf
from math import sqrt
from torchvision import transforms
import torch.nn.functional as F
import cv2
import tensorflow as tf2
import tensorflow.compat.v1 as tf

# For DINO
sys.path.insert(0, "src/open_vocabulary_segmentation")
from models import build_model  # assumes models.py is available

import clip

import metric_utils

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
    return all_text_embeddings.cpu().numpy().T.astype(np.float32)

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

def eval_talk2dino(args):
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
        text_emb = build_dino_text_embedding(labelset, dino_model, device)
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

def eval_openseg(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scenes_dir = args.scenes_dir
    
    # Get all scene directories
    scene_dirs = [d for d in os.listdir(scenes_dir) if os.path.isdir(os.path.join(scenes_dir, d))]
    print(f"Found {len(scene_dirs)} scenes in {scenes_dir}")
    
    # Load OpenSeg model
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.experimental.set_synchronous_execution(False)
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=1024 * 6)])
    
    openseg_model = tf2.saved_model.load(args.model_path, tags=[tf.saved_model.tag_constants.SERVING])
    
    # Get label mapping, palette, labelset
    label_mapping = metric_utils.read_label_mapping(args.label_file, label_to="cocomapid")
    palette, labelset = metric_utils.get_text_requests(args.dataset_type)
    
    # Build text embeddings once
    text_embedding = build_clip_text_embedding(labelset)
    text_embedding_tf = tf.reshape(text_embedding, [-1, 1, text_embedding.shape[-1]])
    text_embedding_tf = tf.cast(text_embedding_tf, tf.float32)
    
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
        
        # Process images
        for img_file in tqdm(image_files, desc=f"Processing scene {scene_name}"):
            # if "20" in img_file:
            #     break
            img_path = os.path.join(color_dir, img_file)
            
            img_height, img_width = cv2.imread(img_path).shape[:2]
            # Get GT label
            label_img = metric_utils.get_mapped_label(img_height, img_width, img_path, label_mapping)
            if label_img is None:
                continue
            
            # Process image with OpenSeg
            with tf.io.gfile.GFile(img_path, 'rb') as f:
                np_image_string = np.array([f.read()])
            
            with torch.no_grad():
                output = openseg_model.signatures['serving_default'](
                    inp_image_bytes=tf.convert_to_tensor(np_image_string[0]),
                    inp_text_emb=text_embedding_tf)
            
            # Get segmentation mask
            image = output['image'].numpy()
            non_zero_mask = np.any(image != 0, axis=2)
            y_indices, x_indices = np.nonzero(non_zero_mask)
            min_y, max_y = y_indices.min(), y_indices.max()
            min_x, max_x = x_indices.min(), x_indices.max()
            
            # Resize mask to original image size
            source_image = cv2.imread(img_path)
            target_h, target_w = source_image.shape[:2]
            
            embedding_feat_square = output['ppixel_ave_feat'].numpy()
            embedding_feat = embedding_feat_square[:, min_y:max_y+1, min_x:max_x+1, :]
            embedding_feat = torch.from_numpy(embedding_feat).to(device)
            
            text_embedding_tensor = torch.from_numpy(text_embedding).to(device)
            # embedding_feat: (1, H, W, C) -> (1, C, H, W)
            embedding_feat_bchw = embedding_feat.permute(0, 3, 1, 2)
            mask = generate_masks(embedding_feat_bchw, text_embedding_tensor, (target_h, target_w))
            pred_label = torch.argmax(mask.squeeze(0), dim=0).cpu()

            # Save predictions
            colored_pred = metric_utils.render_palette(pred_label, palette).permute(1,2,0).cpu().numpy()
            colored_pred = (colored_pred * 255).astype('uint8')
            pred_path = os.path.join(labelmap_pred_dir, f"{os.path.splitext(img_file)[0]}.png")
            cv2.imwrite(pred_path, cv2.cvtColor(colored_pred, cv2.COLOR_RGB2BGR))
            
            # Save GT
            label_img_tensor = torch.from_numpy(label_img).long()
            colored_gt = metric_utils.render_palette(label_img_tensor, palette).permute(1,2,0).cpu().numpy()
            colored_gt = (colored_gt * 255).astype('uint8')
            gt_label_path = os.path.join(labelmap_gt_dir, f"{os.path.splitext(img_file)[0]}.png")
            cv2.imwrite(gt_label_path, cv2.cvtColor(colored_gt, cv2.COLOR_RGB2BGR))
            
            # Save color image
            color_gt_path = os.path.join(color_gt_dir, img_file)
            cv2.imwrite(color_gt_path, source_image)
            
            # Update confusion matrix
            confusion += metric_utils.confusion_matrix(
                pred_label.reshape(-1), 
                label_img.reshape(-1), 
                len(labelset)
            )
            
            # Clear cache periodically
            if torch.cuda.is_available():
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
    parser.add_argument("--model", type=str, choices=["dino", "openseg"], default="dino", help="Model: extract_compress, dino, clip, or openseg")
    parser.add_argument("--image_dir", type=str, default="/media/titrom/storage/mipt/datasets/ScanNet_12/val/scene0050_02/color/", help="Path to input images (for extract_compress)")
    parser.add_argument("--output_dir", type=str, default="results/", help="Directory to save results (for both modes)")
    parser.add_argument("--scenes_dir", type=str, default="/media/titrom/storage/mipt/datasets/ScanNet_12/val/", help="Path to directory containing ScanNet scenes (for eval)")
    parser.add_argument("--label_file", type=str, default="scannetv2-labels.modified.tsv", help="Label mapping file (e.g. scannetv2-labels.modified.tsv) (for eval)")
    parser.add_argument("--dataset_type", type=str, default="cocomap", help="Dataset type (cocomap or scannet20) (for eval)")
    parser.add_argument("--dino_cfg", type=str, default="talk2dino.yml", help="Path to DINO model config")
    parser.add_argument("--model_path", type=str, default="./openseg_model", help="Path to OpenSeg model directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.model == "dino":
        if not args.scenes_dir or not args.label_file:
            print(f"Please provide --scenes_dir and --label_file for eval mode.")
            exit(1)
        eval_talk2dino(args)
    elif args.model == "openseg":
        if not args.scenes_dir or not args.label_file or not args.model_path:
            print(f"Please provide --scenes_dir, --label_file, and --model_path for openseg mode.")
            exit(1)
        eval_openseg(args) 