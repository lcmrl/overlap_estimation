import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image, ImageOps

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

SCALES = [0.8, 1.0, 1.2]
ROTATIONS = list(range(-30, 35, 10))

# Global variables that will be set by command line args
DINO_VERSION = None
MODEL_NAME = None
PATCH_SIZE = None
dino_model = None
dino_processor = None

# ------------------------------------------------------------
# Padding (top-left aligned - NOT symmetric)
# ------------------------------------------------------------
def pad_to_divisible(img: Image.Image, divisor=None):
    if divisor is None:
        divisor = PATCH_SIZE if PATCH_SIZE is not None else 14
    w, h = img.size
    pw = (divisor - w % divisor) % divisor
    ph = (divisor - h % divisor) % divisor

    if pw or ph:
        # Pad on right and bottom only to keep top-left alignment
        img = ImageOps.expand(img, (0, 0, pw, ph))
    return img

# ------------------------------------------------------------
# Transform
# ------------------------------------------------------------
def transform_image(img: Image.Image):
    t = transforms.ToTensor()(img)
    t = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )(t)
    return t

def scale_image(img: Image.Image, scale: float):
    w, h = img.size
    return img.resize(
        (max(1, int(w * scale)), max(1, int(h * scale))),
        Image.BICUBIC
    )

def rotate_image(img: Image.Image, angle: float):
    return img.rotate(-angle, resample=Image.BICUBIC, expand=True)

# ------------------------------------------------------------
# Patch mask from non-black pixels
# ------------------------------------------------------------
def compute_patch_mask(img: Image.Image):
    """
    Returns:
        mask: (1, 1, H, W) float tensor in {0,1}
    """
    img = pad_to_divisible(img)
    arr = np.array(img)

    non_black = np.any(arr > 0, axis=2).astype(np.float32)

    H, W = non_black.shape
    Hm = H // PATCH_SIZE
    Wm = W // PATCH_SIZE

    non_black = non_black[:Hm * PATCH_SIZE, :Wm * PATCH_SIZE]
    non_black = non_black.reshape(
        Hm, PATCH_SIZE, Wm, PATCH_SIZE
    )

    patch_mask = (non_black.mean(axis=(1, 3)) > 0.5).astype(np.float32)
    mask = torch.from_numpy(patch_mask)

    return mask.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

# ------------------------------------------------------------
# Dense features
# ------------------------------------------------------------
@torch.no_grad()
def extract_dense_features(img: Image.Image):
    # Assume image is already padded to divisible size
    # Get image dimensions for calculating patch grid size
    img_w, img_h = img.size
    H = img_h // PATCH_SIZE
    W = img_w // PATCH_SIZE
    
    if DINO_VERSION == "v2":
        x = transform_image(img).unsqueeze(0).to(DEVICE)
        feats = dino_model.forward_features(x)["x_norm_patchtokens"]
        feats = feats.squeeze(0)  # (N, C) where N = H*W
        
        # Despite the name "x_norm_patchtokens", these are NOT L2-normalized
        # Normalize them for consistent behavior with v3
        feats = F.normalize(feats, dim=1, p=2)
    
    elif DINO_VERSION == "v3":
        # DINOv3: Use manual transforms to preserve image dimensions
        # The processor would resize everything to 224x224, breaking cross-correlation
        x = transform_image(img).unsqueeze(0).to(DEVICE)
        outputs = dino_model(pixel_values=x)
        
        # Calculate feature map size from input
        H = x.shape[-2] // PATCH_SIZE
        W = x.shape[-1] // PATCH_SIZE
        expected_patches = H * W
        
        # Get all hidden states (includes CLS token and possibly register tokens)
        all_tokens = outputs.last_hidden_state.squeeze(0)  # (N_total, C)
        
        # DINOv3 may have: [CLS] + [registers] + [spatial patches] or [CLS] + [spatial patches] + [registers]
        # We need exactly H*W patch tokens
        # Skip CLS token (first), then take exactly expected_patches
        feats = all_tokens[1:1+expected_patches]  # Skip CLS, take spatial patches (N, C)
        
        if feats.shape[0] != expected_patches:
            print(f"WARNING: Expected {expected_patches} patches but got {feats.shape[0]}")
            # If still mismatched, try taking from the end (registers might be after CLS)
            feats = all_tokens[-expected_patches:]
        
        # DINOv3 features are NOT normalized, so we need to normalize them
        # For proper cosine similarity via convolution, each feature vector must be L2-normalized
        feats = F.normalize(feats, dim=1, p=2)  # Normalize along the feature dimension
    
    # feats is now (N, C) where N = H*W, and each vector is L2-normalized
    # Reshape to spatial grid and permute to (C, H, W)
    feats = feats.reshape(H, W, -1).permute(2, 0, 1)

    return feats.cpu()


# ------------------------------------------------------------
# Gaussian center weighting
# ------------------------------------------------------------
def create_gaussian_weights(H, W, sigma_factor=0.3):
    """
    Create 2D Gaussian weight map centered on the template.
    Higher weights at center, lower at edges.
    
    Args:
        H, W: Height and width of template in patches
        sigma_factor: Controls spread (0.3 means sigma = 30% of dimension)
    
    Returns:
        weights: (1, 1, H, W) tensor with Gaussian weights
    """
    # Create coordinate grids
    y = torch.arange(0, H, dtype=torch.float32)
    x = torch.arange(0, W, dtype=torch.float32)
    
    # Center coordinates
    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0
    
    # Compute distances from center
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    
    # Gaussian weights
    sigma_y = sigma_factor * H
    sigma_x = sigma_factor * W
    
    weights = torch.exp(
        -((yy - cy)**2 / (2 * sigma_y**2) + (xx - cx)**2 / (2 * sigma_x**2))
    )
    
    # Normalize so max weight is 1.0
    weights = weights / weights.max()
    
    # Add small constant to avoid zero weights at edges
    weights = weights * 0.9 + 0.1
    
    return weights.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

# ------------------------------------------------------------
# Masked correlation with optional Gaussian center weighting
# ------------------------------------------------------------
def masked_correlate(feats1, feats2, mask2, use_gaussian_weighting=True):
    """
    feats1: (C, H1, W1)
    feats2: (C, H2, W2)
    mask2 : (1, 1, H2, W2)
    
    Returns normalized correlation that is comparable across different template sizes.
    """
    feats1 = feats1.unsqueeze(0)  # (1, C, H1, W1)
    feats2 = feats2.unsqueeze(0)  # (1, C, H2, W2)

    H2, W2 = feats2.shape[2], feats2.shape[3]
    
    if not use_gaussian_weighting:
        # Simple masked correlation without Gaussian weighting
        mask2_expanded = mask2.expand(-1, feats2.shape[1], -1, -1)
        feats2_masked = feats2 * mask2_expanded
        
        corr = F.conv2d(feats1, feats2_masked)
        
        ones = torch.ones_like(mask2)
        valid_count = F.conv2d(ones, mask2)
        
        # Normalize by valid count to get average cosine similarity
        # This makes scores comparable across different template sizes
        corr = corr / (valid_count + 1e-6)
        return corr.squeeze(0).squeeze(0)
    
    # Gaussian weighted correlation - compute per-patch weighted sum
    # Create Gaussian weights
    gaussian_weights = create_gaussian_weights(H2, W2).to(feats2.device)
    
    # Combine with mask
    combined_weights = mask2 * gaussian_weights
    
    # Apply weights to features
    weights_expanded = combined_weights.expand(-1, feats2.shape[1], -1, -1)
    feats2_weighted = feats2 * weights_expanded
    
    # Compute weighted correlation
    corr = F.conv2d(feats1, feats2_weighted)
    
    # Normalize by sum of weights to get weighted average similarity
    # This makes scores comparable across different template sizes
    ones = torch.ones_like(mask2)
    sum_weights = F.conv2d(ones, combined_weights)
    
    corr = corr / (sum_weights + 1e-6)
    
    return corr.squeeze(0).squeeze(0)

# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------
def save_overlay(base_img, tmpl_img, row, col, out_path):
    base = np.array(base_img)
    tmpl = np.array(tmpl_img)

    r1 = min(row + tmpl.shape[0], base.shape[0])
    c1 = min(col + tmpl.shape[1], base.shape[1])

    blended = cv2.addWeighted(
        base[row:r1, col:c1], 0.5,
        tmpl[:r1-row, :c1-col], 0.5, 0
    )

    base[row:r1, col:c1] = blended
    cv2.rectangle(base, (col, row), (c1, r1), (0, 255, 0), 3)
    cv2.imwrite(out_path, cv2.cvtColor(base, cv2.COLOR_RGB2BGR))

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def process_pair(img1_path, img2_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")

    img1_pad = pad_to_divisible(img1)
    feats1 = extract_dense_features(img1_pad)

    best = {"score": -1e9}

    for rot in ROTATIONS:
        for scale in SCALES:
            img2r = rotate_image(img2, rot)
            img2rs = scale_image(img2r, scale)
            img2p = pad_to_divisible(img2rs)

            feats2 = extract_dense_features(img2p)
            mask2 = compute_patch_mask(img2p)
            
            # Resize mask to match feature dimensions (DINOv3 resizes images internally)
            if mask2.shape[-2:] != feats2.shape[-2:]:
                mask2 = F.interpolate(
                    mask2.float(),
                    size=feats2.shape[-2:],
                    mode='nearest'
                )

            if feats2.shape[1] > feats1.shape[1] or feats2.shape[2] > feats1.shape[2]:
                continue

            corr = masked_correlate(
                feats1,
                feats2,
                mask2.to(feats1.device)
            )

            corr_np = corr.numpy()
            idx = corr_np.argmax()
            Hc, Wc = corr_np.shape
            r, c = divmod(idx, Wc)

            score = corr_np[r, c]
            if score > best["score"]:
                best.update(
                    score=score,
                    rotation=rot,
                    scale=scale,
                    row=r,
                    col=c,
                    img2=img2p,
                    corr=corr_np
                )

    pr = best["row"] * PATCH_SIZE
    pc = best["col"] * PATCH_SIZE

    # Get center position of image2
    img2_height, img2_width = np.array(best["img2"]).shape[:2]
    center_row = pr + img2_height / 2.0
    center_col = pc + img2_width / 2.0

    best["img2"].save(os.path.join(output_dir, "img2_transformed.png"))
    save_overlay(
        img1_pad,
        best["img2"],
        pr, pc,
        os.path.join(output_dir, "overlay.png")
    )

    corr = best["corr"]
    corr_vis = ((corr - corr.min()) / (np.ptp(corr) + 1e-6) * 255).astype(np.uint8)
    cv2.imwrite(
        os.path.join(output_dir, "correlation_map.png"),
        cv2.applyColorMap(corr_vis, cv2.COLORMAP_JET)
    )

    # Save results to text file (center position, not top-left corner)
    results_file = os.path.join(output_dir, "results.txt")
    with open(results_file, 'w') as f:
        f.write(f"{center_row},{center_col},{best['rotation']},{best['scale']},{best['score']}\n")
    
    print("Best match:")
    print(f"  Center Position (row, col): ({center_row:.1f}, {center_col:.1f}) pixels")
    print(f"  Top-Left Position (row, col): ({pr}, {pc}) pixels")
    print(f"  Rotation: {best['rotation']}°")
    print(f"  Scale: {best['scale']}")
    print(f"  Score: {best['score']:.4f}")
    print(f"  Results saved to: {results_file}")

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def list_images(directory):
    """Find all image files in directory."""
    images = []
    for name in os.listdir(directory):
        ext = os.path.splitext(name)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            images.append(os.path.join(directory, name))
    return sorted(images)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find best alignment between two images.")
    parser.add_argument("directory", help="Directory containing exactly two images")
    parser.add_argument("--dino-version", choices=["v2", "v3"], default="v2",
                        help="DINO model version: v2 (DINOv2) or v3 (DINOv3, requires HF auth)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress output except errors")
    args = parser.parse_args()

    # Configure DINO version based on command line argument
    DINO_VERSION = args.dino_version
    
    if DINO_VERSION == "v2":
        MODEL_NAME = "dinov2_vits14"  # Options: dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14
        PATCH_SIZE = 14
    elif DINO_VERSION == "v3":
        # NOTE: DINOv3 models are gated on Hugging Face - requires authentication
        MODEL_NAME = "facebook/dinov3-vitl16-pretrain-sat493m"  # DINOv3 from Hugging Face
        PATCH_SIZE = 16
    else:
        raise ValueError(f"Unknown DINO version: {DINO_VERSION}")
    
    print(f"DINO Version: {DINO_VERSION}")
    print("Scales:", SCALES)
    print("Rotations:", ROTATIONS)
    
    # Load DINO model
    print(f"Loading {MODEL_NAME}...")
    
    if DINO_VERSION == "v2":
        try:
            dinov2 = torch.hub.load(
                "facebookresearch/dinov2",
                MODEL_NAME,
                pretrained=True
            ).to(DEVICE)
            dinov2.eval()
            dino_model = dinov2
            print("DINOv2 model loaded successfully!")
        except Exception as e:
            print(f"Error loading DINOv2: {e}")
            print("Attempting to load from local cache...")
            dinov2 = torch.hub.load(
                "facebookresearch/dinov2",
                MODEL_NAME,
                pretrained=True,
                force_reload=False
            ).to(DEVICE)
            dinov2.eval()
            dino_model = dinov2
            print("DINOv2 model loaded from cache.")
    elif DINO_VERSION == "v3":
        try:
            from transformers import AutoImageProcessor, AutoModel
            print("Loading DINOv3 from Hugging Face...")
            print(f"Note: {MODEL_NAME} is a gated model - you need Hugging Face authentication.")
            dino_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
            dino_model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
            dino_model.eval()
            print("DINOv3 model loaded successfully!")
        except ImportError:
            print("\nError: transformers library required for DINOv3.")
            print("Install with: pip install transformers")
            exit(1)
        except Exception as e:
            error_msg = str(e)
            if "gated" in error_msg.lower() or "401" in error_msg or "access" in error_msg.lower():
                print("\n" + "="*70)
                print("ERROR: DINOv3 model requires Hugging Face authentication")
                print("="*70)
                print("\nTo access this gated model, you need to:")
                print("1. Create a Hugging Face account at https://huggingface.co/")
                print("2. Request access to the model at:")
                print(f"   https://huggingface.co/{MODEL_NAME}")
                print("3. Login via command line:")
                print("   huggingface-cli login")
                print("4. Or set your token in environment:")
                print("   export HF_TOKEN=your_token_here")
                print("\nAlternatively, use DINOv2 with --dino-version v2")
                print("="*70)
            else:
                print(f"\nError loading DINOv3: {e}")
            exit(1)

    if not os.path.isdir(args.directory):
        print(f"Error: '{args.directory}' is not a directory")
        exit(1)

    images = list_images(args.directory)
    
    if len(images) < 2:
        print(f"Error: Found only {len(images)} image(s) in '{args.directory}', need at least 2")
        exit(1)
    
    if len(images) > 2:
        print(f"Warning: Found {len(images)} images, using the first two:")
    
    img1_path = images[0]
    img2_path = images[1]
    print(f"  Image 1: {os.path.basename(img1_path)}")
    print(f"  Image 2: {os.path.basename(img2_path)}")

    out_dir = os.path.join(args.directory, "outputs")
    process_pair(img1_path, img2_path, out_dir)
