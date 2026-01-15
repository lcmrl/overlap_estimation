import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "dinov2_vits14"   # vits14 / vitb14 / vitl14
SIMILARITY_THRESHOLD = 0.9     # tune based on data
PATCH_SIZE = 14
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# Transformation search parameters
ROTATION_ANGLES = [0]  # degrees
SCALE_FACTORS = [0.80, 1.00, 1.20]       # relative to original size

# ------------------------------------------------------------
# Load DINOv2
# ------------------------------------------------------------
print("Loading DINOv2 model (this may take a minute on first run)...")
try:
    dinov2 = torch.hub.load(
        "facebookresearch/dinov2",
        MODEL_NAME,
        pretrained=True,
        force_reload=False  # Use cached version if available
    ).to(DEVICE)
    dinov2.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading DINOv2 model: {e}")
    print("This may be due to network issues. Please check your internet connection.")
    print("If the model was previously downloaded, it may be cached locally.")
    raise

# ------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------
def make_divisible(size, divisor=14):
    """Make dimensions divisible by patch size."""
    return (size // divisor) * divisor

def resize_to_divisible(img: Image.Image):
    """Resize image so both sides are divisible by PATCH_SIZE."""
    w, h = img.size
    new_w = make_divisible(w, PATCH_SIZE)
    new_h = make_divisible(h, PATCH_SIZE)
    if new_w != w or new_h != h:
        img = img.resize((new_w, new_h), Image.BICUBIC)
    return img, (new_w, new_h)

def transform_image(img: Image.Image):
    """Transform image while preserving aspect ratio, making size divisible by patch size."""
    img, _ = resize_to_divisible(img)
    img_tensor = transforms.ToTensor()(img)
    img_tensor = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )(img_tensor)
    return img_tensor

# ------------------------------------------------------------
# Dense feature extraction
# ------------------------------------------------------------
@torch.no_grad()
def extract_dense_features(img: Image.Image):
    """
    Returns:
        features: (H_patches, W_patches, C)
    """
    x = transform_image(img).unsqueeze(0).to(DEVICE)

    # get patch tokens only (no CLS)
    feats = dinov2.forward_features(x)["x_norm_patchtokens"]
    feats = feats.squeeze(0)  # (N, C)

    h = x.shape[-2] // PATCH_SIZE
    w = x.shape[-1] // PATCH_SIZE

    feats = feats.reshape(h, w, -1)
    feats = F.normalize(feats, dim=-1)

    return feats.cpu()

# ------------------------------------------------------------
# Cross-correlation to find best position
# ------------------------------------------------------------
def compute_shared_masks(feats1, feats2, threshold=0.7):
    """
    Cross-correlate feats2 onto feats1 to find the best matching position.
    Uses cosine similarity since features are already L2-normalized.
    
    Args:
        feats1: (H1, W1, C) - features from image 1 (larger/base image)
        feats2: (H2, W2, C) - features from image 2 (template to find)
        threshold: unused in this version, kept for compatibility
    
    Returns:
        best_row: row offset where feats2 best matches on feats1
        best_col: column offset where feats2 best matches on feats1
        best_score: correlation score at the best position
        correlation_map: (H1-H2+1, W1-W2+1) map of correlation scores
    """
    H1, W1, C = feats1.shape
    H2, W2, _ = feats2.shape
    
    if H2 > H1 or W2 > W1:
        raise ValueError("feats2 must be smaller than or equal to feats1")
    
    # Initialize correlation map
    corr_h = H1 - H2 + 1
    corr_w = W1 - W2 + 1
    correlation_map = torch.zeros(corr_h, corr_w)
    
    # Features are already L2-normalized, so dot product = cosine similarity
    # Slide feats2 over feats1 and compute average cosine similarity
    for i in range(corr_h):
        for j in range(corr_w):
            # Extract window from feats1
            window = feats1[i:i+H2, j:j+W2, :]  # (H2, W2, C)
            
            # Compute mean cosine similarity (element-wise dot product)
            # Since both are L2-normalized, this gives cosine similarity per patch
            similarity = (window * feats2).sum(dim=-1)  # (H2, W2) - cosine sim per patch
            
            # Average similarity across all patches
            correlation_map[i, j] = similarity.mean()
    
    # Find best position
    best_idx = correlation_map.argmax()
    best_row = best_idx // corr_w
    best_col = best_idx % corr_w
    best_score = correlation_map[best_row, best_col].item()
    
    return best_row.item(), best_col.item(), best_score, correlation_map.numpy()

# ------------------------------------------------------------
# Utility: Upsample mask to image size
# ------------------------------------------------------------
def upsample_mask(mask, target_size):
    return cv2.resize(
        mask.astype(np.uint8),
        target_size,
        interpolation=cv2.INTER_NEAREST
    )

# ------------------------------------------------------------
# Helpers for I/O and visualization
# ------------------------------------------------------------
def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def list_images(folder: str):
    files = []
    for name in os.listdir(folder):
        ext = os.path.splitext(name)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            files.append(os.path.join(folder, name))
    return sorted(files)


def save_correlation_map(correlation_map: np.ndarray, output_path: str):
    if correlation_map.size > 1 and correlation_map.max() != correlation_map.min():
        corr_vis = ((correlation_map - correlation_map.min()) /
                    (correlation_map.max() - correlation_map.min()) * 255).astype(np.uint8)
        corr_vis = cv2.applyColorMap(corr_vis, cv2.COLORMAP_JET)
        cv2.imwrite(output_path, corr_vis)
        return True
    return False


def save_overlay(base_img: Image.Image, overlay_img: Image.Image,
                 pixel_row: int, pixel_col: int, output_path: str, angle: float = 0, scale: float = 1.0):
    base_np = np.array(base_img)
    # Keep the transformed image in color (RGB)
    overlay_rgb = np.array(overlay_img)

    r0, c0 = pixel_row, pixel_col
    r1 = min(r0 + overlay_rgb.shape[0], base_np.shape[0])
    c1 = min(c0 + overlay_rgb.shape[1], base_np.shape[1])

    overlay_crop = overlay_rgb[:r1 - r0, :c1 - c0]
    base_crop = base_np[r0:r1, c0:c1]

    # Blend the transformed colored image2 with image1
    blended = cv2.addWeighted(base_crop, 0.5, overlay_crop, 0.5, 0)
    vis = base_np.copy()
    vis[r0:r1, c0:c1] = blended
    cv2.rectangle(vis, (c0, r0), (c1, r1), (0, 255, 0), 3)
    
    # Add text showing transformation parameters
    text = f"Rotation: {angle:.1f}deg, Scale: {scale:.2f}x"
    cv2.putText(vis, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    
    cv2.imwrite(output_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


def apply_rotation_and_scale(img: Image.Image, angle: float, scale: float) -> Image.Image:
    """Apply rotation and scaling to an image."""
    w, h = img.size
    # Scale first
    new_w = int(w * scale)
    new_h = int(h * scale)
    img_scaled = img.resize((new_w, new_h), Image.BICUBIC)
    
    # Then rotate (expand=True to avoid cropping)
    img_rotated = img_scaled.rotate(angle, resample=Image.BICUBIC, expand=True)
    
    return img_rotated


# ------------------------------------------------------------
# End-to-end processing
# ------------------------------------------------------------
def process_pair(img1_path: str, img2_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    img1 = load_image(img1_path)
    img2 = load_image(img2_path)

    # Keep resized copy of img1 aligned with feature maps
    img1_resized, _ = resize_to_divisible(img1)
    feats1 = extract_dense_features(img1_resized)

    # Search over rotations and scales
    best_overall_score = -float('inf')
    best_result = None
    all_results = []

    total_transforms = len(ROTATION_ANGLES) * len(SCALE_FACTORS)
    print(f"Searching {total_transforms} transformations (rotations × scales)...")
    
    for angle in ROTATION_ANGLES:
        for scale in SCALE_FACTORS:
            try:
                # Apply transformation
                img2_transformed = apply_rotation_and_scale(img2, angle, scale)
                img2_resized, _ = resize_to_divisible(img2_transformed)
                
                # Extract features
                feats2 = extract_dense_features(img2_resized)
                
                # Check if feats2 fits in feats1
                if feats2.shape[0] > feats1.shape[0] or feats2.shape[1] > feats1.shape[1]:
                    continue  # Skip if transformed image is too large
                
                # Compute correlation
                best_row, best_col, best_score, correlation_map = compute_shared_masks(
                    feats1, feats2, SIMILARITY_THRESHOLD
                )
                
                # Store all results for later analysis
                all_results.append({
                    'angle': angle,
                    'scale': scale,
                    'score': best_score,
                    'row': best_row,
                    'col': best_col
                })
                
                # Track best result
                if best_score > best_overall_score:
                    best_overall_score = best_score
                    best_result = {
                        'angle': angle,
                        'scale': scale,
                        'row': best_row,
                        'col': best_col,
                        'score': best_score,
                        'correlation_map': correlation_map,
                        'img2_transformed': img2_resized
                    }
                    print(f"  New best: angle={angle}°, scale={scale:.2f}, score={best_score:.4f}")
            
            except Exception as e:
                print(f"  Skipped angle={angle}°, scale={scale:.2f}: {e}")
                continue
    
    if best_result is None:
        print("No valid transformation found!")
        return
    
    # Print all results sorted by score for debugging
    print(f"\nAll results (sorted by score):")
    for result in sorted(all_results, key=lambda x: x['score'], reverse=True)[:10]:
        print(f"  angle={result['angle']:3.0f}°, scale={result['scale']:.2f}, score={result['score']:8.4f}")
    
    # Use best result
    angle = best_result['angle']
    scale = best_result['scale']
    best_row = best_result['row']
    best_col = best_result['col']
    best_score = best_result['score']
    correlation_map = best_result['correlation_map']
    img2_best = best_result['img2_transformed']
    
    # Convert patch coordinates to pixel coordinates
    pixel_row = best_row * PATCH_SIZE
    pixel_col = best_col * PATCH_SIZE

    # Save transformed image2
    transformed_path = os.path.join(output_dir, "img2_transformed.png")
    img2_best.save(transformed_path)
    
    # Save correlation map
    corr_path = os.path.join(output_dir, "correlation_map.png")
    corr_saved = save_correlation_map(correlation_map, corr_path)

    # Save overlay visualization
    overlay_path = os.path.join(output_dir, "overlay_debug.png")
    save_overlay(img1_resized, img2_best, pixel_row, pixel_col, overlay_path, angle, scale)

    print(f"\nProcessed pair:\n  img1={img1_path}\n  img2={img2_path}")
    print(f"  Best transformation: angle={angle}°, scale={scale:.2f}")
    print(f"  Best position (patch coords): row={best_row}, col={best_col}")
    print(f"  Best position (pixels): row={pixel_row}, col={pixel_col}")
    print(f"  Correlation score: {best_score:.4f}")
    print(f"  Correlation map shape: {correlation_map.shape}")
    if corr_saved:
        print(f"  Saved correlation map: {corr_path}")
    else:
        print("  Correlation map too small/flat to visualize")
    print(f"  Saved transformed img2: {transformed_path}")
    print(f"  Saved overlay: {overlay_path}\n")


def run_on_subfolders(root_dir: str):
    if not os.path.isdir(root_dir):
        raise ValueError(f"Root directory not found: {root_dir}")

    subfolders = [
        os.path.join(root_dir, name)
        for name in sorted(os.listdir(root_dir))
        if os.path.isdir(os.path.join(root_dir, name))
    ]

    if not subfolders:
        print("No subfolders found to process.")
        return

    for sub in subfolders:
        images = list_images(sub)
        if len(images) < 2:
            print(f"Skipping {sub}: found {len(images)} image(s), need at least 2.")
            continue
        if len(images) > 2:
            print(f"Note: {sub} has {len(images)} images; using the first two after sorting.")
        img1_path, img2_path = images[0], images[1]
        output_dir = os.path.join(sub, "outputs")
        process_pair(img1_path, img2_path, output_dir)

# ------------------------------------------------------------
# Example usage
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-correlate images to find best alignment.")
    parser.add_argument("root", help="Root folder containing subfolders; each subfolder has two images.")
    parser.add_argument("--single", nargs=2, metavar=("IMG1", "IMG2"), help="Run on a single pair instead of folder traversal.")
    args = parser.parse_args()

    if args.single:
        img1_path, img2_path = args.single
        output_dir = os.path.join(os.path.dirname(img1_path) or ".", "outputs")
        process_pair(img1_path, img2_path, output_dir)
    else:
        run_on_subfolders(args.root)