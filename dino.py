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

# ------------------------------------------------------------
# Load DINOv2
# ------------------------------------------------------------
dinov2 = torch.hub.load(
    "facebookresearch/dinov2",
    MODEL_NAME,
    pretrained=True
).to(DEVICE)
dinov2.eval()

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
    
    # Slide feats2 over feats1
    for i in range(corr_h):
        for j in range(corr_w):
            # Extract window from feats1
            window = feats1[i:i+H2, j:j+W2, :]  # (H2, W2, C)
            
            # Compute mean cosine similarity between window and feats2
            similarity = (window * feats2).sum(dim=-1)  # (H2, W2)
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
# Example usage
# ------------------------------------------------------------
if __name__ == "__main__":
    img1 = Image.open("./assets/DSC_6466.JPG").convert("RGB")
    img2 = Image.open("./assets/DSC_6470.JPG").convert("RGB")

    # Keep resized copies aligned with feature maps
    img1_resized, _ = resize_to_divisible(img1)
    img2_resized, _ = resize_to_divisible(img2)

    feats1 = extract_dense_features(img1_resized)
    feats2 = extract_dense_features(img2_resized)

    best_row, best_col, best_score, correlation_map = compute_shared_masks(
        feats1, feats2, SIMILARITY_THRESHOLD
    )

    print(f"Best position: row={best_row}, col={best_col}")
    print(f"Correlation score: {best_score:.4f}")
    print(f"Correlation map shape: {correlation_map.shape}")
    
    # Visualize correlation map
    if correlation_map.size > 1 and correlation_map.max() != correlation_map.min():
        corr_vis = ((correlation_map - correlation_map.min()) / 
                    (correlation_map.max() - correlation_map.min()) * 255).astype(np.uint8)
        corr_vis = cv2.applyColorMap(corr_vis, cv2.COLORMAP_JET)
        cv2.imwrite("./assets/correlation_map.png", corr_vis)
        print("Correlation map saved to ./assets/correlation_map.png")
    else:
        print("Correlation map too small to visualize (images have same feature size)")
    
    # Convert patch coordinates to pixel coordinates
    pixel_row = best_row * PATCH_SIZE
    pixel_col = best_col * PATCH_SIZE
    print(f"Best position in pixels: row={pixel_row}, col={pixel_col}")

    # Overlay visualization: place image2 (grayscale) over image1 at best position
    base_np = np.array(img1_resized)
    overlay_gray = np.array(img2_resized.convert("L"))
    overlay_bgr = cv2.cvtColor(overlay_gray, cv2.COLOR_GRAY2BGR)

    r0, c0 = pixel_row, pixel_col
    r1 = min(r0 + overlay_bgr.shape[0], base_np.shape[0])
    c1 = min(c0 + overlay_bgr.shape[1], base_np.shape[1])

    overlay_crop = overlay_bgr[:r1 - r0, :c1 - c0]
    base_crop = base_np[r0:r1, c0:c1]

    blended = cv2.addWeighted(base_crop, 0.5, overlay_crop, 0.5, 0)
    vis = base_np.copy()
    vis[r0:r1, c0:c1] = blended
    cv2.rectangle(vis, (c0, r0), (c1, r1), (0, 255, 0), 2)
    cv2.imwrite("./assets/overlay_debug.png", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print("Overlay visualization saved to ./assets/overlay_debug.png")