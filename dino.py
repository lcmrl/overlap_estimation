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
MODEL_NAME = "dinov2_vits14"
PATCH_SIZE = 14
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
SCALES = [0.6]

TOPK_RATIO = 0.25   # use top 25% most similar patches

# ------------------------------------------------------------
# Load DINOv2
# ------------------------------------------------------------
print("Loading DINOv2 model...")
dinov2 = torch.hub.load(
    "facebookresearch/dinov2",
    MODEL_NAME,
    pretrained=True
).to(DEVICE)
dinov2.eval()
print("Model loaded.")

# ------------------------------------------------------------
# Preprocessing (PAD, do NOT resize)
# ------------------------------------------------------------
def pad_to_divisible(img: Image.Image, divisor=PATCH_SIZE):
    w, h = img.size
    pad_w = (divisor - w % divisor) % divisor
    pad_h = (divisor - h % divisor) % divisor
    if pad_w or pad_h:
        img = ImageOps.expand(img, (0, 0, pad_w, pad_h))
    return img

def transform_image(img: Image.Image):
    img = pad_to_divisible(img)
    tensor = transforms.ToTensor()(img)
    tensor = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )(tensor)
    return tensor

def scale_image(img: Image.Image, scale: float):
    w, h = img.size
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h), Image.BICUBIC)

# ------------------------------------------------------------
# Dense feature extraction
# ------------------------------------------------------------
@torch.no_grad()
def extract_dense_features(img: Image.Image):
    """
    Returns:
        feats: (C, H, W)  L2-normalized
    """
    x = transform_image(img).unsqueeze(0).to(DEVICE)

    feats = dinov2.forward_features(x)["x_norm_patchtokens"]
    feats = feats.squeeze(0)  # (N, C)

    H = x.shape[-2] // PATCH_SIZE
    W = x.shape[-1] // PATCH_SIZE

    feats = feats.reshape(H, W, -1).permute(2, 0, 1)  # (C, H, W)
    feats = F.normalize(feats, dim=0)

    return feats.cpu()

# ------------------------------------------------------------
# Vectorized cross-correlation
# ------------------------------------------------------------
def correlate(feats1, feats2):
    """
    feats1: (C, H1, W1)
    feats2: (C, H2, W2)

    Returns:
        corr_map: (H1-H2+1, W1-W2+1)
    """
    C, H1, W1 = feats1.shape
    _, H2, W2 = feats2.shape

    if H2 > H1 or W2 > W1:
        raise ValueError("Template larger than base image.")

    # Prepare tensors for conv2d
    feats1 = feats1.unsqueeze(0)            # (1, C, H1, W1)
    feats2 = feats2.unsqueeze(0)            # (1, C, H2, W2)

    # Valid convolution = sliding dot product
    corr = F.conv2d(feats1, feats2, stride=1)  # (1, 1, H1-H2+1, W1-W2+1)

    # Top-k aggregation
    k = int(H2 * W2 * TOPK_RATIO)
    corr_flat = corr.view(-1)
    topk_vals = torch.topk(corr_flat, k=k).values

    corr = corr / k
    return corr.squeeze()  # Remove all singleton dimensions -> (H1-H2+1, W1-W2+1)

# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------
def save_overlay(base_img, tmpl_img, row, col, out_path):
    base = np.array(base_img)
    tmpl = np.array(tmpl_img)

    r1 = min(row + tmpl.shape[0], base.shape[0])
    c1 = min(col + tmpl.shape[1], base.shape[1])

    base_crop = base[row:r1, col:c1]
    tmpl_crop = tmpl[:r1-row, :c1-col]

    blended = cv2.addWeighted(base_crop, 0.5, tmpl_crop, 0.5, 0)
    base[row:r1, col:c1] = blended

    cv2.rectangle(base, (col, row), (c1, r1), (0, 255, 0), 3)
    cv2.imwrite(out_path, cv2.cvtColor(base, cv2.COLOR_RGB2BGR))

# ------------------------------------------------------------
# Main processing
# ------------------------------------------------------------
def process_pair(img1_path, img2_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")

    img1_pad = pad_to_divisible(img1)
    feats1 = extract_dense_features(img1_pad)

    best = {
        "score": -float("inf"),
        "scale": None,
        "row": None,
        "col": None,
        "img2_scaled_pad": None,
        "corr_map": None
    }

    print(f"Searching over scales: {SCALES}")

    for scale in SCALES:
        try:
            img2_scaled = scale_image(img2, scale)
            img2_scaled_pad = pad_to_divisible(img2_scaled)

            feats2 = extract_dense_features(img2_scaled_pad)

            # Skip if template larger than base
            if feats2.shape[1] > feats1.shape[1] or feats2.shape[2] > feats1.shape[2]:
                print(f"  Scale {scale:.2f}: skipped (template too large)")
                continue

            corr_map = correlate(feats1, feats2)
            corr_np = corr_map.numpy()

            idx = corr_np.argmax()
            Hc, Wc = corr_np.shape
            row = idx // Wc
            col = idx % Wc
            score = corr_np[row, col]

            print(f"  Scale {scale:.2f}: best score {score:.4f}")

            if score > best["score"]:
                best.update({
                    "score": score,
                    "scale": scale,
                    "row": row,
                    "col": col,
                    "img2_scaled_pad": img2_scaled_pad,
                    "corr_map": corr_np
                })

        except Exception as e:
            print(f"  Scale {scale:.2f}: error ({e})")

    if best["scale"] is None:
        print("No valid scale produced a result.")
        return

    # Convert patch coords → pixel coords
    pixel_row = best["row"] * PATCH_SIZE
    pixel_col = best["col"] * PATCH_SIZE

    # Save overlay
    save_overlay(
        img1_pad,
        best["img2_scaled_pad"],
        pixel_row,
        pixel_col,
        os.path.join(output_dir, "overlay.png")
    )

    # Save correlation map
    corr = best["corr_map"]
    corr_vis = (corr - corr.min()) / (np.ptp(corr) + 1e-6)
    corr_vis = (corr_vis * 255).astype(np.uint8)
    corr_vis = cv2.applyColorMap(corr_vis, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(output_dir, "correlation_map.png"), corr_vis)

    print("\nBest overall match:")
    print(f"  Scale: {best['scale']:.2f}")
    print(f"  Patch coords: row={best['row']}, col={best['col']}")
    print(f"  Pixel coords: row={pixel_row}, col={pixel_col}")
    print(f"  Score: {best['score']:.4f}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("img1")
    parser.add_argument("img2")
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(args.img1) or ".", "outputs")
    process_pair(args.img1, args.img2, out_dir)
