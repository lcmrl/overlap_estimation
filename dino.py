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

SCALES = [0.6]
ROTATIONS = list(range(-60, 60, 15))
print("Scales:", SCALES)
print("Rotations:", ROTATIONS)

# ------------------------------------------------------------
# Load DINOv2
# ------------------------------------------------------------
print("Loading DINOv2...")
dinov2 = torch.hub.load(
    "facebookresearch/dinov2",
    MODEL_NAME,
    pretrained=True
).to(DEVICE)
dinov2.eval()
print("Model loaded.")

# ------------------------------------------------------------
# Padding (top-left aligned - NOT symmetric)
# ------------------------------------------------------------
def pad_to_divisible(img: Image.Image, divisor=PATCH_SIZE):
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
    img = pad_to_divisible(img)
    x = transform_image(img).unsqueeze(0).to(DEVICE)

    feats = dinov2.forward_features(x)["x_norm_patchtokens"]
    feats = feats.squeeze(0)

    H = x.shape[-2] // PATCH_SIZE
    W = x.shape[-1] // PATCH_SIZE

    feats = feats.reshape(H, W, -1).permute(2, 0, 1)
    feats = F.normalize(feats, dim=0)

    return feats.cpu()

# ------------------------------------------------------------
# Masked correlation (CORRECT)
# ------------------------------------------------------------
def masked_correlate(feats1, feats2, mask2):
    """
    feats1: (C, H1, W1)
    feats2: (C, H2, W2)
    mask2 : (1, 1, H2, W2)
    """
    feats1 = feats1.unsqueeze(0)  # (1, C, H1, W1)
    feats2 = feats2.unsqueeze(0)  # (1, C, H2, W2)

    # CRITICAL: Mask features BEFORE correlation to zero out invalid regions
    # Expand mask to all channels: (1, 1, H2, W2) -> (1, C, H2, W2)
    mask2_expanded = mask2.expand(-1, feats2.shape[1], -1, -1)
    feats2_masked = feats2 * mask2_expanded

    # Now correlate with masked features
    corr = F.conv2d(feats1, feats2_masked)

    # Count how many valid patches contributed to each position
    ones = torch.ones_like(mask2)
    valid = F.conv2d(ones, mask2)

    # Normalize by number of valid patches
    corr = corr / (valid + 1e-6)
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

    print("Best match:")
    print(best)

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
