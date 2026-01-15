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
transform = transforms.Compose([
    transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(518),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
])

# ------------------------------------------------------------
# Dense feature extraction
# ------------------------------------------------------------
@torch.no_grad()
def extract_dense_features(img: Image.Image):
    """
    Returns:
        features: (H_patches, W_patches, C)
    """
    x = transform(img).unsqueeze(0).to(DEVICE)

    # get patch tokens only (no CLS)
    feats = dinov2.forward_features(x)["x_norm_patchtokens"]
    feats = feats.squeeze(0)  # (N, C)

    h = x.shape[-2] // PATCH_SIZE
    w = x.shape[-1] // PATCH_SIZE

    feats = feats.reshape(h, w, -1)
    feats = F.normalize(feats, dim=-1)

    return feats.cpu()

# ------------------------------------------------------------
# Compute shared-pixel masks
# ------------------------------------------------------------
def compute_shared_masks(feats1, feats2, threshold=0.7):
    """
    feats1: (H1, W1, C)
    feats2: (H2, W2, C)
    """
    f1 = feats1.reshape(-1, feats1.shape[-1])
    f2 = feats2.reshape(-1, feats2.shape[-1])

    # cosine similarity
    sim = torch.matmul(f1, f2.T)

    max_sim_1, _ = sim.max(dim=1)
    max_sim_2, _ = sim.max(dim=0)

    mask1 = (max_sim_1 > threshold).reshape(feats1.shape[:2])
    mask2 = (max_sim_2 > threshold).reshape(feats2.shape[:2])

    return mask1.numpy(), mask2.numpy()

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

    feats1 = extract_dense_features(img1)
    feats2 = extract_dense_features(img2)

    mask1, mask2 = compute_shared_masks(
        feats1, feats2, SIMILARITY_THRESHOLD
    )

    # Upsample masks to original resolution
    mask1_up = upsample_mask(mask1, img1.size)
    mask2_up = upsample_mask(mask2, img2.size)

    cv2.imwrite("./assets/shared_mask_img1.png", mask1_up * 255)
    cv2.imwrite("./assets/shared_mask_img2.png", mask2_up * 255)