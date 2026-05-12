import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


LABELS = ["LEFT", "RIGHT", "UP", "DOWN", "CENTER", "NOT_VISIBLE"]
LABEL_TO_IDX = {k: i for i, k in enumerate(LABELS)}


@dataclass
class OfflineVisionPrediction:
    label: str
    confidence: float
    probs: Dict[str, float]


def _lazy_import_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        return torch, nn, F
    except Exception as e:
        raise ImportError(
            "PyTorch is required for offline vision classifier. "
            "Install with: pip install torch"
        ) from e


def _lazy_import_pil():
    try:
        from PIL import Image
        return Image
    except Exception as e:
        raise ImportError(
            "Pillow is required for offline vision classifier. "
            "Install with: pip install pillow"
        ) from e


def preprocess_rgb(rgb: np.ndarray, size: int = 128) -> "np.ndarray":
    """
    rgb: HxWx3 uint8
    returns: 3xSxS float32 in [0,1]
    """
    Image = _lazy_import_pil()
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)
    img = Image.fromarray(rgb, mode="RGB").resize((size, size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    return arr


def discretize_from_pixel_error(
    px: float,
    py: float,
    threshold: float = 15.0,
) -> str:
    """
    Convert continuous pixel error to a discrete label.
    px, py follow compute_pixel_error: centroid - image_center.
    +px: centroid is right of center
    +py: centroid is below center (image y increases downward)
    """
    if abs(px) < threshold and abs(py) < threshold:
        return "CENTER"
    if abs(px) >= abs(py):
        return "RIGHT" if px > 0 else "LEFT"
    return "DOWN" if py > 0 else "UP"


def _build_tiny_cnn(num_classes: int = 6):
    torch, nn, F = _lazy_import_torch()

    class TinyCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
            self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
            self.fc1 = nn.Linear(128 * 8 * 8, 256)
            self.fc2 = nn.Linear(256, num_classes)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv4(x))
            x = x.flatten(1)
            x = F.relu(self.fc1(x))
            return self.fc2(x)

    return TinyCNN()


class OfflineVisionClassifier:
    """
    Offline image classifier that predicts a discrete relation label from wrist RGB.
    Trained on auto-labeled sim images (segmentation centroid → label).
    """

    def __init__(
        self,
        model_path: str,
        input_size: int = 128,
        device: Optional[str] = None,
    ):
        self.model_path = model_path
        self.input_size = input_size
        torch, nn, F = _lazy_import_torch()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model = _build_tiny_cnn(num_classes=len(LABELS)).to(self.device)
        self.model.eval()

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Offline vision model weights not found at: {self.model_path}\n"
                "Train it with: python experiments/train_offline_vision_classifier.py"
            )

        state = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state)

    def predict(self, rgb: np.ndarray) -> OfflineVisionPrediction:
        torch, nn, F = _lazy_import_torch()

        chw = preprocess_rgb(rgb, size=self.input_size)
        x = torch.from_numpy(chw).unsqueeze(0).to(self.device)  # 1x3xSxS

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy().reshape(-1)

        idx = int(np.argmax(probs))
        label = LABELS[idx]
        confidence = float(probs[idx])
        prob_map = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}

        return OfflineVisionPrediction(label=label, confidence=confidence, probs=prob_map)

