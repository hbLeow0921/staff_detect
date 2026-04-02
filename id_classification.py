"""
Staff / ID-badge Classification Module — Enhanced + Speed-optimised.

Key features:
1. Upper-body crop   — nametags sit on the chest/torso area.
2. ImageNet norm     — matches Swin pre-training distribution.
3. Multi-crop ensemble — weighted full-body + upper-body scores.
4. Horizontal flip TTA — reduces pose bias (badge on left or right).
5. Batched inference — all crops (full, upper, flips) run in ONE forward
                       pass on the GPU, eliminating per-crop overhead.
"""

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image


class ID_Classificaiton:
    """Classify a person crop as staff (has badge) or non-staff."""

    def __init__(
        self,
        model_path: str,
        height: int = 256,
        width: int = 256,
        upper_body_ratio: float = 0.5,
        full_weight: float = 0.4,
        upper_weight: float = 0.6,
        use_flip_tta: bool = True,
    ):
        self.HEIGHT = height
        self.WIDTH = width
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.model = torch.load(model_path, map_location=self.device)
        self.model.to(self.device)
        self.model.eval()

        self.upper_body_ratio = upper_body_ratio
        total = full_weight + upper_weight
        self.full_weight = full_weight / total
        self.upper_weight = upper_weight / total
        self.use_flip_tta = use_flip_tta

        # Transform: resize -> tensor (no normalisation — matches training)
        # The model was trained with raw [0,1] pixel values, not ImageNet
        # normalised inputs, so we must NOT apply Normalize() here.
        self._transform = T.Compose([
            T.Resize((self.HEIGHT, self.WIDTH)),
            T.ToTensor(),
        ])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _to_pil(self, image):
        """Accept numpy array (RGB) or PIL Image."""
        if isinstance(image, np.ndarray):
            return Image.fromarray(image)
        return image

    def _prepare_tensor(self, pil_img):
        """Transform a PIL image to a normalised tensor (no batch dim)."""
        return self._transform(pil_img)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def output_detailed(self, image):
        """
        Classify a person crop and return all sub-scores.

        Uses BATCHED inference — all sub-crops run in a single forward
        pass, which is significantly faster on GPU than separate calls.

        Returns dict: {'score', 'full_score', 'upper_score'}
        """
        pil_img = self._to_pil(image)
        w, h = pil_img.size

        # Prepare upper-body crop
        upper_h = max(1, int(h * self.upper_body_ratio))
        upper_crop = pil_img.crop((0, 0, w, upper_h))

        # Build batch: [full, upper] + optionally their flips
        imgs = [pil_img, upper_crop]
        if self.use_flip_tta:
            imgs.append(pil_img.transpose(Image.FLIP_LEFT_RIGHT))
            imgs.append(upper_crop.transpose(Image.FLIP_LEFT_RIGHT))

        # Stack into a single batch tensor
        batch = torch.stack([self._prepare_tensor(im) for im in imgs])
        batch = batch.to(self.device)

        # Single forward pass for all crops
        with torch.no_grad():
            scores = self.model(batch).cpu().numpy().flatten()

        # Unpack scores
        if self.use_flip_tta:
            # scores: [full, upper, full_flip, upper_flip]
            full_score = float((scores[0] + scores[2]) / 2.0)
            upper_score = float((scores[1] + scores[3]) / 2.0)
        else:
            # scores: [full, upper]
            full_score = float(scores[0])
            upper_score = float(scores[1])

        final = self.full_weight * full_score + self.upper_weight * upper_score

        return {
            'score': float(final),
            'full_score': full_score,
            'upper_score': upper_score,
        }

    def output(self, image):
        """Backward-compatible: returns only the final ensemble score."""
        return self.output_detailed(image)['score']
