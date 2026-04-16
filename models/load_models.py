"""Utilities to discover and load trained deepfake detector checkpoints.

This script is compatible with checkpoints saved by the notebook, where files are
named like: model_<mode>_<timestamp>_<epoch>
Example: model_hybrid_20260412_213723_9
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.nn as nn
from torchvision import models


CHECKPOINT_RE = re.compile(r"^model_(rgb|fft|hybrid)_(\d{8}_\d{6})_(\d+)$")


@dataclass(frozen=True)
class CheckpointInfo:
    path: Path
    mode: str
    timestamp: str
    epoch: int


class HybridDeepfakeDetector(nn.Module):
    """Two-branch EfficientNet model used in the training notebook."""

    def __init__(self, mode: str = "hybrid", use_imagenet_weights: bool = False):
        super().__init__()
        self.mode = mode
        assert self.mode in ["rgb", "fft", "hybrid"], "Mode must be 'rgb', 'fft', or 'hybrid'"

        pretrained = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if use_imagenet_weights else None

        if self.mode in ["rgb", "hybrid"]:
            self.rgb_branch = models.efficientnet_b0(weights=pretrained)
            self.num_ftrs = self.rgb_branch.classifier[1].in_features
            self.rgb_branch.classifier = nn.Identity()

        if self.mode in ["fft", "hybrid"]:
            self.fft_branch = models.efficientnet_b0(weights=pretrained)
            self.num_ftrs = self.fft_branch.classifier[1].in_features
            self.fft_branch.classifier = nn.Identity()

        feature_dim = self.num_ftrs * 2 if self.mode == "hybrid" else self.num_ftrs
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, rgb_x: Optional[torch.Tensor] = None, fft_x: Optional[torch.Tensor] = None) -> torch.Tensor:
        features = []

        if self.mode in ["rgb", "hybrid"] and rgb_x is not None:
            features.append(self.rgb_branch(rgb_x))

        if self.mode in ["fft", "hybrid"] and fft_x is not None:
            features.append(self.fft_branch(fft_x))

        if not features:
            raise ValueError("No input provided for selected model mode.")

        combined = torch.cat(features, dim=1) if len(features) > 1 else features[0]
        out = self.fc(combined)
        return out.squeeze()


def parse_checkpoint_name(path: Path) -> Optional[CheckpointInfo]:
    match = CHECKPOINT_RE.match(path.name)
    if not match:
        return None

    mode, timestamp, epoch = match.groups()
    return CheckpointInfo(path=path, mode=mode, timestamp=timestamp, epoch=int(epoch))


def list_checkpoints(models_dir: str | Path) -> list[CheckpointInfo]:
    models_path = Path(models_dir)
    if not models_path.exists():
        raise FileNotFoundError(f"Models directory does not exist: {models_path}")

    parsed = [parse_checkpoint_name(p) for p in models_path.iterdir() if p.is_file()]
    checkpoints = [cp for cp in parsed if cp is not None]

    # Sort by timestamp first, then epoch so "latest" is deterministic.
    checkpoints.sort(key=lambda c: (c.timestamp, c.epoch))
    return checkpoints


def filter_by_mode(checkpoints: Iterable[CheckpointInfo], mode: Optional[str]) -> list[CheckpointInfo]:
    if mode is None:
        return list(checkpoints)

    mode = mode.lower()
    if mode not in {"rgb", "fft", "hybrid"}:
        raise ValueError("mode must be one of: rgb, fft, hybrid")

    return [cp for cp in checkpoints if cp.mode == mode]


def latest_checkpoint(models_dir: str | Path, mode: Optional[str] = None) -> CheckpointInfo:
    checkpoints = filter_by_mode(list_checkpoints(models_dir), mode)
    if not checkpoints:
        mode_hint = f" for mode '{mode}'" if mode else ""
        raise FileNotFoundError(f"No checkpoints found in {models_dir}{mode_hint}")
    return checkpoints[-1]


def load_model(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    mode: Optional[str] = None,
    use_imagenet_weights: bool = False,
) -> nn.Module:
    """Instantiate model architecture and load checkpoint weights.

    Args:
        checkpoint_path: Path to state_dict checkpoint file.
        device: Torch device (e.g., "cpu", "cuda", "cuda:0").
        mode: Model mode override. If not provided, inferred from filename.
        use_imagenet_weights: Whether to initialize EfficientNet with ImageNet
            weights before loading checkpoint. Usually not required.
    """

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    inferred = parse_checkpoint_name(ckpt_path)
    resolved_mode = (mode or (inferred.mode if inferred else None))
    if resolved_mode is None:
        raise ValueError("Could not infer mode from filename. Pass mode explicitly.")

    model = HybridDeepfakeDetector(mode=resolved_mode, use_imagenet_weights=use_imagenet_weights)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _resolve_checkpoint(args: argparse.Namespace) -> CheckpointInfo | Path:
    if args.checkpoint:
        return Path(args.checkpoint)

    if args.latest or args.mode:
        return latest_checkpoint(models_dir=args.models_dir, mode=args.mode).path

    # Default behavior: load latest checkpoint across all modes.
    return latest_checkpoint(models_dir=args.models_dir).path


def main() -> None:
    parser = argparse.ArgumentParser(description="Load deepfake detector checkpoints")
    parser.add_argument("--models-dir", default=Path(__file__).resolve().parent, help="Directory containing model checkpoint files")
    parser.add_argument("--checkpoint", default=None, help="Path to an explicit checkpoint file")
    parser.add_argument("--mode", choices=["rgb", "fft", "hybrid"], default=None, help="Mode to filter when selecting latest")
    parser.add_argument("--latest", action="store_true", help="Load latest checkpoint (optionally filtered by --mode)")
    parser.add_argument("--device", default="cpu", help='Torch device (e.g., "cpu", "cuda", "cuda:0")')
    parser.add_argument("--use-imagenet-weights", action="store_true", help="Initialize EfficientNet with pretrained ImageNet weights before loading checkpoint")

    args = parser.parse_args()

    checkpoint = _resolve_checkpoint(args)
    model = load_model(
        checkpoint_path=checkpoint,
        device=args.device,
        mode=args.mode,
        use_imagenet_weights=args.use_imagenet_weights,
    )

    print(f"Loaded checkpoint: {checkpoint}")
    print(f"Model mode: {model.mode}")
    print(f"Device: {args.device}")


if __name__ == "__main__":
    main()
