#!/usr/bin/env python3
"""
Download HuggingFace models for WaveCommit project.
Saves models to the models/ directory structure.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import os
import sys
import torch
from safetensors import safe_open
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModel,
    AutoModelForAudioClassification,
)

# Model definitions
EMBEDDING_MODELS = {
    'hubert-large-ll60k': 'facebook/hubert-large-ll60k'
}

PREDICTION_HEAD_MODELS = {
    'hubert-large-superb-er': 'superb/hubert-large-superb-er',
    'hubert-large-superb-sid': 'superb/hubert-large-superb-sid',
    'hubert-large-superb-ks': 'superb/hubert-large-superb-ks',
    'hubert-large-superb-ic': 'superb/hubert-large-superb-ic'
}

REQUIRED_HEAD_KEY_SETS = (
    ('projector.weight', 'classifier.weight'),
    ('classifier.projector.weight', 'classifier.classifier.weight'),
)


def find_weight_file(model_dir: Path) -> Optional[Path]:
    """Find model weights file in a HuggingFace model directory."""
    safetensors_path = model_dir / 'model.safetensors'
    if safetensors_path.exists():
        return safetensors_path

    bin_path = model_dir / 'pytorch_model.bin'
    if bin_path.exists():
        return bin_path

    return None


def load_state_dict_keys(weight_file: Path) -> List[str]:
    """Load state dict keys from a safetensors or PyTorch binary file."""
    if weight_file.suffix == '.safetensors':
        with safe_open(weight_file, framework='pt', device='cpu') as f:
            return list(f.keys())

    data = torch.load(weight_file, map_location='cpu')
    if isinstance(data, dict) and 'state_dict' in data and isinstance(data['state_dict'], dict):
        data = data['state_dict']
    if not isinstance(data, dict):
        return []
    return list(data.keys())


def validate_prediction_head_artifacts(model_dir: Path) -> Tuple[bool, List[str], str]:
    """
    Check if a local prediction-head directory contains classifier weights.

    Returns:
        (is_valid, missing_keys, details)
    """
    weight_file = find_weight_file(model_dir)
    if weight_file is None:
        expected = [', '.join(keys) for keys in REQUIRED_HEAD_KEY_SETS]
        return False, expected, f'No weights file found in {model_dir}'

    try:
        keys = set(load_state_dict_keys(weight_file))
    except Exception as exc:
        expected = [', '.join(keys) for keys in REQUIRED_HEAD_KEY_SETS]
        return False, expected, f'Failed reading {weight_file.name}: {exc}'

    for key_set in REQUIRED_HEAD_KEY_SETS:
        if all(k in keys for k in key_set):
            return True, [], f'Found valid prediction-head weights in {weight_file.name}'

    expected = [', '.join(keys) for keys in REQUIRED_HEAD_KEY_SETS]
    return False, expected, f'Missing prediction-head keys in {weight_file.name}'


def download_model(
    model_id: str,
    save_dir: Path,
    model_name: str,
    model_kind: str,
    force: bool = False,
) -> bool:
    """Download and validate a HuggingFace model."""
    print(f"\nDownloading {model_name} ({model_id})...")
    target_dir = save_dir / model_name

    if target_dir.exists() and any(target_dir.iterdir()) and not force:
        if model_kind == 'prediction_head':
            ok, missing, details = validate_prediction_head_artifacts(target_dir)
            if ok:
                print(f"  Already exists at {target_dir}, validated: {details}")
                return True
            print(f"  Existing files are incomplete: {details}")
            print(f"  Missing keys: {missing}")
            print("  Re-downloading prediction-head weights...")
        else:
            print(f"  Already exists at {target_dir}, skipping...")
            return True

    try:
        target_dir.mkdir(parents=True, exist_ok=True)

        # Download config
        print("  Downloading config...")
        config = AutoConfig.from_pretrained(model_id)
        config.save_pretrained(target_dir)

        # Download model weights
        print("  Downloading model weights...")
        if model_kind == 'prediction_head':
            model = AutoModelForAudioClassification.from_pretrained(model_id)
        else:
            model = AutoModel.from_pretrained(model_id)
        model.save_pretrained(target_dir)

        # Try to download processor/feature extractor if available
        try:
            feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
            feature_extractor.save_pretrained(target_dir)
            print("  Downloaded feature extractor")
        except Exception:
            pass

        if model_kind == 'prediction_head':
            ok, missing, details = validate_prediction_head_artifacts(target_dir)
            if not ok:
                print(f"  ✗ Invalid prediction-head artifacts: {details}")
                print(f"  Missing keys: {missing}")
                return False

        print(f"  ✓ Saved to {target_dir}")
        return True

    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Download WaveCommit HuggingFace models')
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if local directories are present',
    )
    args = parser.parse_args()

    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    models_dir = project_root / 'models'
    
    print("=" * 60)
    print("WaveCommit Model Downloader")
    print("=" * 60)
    
    # Download embedding extractor
    print("\n1. Downloading embedding extractors...")
    embedding_dir = models_dir / 'embedding_extractor'
    
    for name, model_id in EMBEDDING_MODELS.items():
        success = download_model(
            model_id=model_id,
            save_dir=embedding_dir,
            model_name=name,
            model_kind='embedding',
            force=args.force,
        )
        if not success:
            print(f"Warning: Failed to download {name}")
    
    # Download prediction heads
    print("\n2. Downloading prediction heads...")
    heads_dir = models_dir / 'prediction_heads'
    
    for name, model_id in PREDICTION_HEAD_MODELS.items():
        success = download_model(
            model_id=model_id,
            save_dir=heads_dir,
            model_name=name,
            model_kind='prediction_head',
            force=args.force,
        )
        if not success:
            print(f"Warning: Failed to download {name}")
    
    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)
    
    # List downloaded models
    print("\nDownloaded models:")
    for subdir in ['embedding_extractor', 'prediction_heads']:
        dir_path = models_dir / subdir
        if dir_path.exists():
            print(f"\n  {subdir}/")
            for item in sorted(dir_path.iterdir()):
                if item.is_dir():
                    size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                    size_mb = size / (1024 * 1024)
                    print(f"    {item.name}/ ({size_mb:.1f} MB)")

if __name__ == '__main__':
    main()
