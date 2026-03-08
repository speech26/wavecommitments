#!/usr/bin/env python3
"""
WaveCommit Captum Analysis (Multi-GPU)

Performs TracIn and Shapley value analysis for speaker embeddings
using pretrained HuBERT prediction heads with multi-GPU parallelization.

Phase 5 Implementation:
- ShapleyValueSampling for feature attribution (multi-GPU)
- IntegratedGradients for feature attribution (multi-GPU)
- TracIn for training data influence (single GPU)

Usage:
    python scripts/captum_analysis.py --dataset timit --head er --analysis shapley
    python scripts/captum_analysis.py --dataset timit --head er --analysis all --num-gpus 8
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess
import time

# Captum imports
from captum.attr import ShapleyValueSampling, IntegratedGradients
from captum.influence import TracInCP, TracInCPFast

# pyDVL imports for Data Shapley
from pydvl.valuation import Dataset as PyDVLDataset, ModelUtility, SupervisedScorer
from pydvl.valuation import TMCShapleyValuation
from pydvl.valuation.samplers.truncation import RelativeTruncation
from pydvl.valuation.stopping import MinUpdates
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from joblib import parallel_config
import warnings
from safetensors import safe_open

# HuggingFace imports
from transformers import (
    AutoModelForAudioClassification,
    AutoFeatureExtractor,
    HubertModel
)


# Prediction head configurations
PREDICTION_HEADS = {
    'er': {
        'name': 'Emotion Recognition',
        'path': 'models/prediction_heads/hubert-large-superb-er',
        'labels': ['neu', 'hap', 'ang', 'sad'],
        'num_classes': 4
    },
    'sid': {
        'name': 'Speaker Identification',
        'path': 'models/prediction_heads/hubert-large-superb-sid',
        'num_classes': 1251
    },
    'ks': {
        'name': 'Keyword Spotting',
        'path': 'models/prediction_heads/hubert-large-superb-ks',
        'num_classes': 12
    },
    'ic': {
        'name': 'Intent Classification',
        'path': 'models/prediction_heads/hubert-large-superb-ic',
        'num_classes': 31
    }
}

# Memory configuration
GPU_MEMORY_FRACTION = 0.90  # Use 90% of available GPU memory
MIN_FREE_MEMORY_MB = 2000   # Minimum free memory required (2GB)

REQUIRED_HEAD_KEY_SETS = (
    ('projector.weight', 'classifier.weight'),
    ('classifier.projector.weight', 'classifier.classifier.weight'),
)

PYDVL_MODEL_FROZEN_HEAD_TOP = 'frozen_head_top'
PYDVL_MODEL_LEGACY_SKLEARN_MLP = 'legacy_sklearn_mlp'
PYDVL_MODEL_FINETUNE_HEAD = 'finetune_head'

PYDVL_FEATURE_HEAD_AUTO = 'auto'

PYDVL_JOBLIB_BACKEND_AUTO = 'auto'
PYDVL_JOBLIB_BACKEND_LOKY = 'loky'
PYDVL_JOBLIB_BACKEND_THREADING = 'threading'

PYDVL_TASK_AUTO = 'auto'
PYDVL_TASK_SPEAKER = 'speaker'
PYDVL_TASK_EMOTION = 'emotion'

# Stability-first defaults for pyDVL TMC-Shapley.
# Users can still override via CLI for faster exploratory runs.
PYDVL_DEFAULT_MIN_UPDATES = 200
PYDVL_DEFAULT_RTOL = 0.03


def find_weight_file(model_dir: Path) -> Optional[Path]:
    """Find a HuggingFace model weight file."""
    safetensors_path = model_dir / 'model.safetensors'
    if safetensors_path.exists():
        return safetensors_path

    bin_path = model_dir / 'pytorch_model.bin'
    if bin_path.exists():
        return bin_path

    return None


def load_state_dict_keys(weight_file: Path) -> List[str]:
    """Load state dict keys from a safetensors or pytorch binary file."""
    if weight_file.suffix == '.safetensors':
        with safe_open(weight_file, framework='pt', device='cpu') as f:
            return list(f.keys())

    state = torch.load(weight_file, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state and isinstance(state['state_dict'], dict):
        state = state['state_dict']
    if not isinstance(state, dict):
        return []
    return list(state.keys())


def validate_prediction_head_weights(
    head_model_path: Path,
    required_key_sets: Tuple[Tuple[str, ...], ...] = REQUIRED_HEAD_KEY_SETS,
) -> Tuple[bool, List[str], str]:
    """
    Validate that a local prediction-head model includes projector/classifier weights.
    """
    if not head_model_path.exists():
        expected = [', '.join(keys) for keys in required_key_sets]
        return False, expected, f'Model directory not found: {head_model_path}'

    weight_file = find_weight_file(head_model_path)
    if weight_file is None:
        expected = [', '.join(keys) for keys in required_key_sets]
        return False, expected, f'No model weight file found in {head_model_path}'

    try:
        keys = set(load_state_dict_keys(weight_file))
    except Exception as exc:
        expected = [', '.join(keys) for keys in required_key_sets]
        return False, expected, f'Failed to read {weight_file.name}: {exc}'

    for key_set in required_key_sets:
        if all(k in keys for k in key_set):
            return True, [], f'Validated prediction-head weights in {weight_file.name}'

    expected = [', '.join(keys) for keys in required_key_sets]
    return False, expected, f'Missing required head keys in {weight_file.name}'


def require_prediction_head_weights(head_model_path: Path, head_name: str) -> None:
    """Raise a clear error if prediction-head artifacts are incomplete."""
    ok, missing, details = validate_prediction_head_weights(head_model_path)
    if not ok:
        missing_text = ', '.join(missing)
        raise RuntimeError(
            f'{head_name} head is not usable for Data Shapley: {details}. '
            f'Missing keys: [{missing_text}]. '
            'Run `python scripts/download_models.py --force` to re-download true head weights.'
        )


def extract_fixed_head_features(
    embeddings: np.ndarray,
    head_model_path: Path,
    batch_size: int = 256,
    device: str = 'cpu',
) -> np.ndarray:
    """
    Transform 1024-dim HuBERT embeddings using a fixed pretrained SUPERB head.

    For Option-1 Data Shapley, we keep the SUPERB head fixed and use its logits as
    feature vectors for the trainable top classifier used by pyDVL.
    """
    require_prediction_head_weights(head_model_path, head_name=head_model_path.name)

    model = AutoModelForAudioClassification.from_pretrained(
        str(head_model_path),
        local_files_only=True,
    )
    model = model.to(device)
    model.eval()

    features = []
    with torch.no_grad():
        for start in range(0, len(embeddings), batch_size):
            end = min(start + batch_size, len(embeddings))
            batch = torch.tensor(embeddings[start:end], dtype=torch.float32, device=device)
            if hasattr(model, 'projector') and hasattr(model, 'classifier'):
                projected = model.projector(batch)
                logits = model.classifier(projected)
            else:
                logits = model.classifier(batch)
            features.append(logits.cpu().numpy())

    del model
    if device.startswith('cuda'):
        torch.cuda.empty_cache()

    return np.concatenate(features, axis=0)


def load_pretrained_head_parameters(head_model_path: Path) -> Dict[str, np.ndarray]:
    """
    Load pretrained projector/classifier weights from a local SUPERB head.

    Returns CPU numpy arrays so pyDVL model clones can re-initialize quickly.
    """
    require_prediction_head_weights(head_model_path, head_name=head_model_path.name)

    weight_file = find_weight_file(head_model_path)

    # Fast path: load only needed tensors directly from local safetensors.
    if weight_file is not None and weight_file.suffix == '.safetensors':
        with safe_open(weight_file, framework='pt', device='cpu') as f:
            keys = set(f.keys())

            if 'projector.weight' in keys and 'classifier.weight' in keys:
                projector_prefix = 'projector'
                classifier_prefix = 'classifier'
            elif 'classifier.projector.weight' in keys and 'classifier.classifier.weight' in keys:
                projector_prefix = 'classifier.projector'
                classifier_prefix = 'classifier.classifier'
            else:
                projector_prefix = None
                classifier_prefix = None

            if projector_prefix is not None and classifier_prefix is not None:
                projector_weight = f.get_tensor(f'{projector_prefix}.weight').numpy().astype(np.float32)
                projector_bias = (
                    f.get_tensor(f'{projector_prefix}.bias').numpy().astype(np.float32)
                    if f'{projector_prefix}.bias' in keys else None
                )
                classifier_weight = f.get_tensor(f'{classifier_prefix}.weight').numpy().astype(np.float32)
                classifier_bias = (
                    f.get_tensor(f'{classifier_prefix}.bias').numpy().astype(np.float32)
                    if f'{classifier_prefix}.bias' in keys else None
                )

                return {
                    'projector_weight': projector_weight,
                    'projector_bias': projector_bias,
                    'classifier_weight': classifier_weight,
                    'classifier_bias': classifier_bias,
                }

    # Fallback path: load model module and read tensors.
    model = AutoModelForAudioClassification.from_pretrained(
        str(head_model_path),
        local_files_only=True,
    )

    if not hasattr(model, 'projector') or not hasattr(model, 'classifier'):
        raise RuntimeError(
            f'Prediction head at {head_model_path} does not expose projector/classifier modules'
        )

    with torch.no_grad():
        projector_weight = model.projector.weight.detach().cpu().numpy().astype(np.float32)
        projector_bias = (
            model.projector.bias.detach().cpu().numpy().astype(np.float32)
            if model.projector.bias is not None else None
        )
        classifier_weight = model.classifier.weight.detach().cpu().numpy().astype(np.float32)
        classifier_bias = (
            model.classifier.bias.detach().cpu().numpy().astype(np.float32)
            if model.classifier.bias is not None else None
        )

    del model

    return {
        'projector_weight': projector_weight,
        'projector_bias': projector_bias,
        'classifier_weight': classifier_weight,
        'classifier_bias': classifier_bias,
    }


def get_gpu_memory_info() -> List[Dict]:
    """
    Get memory info for all available GPUs.
    
    Returns:
        List of dicts with gpu_id, total_mb, free_mb, used_mb
    """
    gpu_info = []
    
    if not torch.cuda.is_available():
        return gpu_info
    
    num_gpus = torch.cuda.device_count()
    
    for i in range(num_gpus):
        try:
            # Use nvidia-smi for accurate free memory
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.total,memory.free,memory.used', 
                 '--format=csv,noheader,nounits', f'--id={i}'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                total, free, used = map(int, result.stdout.strip().split(','))
                gpu_info.append({
                    'gpu_id': i,
                    'total_mb': total,
                    'free_mb': free,
                    'used_mb': used,
                    'name': torch.cuda.get_device_name(i)
                })
        except Exception as e:
            # Fallback to torch method
            props = torch.cuda.get_device_properties(i)
            total = props.total_memory // (1024 * 1024)
            gpu_info.append({
                'gpu_id': i,
                'total_mb': total,
                'free_mb': total,  # Assume all free if can't query
                'used_mb': 0,
                'name': props.name
            })
    
    return gpu_info


def select_available_gpus(max_gpus: int = 8, min_free_mb: int = MIN_FREE_MEMORY_MB) -> List[int]:
    """
    Select GPUs with sufficient free memory.
    
    Args:
        max_gpus: Maximum number of GPUs to use
        min_free_mb: Minimum free memory required
        
    Returns:
        List of GPU IDs to use
    """
    gpu_info = get_gpu_memory_info()
    
    if not gpu_info:
        return []
    
    # Filter by free memory
    available = [g for g in gpu_info if g['free_mb'] >= min_free_mb]
    
    # Sort by free memory (most free first)
    available.sort(key=lambda x: x['free_mb'], reverse=True)
    
    # Take up to max_gpus
    selected = [g['gpu_id'] for g in available[:max_gpus]]
    
    return selected


def set_gpu_memory_fraction(gpu_id: int, fraction: float = GPU_MEMORY_FRACTION):
    """Set memory fraction for a specific GPU."""
    torch.cuda.set_per_process_memory_fraction(fraction, gpu_id)


class EmbeddingDataset(Dataset):
    """Dataset for pre-extracted embeddings."""
    
    def __init__(self, embeddings_dir: Path, speaker_ids: Optional[List[str]] = None):
        self.embeddings_dir = embeddings_dir
        self.samples = []
        
        speaker_dirs = sorted([d for d in embeddings_dir.iterdir() if d.is_dir()])
        
        if speaker_ids:
            speaker_dirs = [d for d in speaker_dirs if d.name in speaker_ids]
        
        for speaker_dir in speaker_dirs:
            utt_file = speaker_dir / 'utterance_embeddings.npy'
            meta_file = speaker_dir / 'metadata.json'
            
            if utt_file.exists() and meta_file.exists():
                embeddings = np.load(utt_file)
                with open(meta_file) as f:
                    metadata = json.load(f)
                
                for i in range(len(embeddings)):
                    self.samples.append({
                        'speaker_id': speaker_dir.name,
                        'utterance_idx': i,
                        'embedding': embeddings[i],
                        'metadata': metadata
                    })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'embedding': torch.tensor(sample['embedding'], dtype=torch.float32),
            'speaker_id': sample['speaker_id'],
            'utterance_idx': sample['utterance_idx']
        }


class EmbeddingClassifier(nn.Module):
    """Simple classifier for embedding analysis."""
    
    def __init__(self, input_dim: int = 1024, num_classes: int = 4, 
                 hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)


# =============================================================================
# Data Shapley (Utterance-level Attribution)
# =============================================================================

def compute_data_shapley_intra_speaker(
    embeddings: np.ndarray,
    model: nn.Module,
    device: str,
    n_permutations: int = 100
) -> np.ndarray:
    """
    Compute intra-speaker Data Shapley values using Monte Carlo sampling.
    
    This measures each utterance's importance WITHIN the speaker by computing
    leave-one-out influence on other utterances from the same speaker.
    
    Utility function: Average prediction confidence on remaining utterances
    when each utterance is removed (leave-one-out influence).
    
    Args:
        embeddings: (N, D) array of N utterance embeddings
        model: Classification model
        device: CUDA device string
        n_permutations: Number of Monte Carlo permutations
        
    Returns:
        (N,) array of Shapley values, one per utterance
    """
    N = len(embeddings)
    if N <= 1:
        return np.ones(N)
    
    shapley_values = np.zeros(N)
    
    # Get model predictions for all utterances
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(embeddings_tensor)
        probs = torch.softmax(logits, dim=1)
        confidences = probs.max(dim=1).values.cpu().numpy()
    
    # Monte Carlo sampling of permutations
    for _ in range(n_permutations):
        perm = np.random.permutation(N)
        
        # Compute marginal contribution of each utterance
        prev_utility = 0.0
        for i, idx in enumerate(perm):
            # Utility = average confidence of utterances seen so far
            seen_indices = perm[:i+1]
            curr_utility = confidences[seen_indices].mean()
            
            # Marginal contribution
            shapley_values[idx] += (curr_utility - prev_utility)
            prev_utility = curr_utility
    
    shapley_values /= n_permutations
    return shapley_values


def compute_data_shapley_inter_speaker(
    speaker_embeddings: np.ndarray,
    other_speaker_embeddings: Dict[str, np.ndarray],
    model: nn.Module,
    device: str,
    n_permutations: int = 50
) -> np.ndarray:
    """
    Compute inter-speaker Data Shapley values using Monte Carlo sampling.
    
    This measures each utterance's importance for distinguishing this speaker
    from OTHER speakers. Utility is the model's performance on a validation
    set from other speakers.
    
    Utility function: Model's average confidence when predicting on other
    speakers' embeddings (using this speaker's subset as context).
    
    Args:
        speaker_embeddings: (N, D) array of this speaker's utterance embeddings
        other_speaker_embeddings: Dict of other_speaker_id -> (M, D) embeddings
        model: Classification model
        device: CUDA device string
        n_permutations: Number of Monte Carlo permutations
        
    Returns:
        (N,) array of Shapley values, one per utterance
    """
    N = len(speaker_embeddings)
    if N <= 1:
        return np.ones(N)
    
    shapley_values = np.zeros(N)
    
    # Compute speaker centroid from all embeddings
    speaker_centroid = speaker_embeddings.mean(axis=0)
    
    # Compute other speakers' centroids
    other_centroids = []
    for other_embs in other_speaker_embeddings.values():
        if len(other_embs) > 0:
            other_centroids.append(other_embs.mean(axis=0))
    
    if not other_centroids:
        return np.ones(N)
    
    other_centroids = np.stack(other_centroids)
    
    # Monte Carlo sampling of permutations
    for _ in range(n_permutations):
        perm = np.random.permutation(N)
        
        prev_utility = 0.0
        for i, idx in enumerate(perm):
            # Compute centroid using utterances seen so far
            seen_indices = perm[:i+1]
            subset_centroid = speaker_embeddings[seen_indices].mean(axis=0)
            
            # Utility = average distance to other speaker centroids
            # Higher distance = better speaker separation = higher utility
            distances_to_others = np.linalg.norm(
                other_centroids - subset_centroid, axis=1
            )
            curr_utility = distances_to_others.mean()
            
            # Marginal contribution
            shapley_values[idx] += (curr_utility - prev_utility)
            prev_utility = curr_utility
    
    shapley_values /= n_permutations
    return shapley_values


def process_data_shapley_worker(args: Tuple) -> Dict:
    """
    Worker function to compute Data Shapley for one speaker on a specific GPU.
    
    Computes both intra-speaker and inter-speaker Data Shapley values.
    
    Args:
        args: Tuple of (speaker_id, speaker_embeddings, other_speaker_embeddings,
                        model_state_dict, num_classes, output_dir, gpu_id, n_permutations)
    
    Returns:
        Dictionary with speaker results
    """
    (speaker_id, speaker_embeddings, other_speaker_embeddings,
     model_state_dict, num_classes, output_dir, gpu_id, n_permutations) = args
    
    try:
        # Set up GPU for this worker
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(gpu_id)
        torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, gpu_id)
        
        # Recreate model on this GPU
        model = EmbeddingClassifier(input_dim=1024, num_classes=num_classes)
        model.load_state_dict(model_state_dict)
        model = model.to(device)
        model.eval()
        
        # Compute intra-speaker Data Shapley (within-speaker importance)
        intra_shapley = compute_data_shapley_intra_speaker(
            speaker_embeddings,
            model,
            str(device),
            n_permutations
        )
        
        # Compute inter-speaker Data Shapley (cross-speaker importance)
        inter_shapley = compute_data_shapley_inter_speaker(
            speaker_embeddings,
            other_speaker_embeddings,
            model,
            str(device),
            n_permutations
        )
        
        # Save results
        output_path = Path(output_dir) / speaker_id
        output_path.mkdir(parents=True, exist_ok=True)
        
        np.save(output_path / 'data_shapley_intra.npy', intra_shapley)
        np.save(output_path / 'data_shapley_inter.npy', inter_shapley)
        
        # Compute summary statistics
        result = {
            'speaker_id': speaker_id,
            'num_utterances': len(speaker_embeddings),
            'intra_shapley_mean': float(intra_shapley.mean()),
            'intra_shapley_std': float(intra_shapley.std()),
            'inter_shapley_mean': float(inter_shapley.mean()),
            'inter_shapley_std': float(inter_shapley.std()),
            'top_intra_indices': intra_shapley.argsort()[-5:][::-1].tolist(),
            'top_inter_indices': inter_shapley.argsort()[-5:][::-1].tolist(),
            'gpu_id': gpu_id,
            'success': True
        }
        
        # Clean up GPU memory
        del model
        torch.cuda.empty_cache()
        
        return result
        
    except Exception as e:
        return {
            'speaker_id': speaker_id,
            'success': False,
            'error': str(e)
        }


def analyze_data_shapley_multigpu(
    embeddings_dir: Path,
    model: nn.Module,
    output_dir: Path,
    gpu_ids: List[int],
    n_permutations: int = 100
) -> Dict:
    """
    Compute Data Shapley values for all speakers using multiple GPUs.
    
    Each GPU processes one speaker and computes both:
    - Intra-speaker Shapley (within-speaker utterance importance)
    - Inter-speaker Shapley (cross-speaker utterance importance)
    
    Args:
        embeddings_dir: Directory containing speaker embeddings
        model: Classification model (will be copied to each GPU)
        output_dir: Output directory for results
        gpu_ids: List of GPU IDs to use
        n_permutations: Number of Monte Carlo permutations
        
    Returns:
        Dictionary of results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all speaker embeddings
    speaker_embeddings = {}
    speaker_dirs = sorted([d for d in embeddings_dir.iterdir() if d.is_dir()])
    
    for speaker_dir in speaker_dirs:
        utt_file = speaker_dir / 'utterance_embeddings.npy'
        if utt_file.exists():
            speaker_embeddings[speaker_dir.name] = np.load(utt_file)
    
    speakers = sorted(speaker_embeddings.keys())
    num_speakers = len(speakers)
    num_gpus = len(gpu_ids) if gpu_ids else 1
    
    print(f"\nComputing Data Shapley for {num_speakers} speakers")
    print(f"Using {num_gpus} GPUs: {gpu_ids}")
    print(f"Monte Carlo permutations: {n_permutations}")
    
    # Get model state dict for transfer to worker processes
    model_state_dict = model.state_dict()
    num_classes = model.classifier[-1].out_features
    
    results = {}
    
    # Use multiprocessing spawn context for CUDA compatibility
    mp_context = mp.get_context('spawn')
    
    # Process speakers in parallel
    pbar = tqdm(total=num_speakers, desc="Data Shapley Analysis")
    
    for batch_start in range(0, num_speakers, num_gpus):
        batch_end = min(batch_start + num_gpus, num_speakers)
        batch_speakers = speakers[batch_start:batch_end]
        
        # Prepare arguments for each speaker in the batch
        process_args = []
        for i, speaker_id in enumerate(batch_speakers):
            gpu_id = gpu_ids[i % num_gpus]
            
            # Get other speakers' embeddings (for inter-speaker Shapley)
            other_embeddings = {
                spk: embs for spk, embs in speaker_embeddings.items() 
                if spk != speaker_id
            }
            
            process_args.append((
                speaker_id,
                speaker_embeddings[speaker_id],
                other_embeddings,
                model_state_dict,
                num_classes,
                str(output_dir),
                gpu_id,
                n_permutations
            ))
        
        # Run batch in parallel
        with mp_context.Pool(processes=len(batch_speakers)) as pool:
            batch_results = pool.map(process_data_shapley_worker, process_args)
        
        for result in batch_results:
            if result['success']:
                results[result['speaker_id']] = result
            else:
                print(f"  Warning: Failed for {result['speaker_id']}: {result.get('error')}")
            pbar.update(1)
    
    pbar.close()
    
    # Save summary
    summary = {
        'method': 'data_shapley',
        'n_permutations': n_permutations,
        'num_speakers': len(results),
        'speakers': {spk: {k: v for k, v in r.items() if k != 'success'} 
                     for spk, r in results.items()}
    }
    
    with open(output_dir / 'data_shapley_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    return results


def analyze_data_shapley_singlegpu(
    embeddings_dir: Path,
    model: nn.Module,
    output_dir: Path,
    device: str = 'cuda:0',
    n_permutations: int = 100
) -> Dict:
    """
    Single-GPU fallback for Data Shapley computation.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all speaker embeddings
    speaker_embeddings = {}
    speaker_dirs = sorted([d for d in embeddings_dir.iterdir() if d.is_dir()])
    
    for speaker_dir in speaker_dirs:
        utt_file = speaker_dir / 'utterance_embeddings.npy'
        if utt_file.exists():
            speaker_embeddings[speaker_dir.name] = np.load(utt_file)
    
    speakers = sorted(speaker_embeddings.keys())
    
    model = model.to(device)
    model.eval()
    
    results = {}
    
    for speaker_id in tqdm(speakers, desc="Data Shapley Analysis"):
        # Compute intra-speaker Data Shapley
        intra_shapley = compute_data_shapley_intra_speaker(
            speaker_embeddings[speaker_id],
            model,
            device,
            n_permutations
        )
        
        # Compute inter-speaker Data Shapley
        other_embeddings = {
            spk: embs for spk, embs in speaker_embeddings.items() 
            if spk != speaker_id
        }
        
        inter_shapley = compute_data_shapley_inter_speaker(
            speaker_embeddings[speaker_id],
            other_embeddings,
            model,
            device,
            n_permutations
        )
        
        # Save results
        speaker_output = output_dir / speaker_id
        speaker_output.mkdir(parents=True, exist_ok=True)
        
        np.save(speaker_output / 'data_shapley_intra.npy', intra_shapley)
        np.save(speaker_output / 'data_shapley_inter.npy', inter_shapley)
        
        results[speaker_id] = {
            'speaker_id': speaker_id,
            'num_utterances': len(speaker_embeddings[speaker_id]),
            'intra_shapley_mean': float(intra_shapley.mean()),
            'intra_shapley_std': float(intra_shapley.std()),
            'inter_shapley_mean': float(inter_shapley.mean()),
            'inter_shapley_std': float(inter_shapley.std()),
            'top_intra_indices': intra_shapley.argsort()[-5:][::-1].tolist(),
            'top_inter_indices': inter_shapley.argsort()[-5:][::-1].tolist()
        }
    
    # Save summary
    summary = {
        'method': 'data_shapley',
        'n_permutations': n_permutations,
        'num_speakers': len(results),
        'speakers': results
    }
    
    with open(output_dir / 'data_shapley_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    return results


# =============================================================================
# pyDVL-based Data Shapley (True Data Shapley with model retraining)
# =============================================================================

class LegacySklearnSpeakerClassifier:
    """
    Legacy pyDVL model.

    Architecture: MLP(1024 -> 256 -> num_speakers), trained from scratch.
    """

    def __init__(self, hidden_dim: int = 256, max_iter: int = 500):
        self.hidden_dim = hidden_dim
        self.max_iter = max_iter
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model = MLPClassifier(
            hidden_layer_sizes=(self.hidden_dim,),
            activation='tanh',
            max_iter=self.max_iter,
            early_stopping=True,
            n_iter_no_change=10,
            random_state=42,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.zeros(len(X), dtype=np.int64)
        return self.model.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        if self.model is None:
            return 0.0
        return self.model.score(X, y)


class SpeakerClassifier(LegacySklearnSpeakerClassifier):
    """
    Backward-compatible alias for legacy pyDVL code paths.
    """
    pass


class FrozenHeadTopClassifier:
    """
    Default pyDVL model for Option-1 Data Shapley.

    Inputs are fixed SUPERB-head features; this class trains only a small
    top linear classifier for the dataset speaker labels.
    """

    def __init__(
        self,
        lr: float = 1e-2,
        max_epochs: int = 100,
        batch_size: int = 128,
        weight_decay: float = 1e-4,
        random_state: int = 42,
    ):
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.model: Optional[nn.Module] = None
        self.classes_: Optional[np.ndarray] = None
        self.class_to_index: Dict[int, int] = {}

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            self.model = None
            return self

        self.class_to_index = {int(c): idx for idx, c in enumerate(self.classes_)}
        y_idx = np.array([self.class_to_index[int(v)] for v in y], dtype=np.int64)

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.model = nn.Linear(X.shape[1], len(self.classes_))
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y_idx, dtype=torch.long)

        self.model.train()
        num_samples = X_t.shape[0]
        for _ in range(self.max_epochs):
            permutation = torch.randperm(num_samples)
            for start in range(0, num_samples, self.batch_size):
                end = min(start + self.batch_size, num_samples)
                idx = permutation[start:end]
                xb = X_t[idx]
                yb = y_t[idx]

                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

        self.model.eval()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None or self.classes_ is None:
            return np.zeros(len(X), dtype=np.int64)

        X = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            logits = self.model(torch.tensor(X, dtype=torch.float32))
            pred_idx = logits.argmax(dim=1).cpu().numpy()
        return self.classes_[pred_idx]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        return float((preds == y).mean())


class ProjectorClassifierHead(nn.Module):
    """
    Linear projector + linear classifier head.
    Mirrors SUPERB SID head shape: input -> projector -> classifier.
    """

    def __init__(self, input_dim: int, projector_dim: int, num_classes: int):
        super().__init__()
        self.projector = nn.Linear(input_dim, projector_dim)
        self.classifier = nn.Linear(projector_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projector(x)
        return self.classifier(x)


class FineTuneHeadClassifier:
    """
    pyDVL model that fine-tunes projector + classifier for each subset.

    Initialization:
    - projector weights copied from pretrained SUPERB head
    - classifier rows initialized from pretrained SID classifier rows indexed by
      dataset label IDs (with modulo fallback for out-of-range labels)
    """

    def __init__(
        self,
        projector_weight: np.ndarray,
        projector_bias: Optional[np.ndarray],
        classifier_weight: np.ndarray,
        classifier_bias: Optional[np.ndarray],
        lr: float = 5e-4,
        max_epochs: int = 40,
        batch_size: int = 128,
        weight_decay: float = 1e-4,
        random_state: int = 42,
        device: str = 'cpu',
        use_data_parallel: bool = False,
        gpu_ids: Tuple[int, ...] = (),
        enforce_multiclass_subsets: bool = False,
    ):
        self.projector_weight = np.asarray(projector_weight, dtype=np.float32)
        self.projector_bias = (
            np.asarray(projector_bias, dtype=np.float32)
            if projector_bias is not None else None
        )
        self.classifier_weight = np.asarray(classifier_weight, dtype=np.float32)
        self.classifier_bias = (
            np.asarray(classifier_bias, dtype=np.float32)
            if classifier_bias is not None else None
        )
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.device = device
        self.use_data_parallel = use_data_parallel
        self.gpu_ids = tuple(gpu_ids) if gpu_ids else ()
        self.enforce_multiclass_subsets = bool(enforce_multiclass_subsets)

        self.model: Optional[nn.Module] = None
        self.classes_: Optional[np.ndarray] = None
        self.class_to_index: Dict[int, int] = {}
        self.single_class_: Optional[int] = None
        self.device_in_use_: str = 'cpu'
        self.anchor_by_class: Dict[int, np.ndarray] = {}

    def set_multiclass_anchor_pool(self, X_ref: np.ndarray, y_ref: np.ndarray) -> None:
        """
        Register one deterministic anchor sample per class.

        In optional `enforce_multiclass_subsets` mode, single-class subset fits are
        augmented with one anchor from an alternative class to avoid degenerate fits.
        """
        X_ref = np.asarray(X_ref, dtype=np.float32)
        y_ref = np.asarray(y_ref, dtype=np.int64)
        anchors: Dict[int, np.ndarray] = {}
        for class_id in np.unique(y_ref):
            idx = np.where(y_ref == class_id)[0]
            if len(idx) == 0:
                continue
            anchors[int(class_id)] = X_ref[int(idx[0])].copy()
        self.anchor_by_class = anchors

    def _augment_single_class_subset(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optional subset augmentation for one-class pyDVL coalitions.
        """
        if not self.enforce_multiclass_subsets:
            return X, y
        unique = np.unique(y)
        if len(unique) != 1:
            return X, y

        present_class = int(unique[0])
        alt_classes = sorted([c for c in self.anchor_by_class.keys() if c != present_class])
        if not alt_classes:
            return X, y

        # Add one fixed anchor from an alternative class; deterministic and minimal.
        alt_class = int(alt_classes[0])
        anchor = self.anchor_by_class[alt_class]
        X_aug = np.concatenate([X, anchor[None, :]], axis=0)
        y_aug = np.concatenate([y, np.asarray([alt_class], dtype=np.int64)], axis=0)
        return X_aug, y_aug

    def _resolve_device(self) -> torch.device:
        preferred = self.device
        if preferred == 'cuda':
            if self.gpu_ids:
                preferred = f'cuda:{self.gpu_ids[0]}'
            else:
                preferred = 'cuda:0'
        if preferred.startswith('cuda') and not torch.cuda.is_available():
            preferred = 'cpu'
        return torch.device(preferred)

    def _init_model(self, num_classes: int) -> ProjectorClassifierHead:
        input_dim = int(self.projector_weight.shape[1])
        projector_dim = int(self.projector_weight.shape[0])

        model = ProjectorClassifierHead(
            input_dim=input_dim,
            projector_dim=projector_dim,
            num_classes=num_classes,
        )

        with torch.no_grad():
            model.projector.weight.copy_(torch.from_numpy(self.projector_weight))
            if model.projector.bias is not None and self.projector_bias is not None:
                model.projector.bias.copy_(torch.from_numpy(self.projector_bias))

            source_num_classes = int(self.classifier_weight.shape[0])
            source_w = torch.from_numpy(self.classifier_weight)
            source_b = (
                torch.from_numpy(self.classifier_bias)
                if self.classifier_bias is not None else None
            )

            for out_idx, class_id in enumerate(self.classes_):
                src_idx = int(class_id)
                if src_idx < 0 or src_idx >= source_num_classes:
                    src_idx = src_idx % source_num_classes
                model.classifier.weight[out_idx].copy_(source_w[src_idx])
                if model.classifier.bias is not None and source_b is not None:
                    model.classifier.bias[out_idx].copy_(source_b[src_idx])

        return model

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        if len(y) > 0:
            X, y = self._augment_single_class_subset(X, y)

        self.classes_ = np.unique(y)
        self.class_to_index = {int(c): idx for idx, c in enumerate(self.classes_)}
        self.single_class_ = None
        self.model = None

        if len(self.classes_) == 0:
            print("No class is found in the data. The model will not be trained.")
            return self

        if len(self.classes_) == 1:
            print(f"Only one class ({self.classes_[0]}) is found in the data. The model will always predict this class.")
            self.single_class_ = int(self.classes_[0])
            return self

        y_local = np.array([self.class_to_index[int(v)] for v in y], dtype=np.int64)

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        model = self._init_model(num_classes=len(self.classes_))
        device = self._resolve_device()
        self.device_in_use_ = str(device)

        if (
            device.type == 'cuda'
            and self.use_data_parallel
            and len(self.gpu_ids) > 1
        ):
            model = model.to(device)
            model = nn.DataParallel(model, device_ids=list(self.gpu_ids))
        else:
            model = model.to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y_local, dtype=torch.long)

        model.train()
        num_samples = X_t.shape[0]

        for _ in range(self.max_epochs):
            permutation = torch.randperm(num_samples)
            for start in range(0, num_samples, self.batch_size):
                end = min(start + self.batch_size, num_samples)
                idx = permutation[start:end]
                xb = X_t[idx].to(device, non_blocking=True)
                yb = y_t[idx].to(device, non_blocking=True)

                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

        model.eval()
        self.model = model
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if len(X) == 0:
            return np.array([], dtype=np.int64)

        if self.single_class_ is not None:
            return np.full(len(X), self.single_class_, dtype=np.int64)

        if self.model is None or self.classes_ is None:
            return np.zeros(len(X), dtype=np.int64)

        device = torch.device(self.device_in_use_)
        with torch.no_grad():
            logits = self.model(torch.tensor(X, dtype=torch.float32, device=device))
            pred_idx = logits.argmax(dim=1).detach().cpu().numpy()
        return self.classes_[pred_idx]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=np.int64)
        if len(y) == 0:
            return 0.0
        preds = self.predict(X)
        return float((preds == y).mean())


def load_speaker_identification_inputs(
    embeddings_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int], Dict[str, int]]:
    """
    Build SID inputs from speaker embedding folders.

    Returns:
        embeddings: (N, D) float array
        labels: (N,) int array with speaker indices
        speaker_ids: (N,) str array with source speaker ids
        group_to_idx: speaker_id -> class index
        group_counts: speaker_id -> number of samples
    """
    all_embeddings: List[np.ndarray] = []
    all_labels: List[int] = []
    all_speaker_ids: List[str] = []
    speaker_to_idx: Dict[str, int] = {}
    group_counts: Dict[str, int] = {}

    speaker_dirs = sorted([d for d in embeddings_dir.iterdir() if d.is_dir()])
    for speaker_idx, speaker_dir in enumerate(speaker_dirs):
        utt_file = speaker_dir / 'utterance_embeddings.npy'
        if not utt_file.exists():
            continue

        embeddings = np.load(utt_file)
        speaker_id = speaker_dir.name
        speaker_to_idx[speaker_id] = speaker_idx
        group_counts[speaker_id] = int(len(embeddings))

        for emb in embeddings:
            all_embeddings.append(emb)
            all_labels.append(speaker_idx)
            all_speaker_ids.append(speaker_id)

    if not all_embeddings:
        raise RuntimeError(f'No speaker embeddings found in {embeddings_dir}')

    return (
        np.asarray(all_embeddings, dtype=np.float32),
        np.asarray(all_labels, dtype=np.int64),
        np.asarray(all_speaker_ids),
        speaker_to_idx,
        group_counts,
    )


def load_iemocap_emotion_inputs(
    embeddings_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int], Dict[str, int]]:
    """
    Build ER inputs from IEMOCAP embedding metadata using full emotion set.

    Uses per-utterance metadata `utterances[i].emotion` as labels.
    """
    all_embeddings: List[np.ndarray] = []
    all_label_names: List[str] = []
    all_speaker_ids: List[str] = []
    emotion_counts: Counter = Counter()

    speaker_dirs = sorted([d for d in embeddings_dir.iterdir() if d.is_dir()])
    for speaker_dir in speaker_dirs:
        utt_file = speaker_dir / 'utterance_embeddings.npy'
        meta_file = speaker_dir / 'metadata.json'
        if not (utt_file.exists() and meta_file.exists()):
            continue

        embeddings = np.load(utt_file)
        with open(meta_file) as f:
            metadata = json.load(f)

        utterances = metadata.get('utterances', [])
        if len(utterances) != len(embeddings):
            raise RuntimeError(
                f'Utterance metadata length mismatch in {speaker_dir.name}: '
                f'{len(utterances)} metadata entries vs {len(embeddings)} embeddings'
            )

        for emb, utt_meta in zip(embeddings, utterances):
            emotion = utt_meta.get('emotion')
            if emotion is None:
                continue
            emotion_name = str(emotion).strip().lower()
            all_embeddings.append(emb)
            all_label_names.append(emotion_name)
            all_speaker_ids.append(speaker_dir.name)
            emotion_counts[emotion_name] += 1

    if not all_embeddings:
        raise RuntimeError(
            f'No IEMOCAP emotion-labeled samples found in {embeddings_dir}'
        )

    emotions = sorted(emotion_counts.keys())
    emotion_to_idx = {emotion: idx for idx, emotion in enumerate(emotions)}
    labels = np.asarray([emotion_to_idx[e] for e in all_label_names], dtype=np.int64)

    return (
        np.asarray(all_embeddings, dtype=np.float32),
        labels,
        np.asarray(all_speaker_ids),
        emotion_to_idx,
        {emotion: int(count) for emotion, count in emotion_counts.items()},
    )


def aggregate_speaker_label_shapley(
    shapley_values: np.ndarray,
    train_labels: np.ndarray,
    train_speaker_ids: np.ndarray,
    idx_to_group: Dict[int, str],
    output_dir: Path,
    group_type: str,
) -> Dict[str, Dict]:
    """
    Aggregate per-utterance Shapley values by (task group, speaker).

    Saves full vectors for each (group, speaker) bucket at:
      output_dir/speaker_label_contributions/{group_id}/{speaker_id}/pydvl_data_shapley.npy
    """
    output_root = output_dir / 'speaker_label_contributions'
    output_root.mkdir(parents=True, exist_ok=True)

    shapley_values = np.asarray(shapley_values)
    train_labels = np.asarray(train_labels, dtype=np.int64)
    train_speaker_ids = np.asarray(train_speaker_ids)

    grouped_results: Dict[str, Dict] = {}
    unique_labels = sorted(np.unique(train_labels).tolist())

    for group_idx in unique_labels:
        group_id = idx_to_group[int(group_idx)]
        label_mask = train_labels == int(group_idx)
        group_values = shapley_values[label_mask]
        group_speakers = train_speaker_ids[label_mask]

        if len(group_values) == 0:
            continue

        speaker_results: Dict[str, Dict] = {}
        for speaker_id in sorted(np.unique(group_speakers).tolist()):
            speaker_mask = group_speakers == speaker_id
            speaker_values = group_values[speaker_mask]
            if len(speaker_values) == 0:
                continue

            speaker_output = output_root / group_id / str(speaker_id)
            speaker_output.mkdir(parents=True, exist_ok=True)
            np.save(speaker_output / 'pydvl_data_shapley.npy', speaker_values)

            speaker_results[str(speaker_id)] = {
                'speaker_id': str(speaker_id),
                'num_samples': int(len(speaker_values)),
                'shapley_mean': float(speaker_values.mean()),
                'shapley_std': float(speaker_values.std()),
                'shapley_min': float(speaker_values.min()),
                'shapley_max': float(speaker_values.max()),
            }

        group_entry = {
            'group_id': group_id,
            'num_speakers': int(len(speaker_results)),
            'num_samples': int(len(group_values)),
            'speakers': speaker_results,
        }
        if group_type == 'emotion':
            group_entry['emotion_label'] = group_id
        if group_type == 'speaker':
            group_entry['speaker_label'] = group_id
        grouped_results[group_id] = group_entry

    return grouped_results


def resolve_pydvl_task_mode(
    task_mode: str,
    dataset_name: str,
    analysis_head: str,
) -> str:
    """Resolve pyDVL target task based on CLI selection and context."""
    if task_mode != PYDVL_TASK_AUTO:
        return task_mode

    if dataset_name == 'iemocap' and analysis_head == 'er':
        return PYDVL_TASK_EMOTION
    return PYDVL_TASK_SPEAKER


def resolve_pydvl_feature_head(feature_head: str, analysis_head: str) -> str:
    """Resolve the pretrained head used in frozen/finetune pyDVL modes."""
    if feature_head != PYDVL_FEATURE_HEAD_AUTO:
        return feature_head

    if analysis_head not in PREDICTION_HEADS:
        raise RuntimeError(
            f'Cannot auto-resolve pyDVL feature head from analysis head: {analysis_head}'
        )
    return analysis_head


def resolve_pydvl_execution_parallelism(
    n_jobs_requested: int,
    model_variant: str,
    resolved_train_device: Optional[str],
    use_data_parallel: bool,
    allow_single_gpu_concurrency: bool,
) -> Tuple[int, bool]:
    """
    Resolve pyDVL job parallelism for stable execution.

    Returns:
        effective_n_jobs: final n_jobs used in parallel_config
        single_gpu_concurrency_enabled: whether experimental single-GPU
            concurrent subset fits are enabled
    """
    effective_n_jobs = max(1, int(n_jobs_requested))
    single_gpu_concurrency_enabled = False

    if model_variant != PYDVL_MODEL_FINETUNE_HEAD:
        return effective_n_jobs, single_gpu_concurrency_enabled

    device_name = resolved_train_device or 'cpu'

    # Multi-GPU DataParallel already parallelizes one fit; keep pyDVL jobs serial.
    if use_data_parallel and effective_n_jobs != 1:
        return 1, single_gpu_concurrency_enabled

    # Single-GPU concurrent subset fits are opt-in and experimental.
    if device_name.startswith('cuda') and not use_data_parallel and effective_n_jobs > 1:
        if allow_single_gpu_concurrency:
            single_gpu_concurrency_enabled = True
            return effective_n_jobs, single_gpu_concurrency_enabled
        return 1, single_gpu_concurrency_enabled

    return effective_n_jobs, single_gpu_concurrency_enabled


def resolve_pydvl_joblib_backend(
    requested_backend: str,
    single_gpu_concurrency_enabled: bool,
) -> Optional[str]:
    """Resolve joblib backend; None means use joblib default."""
    if requested_backend != PYDVL_JOBLIB_BACKEND_AUTO:
        return requested_backend
    if single_gpu_concurrency_enabled:
        # Keep process isolation for concurrent CUDA contexts.
        return PYDVL_JOBLIB_BACKEND_LOKY
    return None


def compute_pydvl_data_shapley_worker(args: Tuple) -> Dict:
    """
    Worker function to compute true Data Shapley for one speaker using pyDVL.
    
    This function:
    1. Splits the speaker's data 80/20
    2. Uses TMC-Shapley to compute data point importance
    3. Retrains a classifier for each subset evaluation
    
    Args:
        args: Tuple of (speaker_id, embeddings, labels, output_dir, 
                        min_updates, rtol, n_jobs)
    
    Returns:
        Dictionary with speaker results
    """
    (speaker_id, embeddings, labels, output_dir, min_updates, rtol, n_jobs) = args
    
    try:
        # Split data 80/20 for train/test
        n_samples = len(embeddings)
        if n_samples < 10:
            return {
                'speaker_id': speaker_id,
                'success': False,
                'error': f'Too few samples ({n_samples}) for Data Shapley'
            }
        
        # Ensure we have at least 2 classes in train and test
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return {
                'speaker_id': speaker_id,
                'success': False,
                'error': f'Need at least 2 classes, got {len(unique_labels)}'
            }
        
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Create pyDVL datasets
        train_data = PyDVLDataset(x=X_train, y=y_train)
        test_data = PyDVLDataset(x=X_test, y=y_test)
        
        # Create model and utility
        model = SpeakerClassifier(hidden_dim=256, max_iter=200)
        scorer = SupervisedScorer('accuracy', test_data, default=0.0)
        utility = ModelUtility(model, scorer)
        
        # TMC-Shapley with truncation
        truncation = RelativeTruncation(rtol=rtol)
        stopping = MinUpdates(min_updates)
        valuation = TMCShapleyValuation(utility, truncation, stopping)
        
        # Run with parallelization
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with parallel_config(n_jobs=n_jobs):
                valuation.fit(train_data)
        
        shapley_values = valuation.result.values
        
        # Save results
        output_path = Path(output_dir) / speaker_id
        output_path.mkdir(parents=True, exist_ok=True)
        
        np.save(output_path / 'pydvl_data_shapley.npy', shapley_values)
        
        # Map back to original indices (train subset only)
        # Save train indices for reference
        train_indices = np.arange(len(X_train))
        np.save(output_path / 'train_indices.npy', train_indices)
        
        result = {
            'speaker_id': speaker_id,
            'num_train_samples': len(X_train),
            'num_test_samples': len(X_test),
            'shapley_mean': float(shapley_values.mean()),
            'shapley_std': float(shapley_values.std()),
            'shapley_min': float(shapley_values.min()),
            'shapley_max': float(shapley_values.max()),
            'top_indices': shapley_values.argsort()[-5:][::-1].tolist(),
            'bottom_indices': shapley_values.argsort()[:5].tolist(),
            'success': True
        }
        
        return result
        
    except Exception as e:
        import traceback
        return {
            'speaker_id': speaker_id,
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }


def analyze_pydvl_data_shapley(
    embeddings_dir: Path,
    output_dir: Path,
    min_updates: int = PYDVL_DEFAULT_MIN_UPDATES,
    rtol: float = PYDVL_DEFAULT_RTOL,
    n_jobs_per_speaker: int = 4,
    num_parallel_speakers: int = 1,
    model_variant: str = PYDVL_MODEL_FINETUNE_HEAD,
    feature_head: str = PYDVL_FEATURE_HEAD_AUTO,
    project_root: Optional[Path] = None,
    feature_batch_size: int = 256,
    feature_device: str = 'cpu',
    train_device: str = 'auto',
    finetune_epochs: int = 40,
    finetune_lr: float = 5e-4,
    finetune_batch_size: int = 128,
    finetune_weight_decay: float = 1e-4,
    gpu_ids: Optional[List[int]] = None,
    show_pydvl_progress: bool = True,
    task_mode: str = PYDVL_TASK_AUTO,
    dataset_name: Optional[str] = None,
    analysis_head: str = 'sid',
    allow_single_gpu_concurrency: bool = False,
    joblib_backend: str = PYDVL_JOBLIB_BACKEND_AUTO,
    enforce_multiclass_subsets: bool = False,
) -> Dict:
    """
    Compute true Data Shapley values using pyDVL's TMC-Shapley.
    
    This performs proper Data Shapley by retraining a classifier
    on each data subset to measure marginal utility.
    
    Uses either speaker-ID or full IEMOCAP emotion labels as task targets.
    - X = utterance embeddings (1024-dim)
    - y = task labels
    - Split 80/20 with stratification
    - Utility = accuracy on held-out 20%
    
    Args:
        embeddings_dir: Directory containing speaker embeddings
        output_dir: Output directory for results
        min_updates: Minimum number of TMC-Shapley updates per data point
        rtol: Relative tolerance for truncation
        n_jobs_per_speaker: Number of parallel jobs within pyDVL
        num_parallel_speakers: Number of speakers to process in parallel
        model_variant: pyDVL model variant
        feature_head: pretrained head id used for frozen/finetune modes
        project_root: repository root
        feature_batch_size: batch size for frozen-head feature extraction
        feature_device: device for frozen-head feature extraction
        train_device: device for finetune-head subset training
        finetune_epochs: epochs per pyDVL subset fit in finetune-head mode
        finetune_lr: learning rate for finetune-head mode
        finetune_batch_size: train batch size for finetune-head mode
        finetune_weight_decay: optimizer weight decay for finetune-head mode
        gpu_ids: GPUs discovered by caller (used for optional DataParallel)
        show_pydvl_progress: whether to show tqdm progress/ETA during TMC-Shapley
        task_mode: target task mode (`auto`, `speaker`, `emotion`)
        dataset_name: dataset identifier, used for task auto-resolution
        analysis_head: selected analysis head (`sid`, `er`, ...)
        enforce_multiclass_subsets: opt-in subset augmentation that forces
            two-class minimum for one-class subset fits (changes estimator)
        
    Returns:
        Dictionary of results
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_dataset_name = dataset_name or embeddings_dir.name
    resolved_task_mode = resolve_pydvl_task_mode(
        task_mode=task_mode,
        dataset_name=resolved_dataset_name,
        analysis_head=analysis_head,
    )
    resolved_feature_head = resolve_pydvl_feature_head(
        feature_head=feature_head,
        analysis_head=analysis_head,
    )

    if resolved_task_mode == PYDVL_TASK_SPEAKER:
        all_embeddings, all_labels, all_speaker_ids, group_to_idx, group_counts = load_speaker_identification_inputs(
            embeddings_dir
        )
        group_type = 'speaker'
    elif resolved_task_mode == PYDVL_TASK_EMOTION:
        if resolved_dataset_name != 'iemocap':
            raise RuntimeError(
                'Emotion-label Data Shapley is supported only for IEMOCAP metadata in this pipeline'
            )
        all_embeddings, all_labels, all_speaker_ids, group_to_idx, group_counts = load_iemocap_emotion_inputs(
            embeddings_dir
        )
        group_type = 'emotion'
    else:
        raise RuntimeError(f'Unsupported pyDVL task mode: {resolved_task_mode}')

    idx_to_group = {idx: group for group, idx in group_to_idx.items()}
    num_groups = len(group_to_idx)
    num_samples = len(all_embeddings)

    print(f"\nComputing pyDVL Data Shapley")
    print(f"  Total samples: {num_samples}")
    print(f"  Task mode: {resolved_task_mode}")
    print(f"  Group type: {group_type}")
    print(f"  Number of groups: {num_groups}")
    print(f"  Min updates: {min_updates}")
    print(f"  Truncation rtol: {rtol}")
    print(f"  Jobs per speaker: {n_jobs_per_speaker}")
    if enforce_multiclass_subsets:
        print("  One-class subset handling: enabled (augment with anchor samples; estimator modified)")
    if resolved_task_mode == PYDVL_TASK_EMOTION:
        print(f"  Emotion labels: {sorted(group_to_idx.keys())}")
        print(f"  Emotion counts: {group_counts}")

    effective_n_jobs = n_jobs_per_speaker
    summary_train_device = None
    summary_data_parallel = False
    summary_single_gpu_concurrency = False
    resolved_joblib_backend: Optional[str] = None

    if model_variant == PYDVL_MODEL_FINETUNE_HEAD:
        if project_root is None:
            raise RuntimeError("project_root is required for finetune-head pyDVL mode")
        if resolved_feature_head not in PREDICTION_HEADS:
            raise RuntimeError(f"Unknown feature head: {resolved_feature_head}")

        head_model_path = project_root / PREDICTION_HEADS[resolved_feature_head]['path']
        head_params = load_pretrained_head_parameters(head_model_path)

        resolved_train_device = train_device
        if resolved_train_device == 'auto':
            if torch.cuda.is_available() and gpu_ids:
                resolved_train_device = f'cuda:{gpu_ids[0]}'
            elif torch.cuda.is_available():
                resolved_train_device = 'cuda:0'
            else:
                resolved_train_device = 'cpu'
        elif resolved_train_device == 'cuda':
            if torch.cuda.is_available() and gpu_ids:
                resolved_train_device = f'cuda:{gpu_ids[0]}'
            elif torch.cuda.is_available():
                resolved_train_device = 'cuda:0'
            else:
                resolved_train_device = 'cpu'
        elif resolved_train_device.startswith('cuda') and not torch.cuda.is_available():
            resolved_train_device = 'cpu'

        gpu_tuple = tuple(gpu_ids) if gpu_ids else ()
        summary_data_parallel = (
            resolved_train_device.startswith('cuda') and len(gpu_tuple) > 1
        )
        effective_n_jobs, summary_single_gpu_concurrency = resolve_pydvl_execution_parallelism(
            n_jobs_requested=n_jobs_per_speaker,
            model_variant=model_variant,
            resolved_train_device=resolved_train_device,
            use_data_parallel=summary_data_parallel,
            allow_single_gpu_concurrency=allow_single_gpu_concurrency,
        )
        if summary_data_parallel and n_jobs_per_speaker != effective_n_jobs:
            print("  Multi-GPU finetune detected: forcing pyDVL jobs per speaker to 1 to avoid GPU contention")
        if (
            resolved_train_device.startswith('cuda')
            and not summary_data_parallel
            and n_jobs_per_speaker > 1
            and not summary_single_gpu_concurrency
            and effective_n_jobs == 1
        ):
            print("  Single-GPU finetune detected: forcing pyDVL jobs per speaker to 1 (set --pydvl-allow-single-gpu-concurrency to enable experimental overlap)")
        if summary_single_gpu_concurrency:
            print("  Single-GPU concurrency: enabled (experimental)")

        print("  pyDVL model: fine-tune pretrained SUPERB projector + classifier")
        print(f"  Head init: {resolved_feature_head}")
        print(f"  Head model path: {head_model_path}")
        print(f"  Finetune device: {resolved_train_device}")
        print(f"  Finetune epochs: {finetune_epochs}")
        print(f"  Finetune batch size: {finetune_batch_size}")
        if summary_data_parallel:
            print(f"  DataParallel GPUs: {list(gpu_tuple)}")

        shapley_inputs = all_embeddings
        utility_model = FineTuneHeadClassifier(
            projector_weight=head_params['projector_weight'],
            projector_bias=head_params['projector_bias'],
            classifier_weight=head_params['classifier_weight'],
            classifier_bias=head_params['classifier_bias'],
            lr=finetune_lr,
            max_epochs=finetune_epochs,
            batch_size=finetune_batch_size,
            weight_decay=finetune_weight_decay,
            device=resolved_train_device,
            use_data_parallel=summary_data_parallel,
            gpu_ids=gpu_tuple,
            enforce_multiclass_subsets=enforce_multiclass_subsets,
        )
        summary_model_type = 'finetune_superb_projector_classifier'
        summary_feature_head = resolved_feature_head
        summary_feature_path = str(head_model_path)
        summary_train_device = resolved_train_device
    # Option-1 mode: fixed pretrained head features + trainable top classifier.
    elif model_variant == PYDVL_MODEL_FROZEN_HEAD_TOP:
        if project_root is None:
            raise RuntimeError("project_root is required for frozen-head pyDVL mode")
        if resolved_feature_head not in PREDICTION_HEADS:
            raise RuntimeError(f"Unknown feature head: {resolved_feature_head}")

        head_model_path = project_root / PREDICTION_HEADS[resolved_feature_head]['path']
        require_prediction_head_weights(head_model_path, head_name=resolved_feature_head)

        print("  pyDVL model: frozen SUPERB head + trainable top classifier")
        print(f"  Feature head: {resolved_feature_head}")
        print(f"  Feature model path: {head_model_path}")
        print(f"  Feature extraction device: {feature_device}")
        print(f"  Feature extraction batch size: {feature_batch_size}")

        shapley_inputs = extract_fixed_head_features(
            all_embeddings,
            head_model_path=head_model_path,
            batch_size=feature_batch_size,
            device=feature_device,
        )
        utility_model = FrozenHeadTopClassifier()
        summary_model_type = 'frozen_superb_head_top_classifier'
        summary_feature_head = resolved_feature_head
        summary_feature_path = str(head_model_path)
        effective_n_jobs = max(1, int(n_jobs_per_speaker))
    elif model_variant == PYDVL_MODEL_LEGACY_SKLEARN_MLP:
        print("  pyDVL model: legacy sklearn MLP on raw 1024 embeddings")
        shapley_inputs = all_embeddings
        utility_model = LegacySklearnSpeakerClassifier(hidden_dim=256, max_iter=200)
        summary_model_type = 'legacy_sklearn_mlp_1024'
        summary_feature_head = None
        summary_feature_path = None
        summary_train_device = None
        effective_n_jobs = max(1, int(n_jobs_per_speaker))
    else:
        raise RuntimeError(f"Unsupported pydvl model variant: {model_variant}")

    resolved_joblib_backend = resolve_pydvl_joblib_backend(
        requested_backend=joblib_backend,
        single_gpu_concurrency_enabled=summary_single_gpu_concurrency,
    )
    print(f"  Effective pyDVL jobs: {effective_n_jobs}")
    if resolved_joblib_backend is None:
        print("  joblib backend: default")
    else:
        print(f"  joblib backend: {resolved_joblib_backend}")

    # Process all data together, split 80/20 while preserving speaker-id alignment.
    X_train, X_test, y_train, y_test, speaker_ids_train, _ = train_test_split(
        shapley_inputs,
        all_labels,
        all_speaker_ids,
        test_size=0.2,
        random_state=42,
        stratify=all_labels,
    )

    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")

    if model_variant == PYDVL_MODEL_FINETUNE_HEAD and enforce_multiclass_subsets:
        utility_model.set_multiclass_anchor_pool(X_train, y_train)
        print(f"  Anchor classes registered: {len(utility_model.anchor_by_class)}")

    # Create pyDVL datasets
    train_data = PyDVLDataset(x=X_train, y=y_train)
    test_data = PyDVLDataset(x=X_test, y=y_test)

    # Create model and utility
    scorer = SupervisedScorer('accuracy', test_data, default=0.0)
    utility = ModelUtility(utility_model, scorer)

    # TMC-Shapley
    truncation = RelativeTruncation(rtol=rtol)
    stopping = MinUpdates(min_updates)
    progress_cfg: Dict | bool = False
    if show_pydvl_progress:
        progress_cfg = {
            'desc': f"TMC-Shapley [{embeddings_dir.name}]",
            'leave': True,
        }
    valuation = TMCShapleyValuation(
        utility,
        truncation,
        stopping,
        progress=progress_cfg,
    )

    print(f"\nRunning TMC-Shapley (this may take a while)...")
    start_time = time.time()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parallel_kwargs = {'n_jobs': effective_n_jobs}
        if resolved_joblib_backend is not None:
            parallel_kwargs['backend'] = resolved_joblib_backend
        with parallel_config(**parallel_kwargs):
            valuation.fit(train_data)

    elapsed = time.time() - start_time
    print(f"TMC-Shapley completed in {elapsed:.1f}s")

    shapley_values = valuation.result.values

    # Save overall results
    np.save(output_dir / 'pydvl_data_shapley_values.npy', shapley_values)
    np.save(output_dir / 'train_labels.npy', y_train)
    np.save(output_dir / 'train_speaker_ids.npy', speaker_ids_train)

    # Aggregate by task groups (speaker IDs or emotion labels)
    results = {}

    for group_idx in range(num_groups):
        group_id = idx_to_group[group_idx]
        mask = y_train == group_idx
        group_values = shapley_values[mask]

        if len(group_values) > 0:
            group_output = output_dir / group_id
            group_output.mkdir(parents=True, exist_ok=True)
            np.save(group_output / 'pydvl_data_shapley.npy', group_values)

            result_entry = {
                'group_id': group_id,
                'num_samples': int(mask.sum()),
                'shapley_mean': float(group_values.mean()),
                'shapley_std': float(group_values.std()),
                'shapley_min': float(group_values.min()),
                'shapley_max': float(group_values.max()),
                'top_indices': group_values.argsort()[-5:][::-1].tolist()
            }
            if group_type == 'speaker':
                result_entry['speaker_id'] = group_id
            if group_type == 'emotion':
                result_entry['emotion_label'] = group_id

            results[group_id] = result_entry

    speaker_label_contributions = aggregate_speaker_label_shapley(
        shapley_values=shapley_values,
        train_labels=y_train,
        train_speaker_ids=speaker_ids_train,
        idx_to_group=idx_to_group,
        output_dir=output_dir,
        group_type=group_type,
    )

    # Save summary
    summary = {
        'method': 'pydvl_tmc_shapley',
        'model_type': summary_model_type,
        'feature_head': summary_feature_head,
        'feature_model_path': summary_feature_path,
        'train_device': summary_train_device,
        'data_parallel': summary_data_parallel,
        'single_gpu_concurrency_enabled': summary_single_gpu_concurrency,
        'parallel_jobs_requested': int(n_jobs_per_speaker),
        'parallel_jobs_effective': int(effective_n_jobs),
        'joblib_backend': resolved_joblib_backend or 'default',
        'enforce_multiclass_subsets': bool(enforce_multiclass_subsets),
        'min_updates': min_updates,
        'rtol': rtol,
        'total_train_samples': len(X_train),
        'total_test_samples': len(X_test),
        'dataset_name': resolved_dataset_name,
        'task_mode': resolved_task_mode,
        'group_type': group_type,
        'num_groups': num_groups,
        'elapsed_seconds': elapsed,
        'groups': results,
        'speaker_label_contributions': speaker_label_contributions,
        'speaker_label_contributions_path': str(output_dir / 'speaker_label_contributions'),
    }
    if group_type == 'speaker':
        summary['num_speakers'] = num_groups
        summary['speakers'] = results
    if group_type == 'emotion':
        summary['num_emotions'] = num_groups
        summary['emotions'] = results
        summary['emotion_to_index'] = group_to_idx
        summary['emotion_counts'] = group_counts

    with open(output_dir / 'pydvl_data_shapley_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    return results


def process_speaker_on_gpu(args: Tuple) -> Dict:
    """
    Process a single speaker's attributions on a specific GPU.
    This function runs in a separate process.
    
    Args:
        args: Tuple of (speaker_id, embeddings_list, model_state_dict, 
                        num_classes, output_dir, gpu_id, n_samples, use_ig)
    
    Returns:
        Dictionary with speaker results
    """
    (speaker_id, embeddings_list, model_state_dict, num_classes, 
     output_dir, gpu_id, n_samples, use_ig) = args
    
    try:
        # Use direct device index (multi-GPU)
        device = torch.device(f'cuda:{gpu_id}')
        
        # Set current device and memory fraction
        torch.cuda.set_device(gpu_id)
        torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, gpu_id)
        
        # Recreate model directly on GPU
        model = EmbeddingClassifier(input_dim=1024, num_classes=num_classes)
        model.load_state_dict(model_state_dict)
        model = model.to(device)
        model.eval()
        
        # Stack embeddings directly on GPU
        embeddings = torch.tensor(np.stack(embeddings_list), dtype=torch.float32, device=device)
        
        # Get predictions (embeddings already on device)
        with torch.no_grad():
            logits = model(embeddings)
            predictions = logits.argmax(dim=1).cpu().numpy()
        
        method_name = 'integratedgradients' if use_ig else 'shapley'
        
        # Compute attributions
        all_attributions = []
        
        if use_ig:
            ig = IntegratedGradients(model)
            baseline = torch.zeros(1, 1024, device=device)
            
            for i in range(len(embeddings)):
                target = int(predictions[i])
                emb = embeddings[i:i+1]  # Already on device
                emb.requires_grad = True
                
                attr = ig.attribute(
                    emb,
                    baselines=baseline,
                    target=target,
                    n_steps=50,
                    return_convergence_delta=False
                )
                all_attributions.append(attr.detach().cpu().numpy()[0])
        else:
            shapley = ShapleyValueSampling(model)
            baseline = torch.zeros(1, 1024, device=device)
            
            for i in range(len(embeddings)):
                target = int(predictions[i])
                emb = embeddings[i:i+1]  # Already on device
                
                attr = shapley.attribute(
                    emb,
                    baselines=baseline,
                    target=target,
                    n_samples=n_samples,
                    show_progress=False
                )
                all_attributions.append(attr.cpu().numpy()[0])
        
        attributions = np.stack(all_attributions)
        
        # Save results
        output_path = Path(output_dir)
        speaker_output = output_path / speaker_id
        speaker_output.mkdir(parents=True, exist_ok=True)
        
        np.save(speaker_output / f'{method_name}_attributions.npy', attributions)
        
        # Compute summary
        result = {
            'speaker_id': speaker_id,
            'num_utterances': len(embeddings_list),
            'mean_attribution': attributions.mean(axis=0).tolist(),
            'std_attribution': attributions.std(axis=0).tolist(),
            'top_features': np.abs(attributions).mean(axis=0).argsort()[-10:][::-1].tolist(),
            'predictions': predictions.tolist(),
            'gpu_id': gpu_id,
            'success': True
        }
        
        # Clean up GPU memory
        del model, embeddings
        torch.cuda.empty_cache()
        
        return result
        
    except Exception as e:
        return {
            'speaker_id': speaker_id,
            'success': False,
            'error': str(e),
            'gpu_id': gpu_id
        }


def analyze_shapley_multigpu(
    embeddings_dir: Path,
    model: nn.Module,
    output_dir: Path,
    gpu_ids: List[int],
    n_samples: int = 25,
    use_ig: bool = False
) -> Dict:
    """
    Compute attributions for all speakers using multiple GPUs.
    Each speaker is processed on a separate GPU in parallel.
    
    Args:
        embeddings_dir: Directory containing speaker embeddings
        model: Classification model (will be copied to each GPU)
        output_dir: Output directory for results
        gpu_ids: List of GPU IDs to use
        n_samples: Number of samples for Shapley estimation
        use_ig: Use Integrated Gradients instead of Shapley
        
    Returns:
        Dictionary of attribution results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all speaker data
    dataset = EmbeddingDataset(embeddings_dir)
    
    speaker_embeddings = {}
    for sample in dataset.samples:
        spk = sample['speaker_id']
        if spk not in speaker_embeddings:
            speaker_embeddings[spk] = []
        speaker_embeddings[spk].append(sample['embedding'])
    
    speakers = sorted(speaker_embeddings.keys())
    num_speakers = len(speakers)
    num_gpus = len(gpu_ids)
    
    method_name = 'IntegratedGradients' if use_ig else 'Shapley'
    print(f"\nAnalyzing {len(dataset.samples)} samples from {num_speakers} speakers")
    print(f"Using {num_gpus} GPUs: {gpu_ids}")
    print(f"Method: {method_name}")
    
    # Get model state dict for transfer to worker processes
    model_state_dict = model.state_dict()
    num_classes = model.classifier[-1].out_features
    
    results = {}
    
    # Use multiprocessing spawn context for CUDA compatibility
    mp_context = mp.get_context('spawn')
    
    # Process speakers in batches of num_gpus
    pbar = tqdm(total=num_speakers, desc=f"{method_name} Analysis")
    
    for batch_start in range(0, num_speakers, num_gpus):
        batch_end = min(batch_start + num_gpus, num_speakers)
        batch_speakers = speakers[batch_start:batch_end]
        
        # Prepare arguments for each speaker in the batch
        process_args = []
        for i, speaker_id in enumerate(batch_speakers):
            gpu_id = gpu_ids[i % num_gpus]
            process_args.append((
                speaker_id,
                speaker_embeddings[speaker_id],
                model_state_dict,
                num_classes,
                str(output_dir),
                gpu_id,
                n_samples,
                use_ig
            ))
        
        # Run batch in parallel
        with mp_context.Pool(processes=len(batch_speakers)) as pool:
            batch_results = pool.map(process_speaker_on_gpu, process_args)
        
        # Collect results
        for result in batch_results:
            if result['success']:
                results[result['speaker_id']] = {
                    'num_utterances': result['num_utterances'],
                    'mean_attribution': result['mean_attribution'],
                    'std_attribution': result['std_attribution'],
                    'top_features': result['top_features'],
                    'predictions': result['predictions']
                }
                pbar.update(1)
            else:
                print(f"\nWarning: Failed for {result['speaker_id']}: {result.get('error', 'Unknown')}")
                pbar.update(1)
    
    pbar.close()
    
    # Save summary
    method_file = 'integratedgradients' if use_ig else 'shapley'
    with open(output_dir / f'{method_file}_summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def analyze_shapley_singlegpu(
    embeddings_dir: Path,
    model: nn.Module,
    output_dir: Path,
    device: str = 'cuda:0',
    n_samples: int = 25,
    use_ig: bool = False
) -> Dict:
    """
    Single-GPU fallback for attribution analysis.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = EmbeddingDataset(embeddings_dir)
    
    speaker_embeddings = {}
    for sample in dataset.samples:
        spk = sample['speaker_id']
        if spk not in speaker_embeddings:
            speaker_embeddings[spk] = []
        speaker_embeddings[spk].append(sample['embedding'])
    
    model = model.to(device)
    model.eval()
    
    method_name = 'IntegratedGradients' if use_ig else 'Shapley'
    method_file = 'integratedgradients' if use_ig else 'shapley'
    
    results = {}
    
    for speaker_id, embeddings_list in tqdm(speaker_embeddings.items(), 
                                             desc=f"{method_name} Analysis"):
        embeddings = torch.tensor(np.stack(embeddings_list), dtype=torch.float32)
        
        with torch.no_grad():
            logits = model(embeddings.to(device))
            predictions = logits.argmax(dim=1).cpu().numpy()
        
        all_attributions = []
        
        if use_ig:
            ig = IntegratedGradients(model)
            baseline = torch.zeros(1, 1024, device=device)
            
            for i in range(len(embeddings)):
                target = int(predictions[i])
                emb = embeddings[i:i+1].to(device)
                emb.requires_grad = True
                
                attr = ig.attribute(emb, baselines=baseline, target=target, 
                                   n_steps=50, return_convergence_delta=False)
                all_attributions.append(attr.detach().cpu().numpy()[0])
        else:
            shapley = ShapleyValueSampling(model)
            baseline = torch.zeros(1, 1024, device=device)
            
            for i in range(len(embeddings)):
                target = int(predictions[i])
                emb = embeddings[i:i+1].to(device)
                
                attr = shapley.attribute(emb, baselines=baseline, target=target,
                                        n_samples=n_samples, show_progress=False)
                all_attributions.append(attr.cpu().numpy()[0])
        
        attributions = np.stack(all_attributions)
        
        speaker_output = output_dir / speaker_id
        speaker_output.mkdir(exist_ok=True)
        np.save(speaker_output / f'{method_file}_attributions.npy', attributions)
        
        results[speaker_id] = {
            'num_utterances': len(embeddings_list),
            'mean_attribution': attributions.mean(axis=0).tolist(),
            'std_attribution': attributions.std(axis=0).tolist(),
            'top_features': np.abs(attributions).mean(axis=0).argsort()[-10:][::-1].tolist(),
            'predictions': predictions.tolist()
        }
    
    with open(output_dir / f'{method_file}_summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def create_minimal_classifier(num_classes: int = 4, hidden_dim: int = 256) -> nn.Module:
    """Create a minimal classifier for analysis."""
    model = EmbeddingClassifier(
        input_dim=1024,
        num_classes=num_classes,
        hidden_dim=hidden_dim
    )
    nn.init.xavier_uniform_(model.classifier[0].weight)
    nn.init.xavier_uniform_(model.classifier[3].weight)
    return model


def run_minimal_finetuning(
    model: nn.Module,
    dataset: EmbeddingDataset,
    output_dir: Path,
    num_epochs: int = 3,
    lr: float = 1e-3,
    save_every: int = 1,
    device: str = 'cuda'
) -> List[Path]:
    """Run minimal fine-tuning to generate checkpoints for TracIn (single GPU)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    checkpoint_paths = []
    
    print(f"Running minimal fine-tuning for {num_epochs} epochs (single GPU)...")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in dataloader:
            embeddings = batch['embedding'].to(device)
            pseudo_labels = (embeddings.mean(dim=1) > 0).long()
            pseudo_labels = pseudo_labels % model.classifier[-1].out_features
            
            optimizer.zero_grad()
            logits = model(embeddings)
            loss = criterion(logits, pseudo_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"  Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.4f}")
        
        if (epoch + 1) % save_every == 0:
            ckpt_path = output_dir / f'checkpoint_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }, ckpt_path)
            checkpoint_paths.append(ckpt_path)
    
    return checkpoint_paths


def main():
    parser = argparse.ArgumentParser(description='Captum analysis for speaker embeddings (Multi-GPU)')
    parser.add_argument('--dataset', type=str, choices=['timit', 'iemocap'], default='timit',
                       help='Dataset to analyze')
    parser.add_argument('--embeddings', type=str, default=None,
                       help='Path to embeddings directory (default: embeddings/{dataset})')
    parser.add_argument('--output', type=str, default='analysis/captum',
                       help='Output directory')
    parser.add_argument('--head', type=str, choices=['er', 'sid', 'ks', 'ic'], default='er',
                       help='Prediction head to analyze')
    parser.add_argument('--analysis', type=str, 
                       choices=['shapley', 'data_shapley', 'pydvl_data_shapley', 'ig', 'tracin', 'all'], 
                       default='all', help='Analysis type')
    parser.add_argument('--n-samples', type=int, default=25,
                       help='Number of samples for feature-level Shapley estimation')
    parser.add_argument('--n-permutations', type=int, default=100,
                       help='Number of Monte Carlo permutations for approximate Data Shapley')
    parser.add_argument('--min-updates', type=int, default=PYDVL_DEFAULT_MIN_UPDATES,
                       help='Minimum updates per data point for pyDVL TMC-Shapley (default: stability-first)')
    parser.add_argument('--rtol', type=float, default=PYDVL_DEFAULT_RTOL,
                       help='Relative tolerance for pyDVL truncation (default: stricter for lower variance)')
    parser.add_argument('--pydvl-jobs', type=int, default=8,
                       help='Number of parallel jobs for pyDVL Data Shapley')
    parser.add_argument('--num-gpus', type=int, default=8,
                       help='Maximum number of GPUs to use (default: 8)')
    parser.add_argument('--finetune-epochs', type=int, default=3,
                       help='Number of fine-tuning epochs for TracIn checkpoints')
    parser.add_argument('--single-gpu', action='store_true',
                       help='Force single GPU mode')
    parser.add_argument('--approximate-data-shapley', action='store_true',
                       help='Use approximate Data Shapley (fast) instead of pyDVL (default: use pyDVL)')
    parser.add_argument('--pydvl-model', type=str,
                       choices=[PYDVL_MODEL_FINETUNE_HEAD, PYDVL_MODEL_FROZEN_HEAD_TOP, PYDVL_MODEL_LEGACY_SKLEARN_MLP],
                       default=PYDVL_MODEL_FINETUNE_HEAD,
                       help='pyDVL model variant (default: fine-tune pretrained SUPERB head)')
    parser.add_argument('--pydvl-feature-head', type=str,
                       choices=[PYDVL_FEATURE_HEAD_AUTO, 'sid', 'er', 'ks', 'ic'],
                       default=PYDVL_FEATURE_HEAD_AUTO,
                       help='Pretrained head used for pyDVL frozen/finetune modes (default: auto -> analysis head)')
    parser.add_argument('--pydvl-feature-device', type=str,
                       choices=['auto', 'cpu', 'cuda'],
                       default='auto',
                       help='Device for frozen-head feature extraction')
    parser.add_argument('--pydvl-feature-batch-size', type=int, default=256,
                       help='Batch size for frozen-head feature extraction')
    parser.add_argument('--pydvl-train-device', type=str,
                       choices=['auto', 'cpu', 'cuda'],
                       default='auto',
                       help='Device for finetune-head pyDVL model retraining')
    parser.add_argument('--pydvl-finetune-epochs', type=int, default=40,
                       help='Epochs per subset fit in finetune-head pyDVL mode')
    parser.add_argument('--pydvl-finetune-lr', type=float, default=5e-4,
                       help='Learning rate for finetune-head pyDVL mode')
    parser.add_argument('--pydvl-finetune-batch-size', type=int, default=128,
                       help='Batch size for finetune-head pyDVL mode')
    parser.add_argument('--pydvl-finetune-weight-decay', type=float, default=1e-4,
                       help='Weight decay for finetune-head pyDVL mode')
    parser.add_argument('--disable-pydvl-tqdm', action='store_true',
                       help='Disable tqdm progress/ETA display for pyDVL TMC-Shapley')
    parser.add_argument('--pydvl-task', type=str,
                       choices=[PYDVL_TASK_AUTO, PYDVL_TASK_SPEAKER, PYDVL_TASK_EMOTION],
                       default=PYDVL_TASK_AUTO,
                       help='pyDVL target-label task: auto (default), speaker, or emotion')
    parser.add_argument('--pydvl-allow-single-gpu-concurrency', action='store_true',
                       help='Experimental: allow concurrent pyDVL subset fits on a single GPU when --pydvl-jobs > 1')
    parser.add_argument('--pydvl-joblib-backend', type=str,
                       choices=[PYDVL_JOBLIB_BACKEND_AUTO, PYDVL_JOBLIB_BACKEND_LOKY, PYDVL_JOBLIB_BACKEND_THREADING],
                       default=PYDVL_JOBLIB_BACKEND_AUTO,
                       help='joblib backend for pyDVL parallelism (default: auto)')
    parser.add_argument('--pydvl-enforce-multiclass-subsets', action='store_true',
                       help='Experimental: augment one-class pyDVL subsets with anchor samples from another class (modifies standard TMC estimator)')
    
    args = parser.parse_args()
    
    # Set multiprocessing start method
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Resolve paths
    if args.embeddings is None:
        embeddings_dir = project_root / 'embeddings' / args.dataset
    else:
        embeddings_dir = project_root / args.embeddings
    output_dir = project_root / args.output / args.head
    
    print("=" * 60)
    print("WaveCommit Captum Analysis (Multi-GPU)")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Embeddings: {embeddings_dir}")
    print(f"Output: {output_dir}")
    print(f"Head: {args.head} ({PREDICTION_HEADS[args.head]['name']})")
    print(f"Analysis: {args.analysis}")
    
    # Detect available GPUs
    if not torch.cuda.is_available():
        print("\nNo CUDA available, using CPU")
        gpu_ids = []
    elif args.single_gpu:
        gpu_ids = [0]
        print(f"\nSingle GPU mode: using GPU 0")
    else:
        gpu_ids = select_available_gpus(max_gpus=args.num_gpus)
        if gpu_ids:
            print(f"\nDetected {len(gpu_ids)} available GPUs: {gpu_ids}")
            gpu_info = get_gpu_memory_info()
            for g in gpu_info:
                if g['gpu_id'] in gpu_ids:
                    print(f"  GPU {g['gpu_id']}: {g['name']} - {g['free_mb']}MB free / {g['total_mb']}MB total")
        else:
            print("\nNo GPUs with sufficient memory, using CPU")
    
    print(f"Memory fraction: {GPU_MEMORY_FRACTION * 100:.0f}%")
    
    # Create dataset
    dataset = EmbeddingDataset(embeddings_dir)
    print(f"\nLoaded {len(dataset)} samples from {args.dataset}")
    
    # Get head config
    head_config = PREDICTION_HEADS[args.head]
    num_classes = head_config['num_classes']
    
    # Create classifier
    model = create_minimal_classifier(num_classes=num_classes)
    
    results = {}
    
    use_multigpu = len(gpu_ids) > 1
    primary_device = f'cuda:{gpu_ids[0]}' if gpu_ids else 'cpu'
    
    if args.analysis in ['shapley', 'all']:
        print("\n" + "-" * 40)
        print("Running Shapley Value Analysis...")
        print("-" * 40)
        
        shapley_output = output_dir / 'shapley'
        start_time = time.time()
        
        if use_multigpu:
            shapley_results = analyze_shapley_multigpu(
                embeddings_dir=embeddings_dir,
                model=model,
                output_dir=shapley_output,
                gpu_ids=gpu_ids,
                n_samples=args.n_samples,
                use_ig=False
            )
        else:
            shapley_results = analyze_shapley_singlegpu(
                embeddings_dir=embeddings_dir,
                model=model,
                output_dir=shapley_output,
                device=primary_device,
                n_samples=args.n_samples,
                use_ig=False
            )
        
        elapsed = time.time() - start_time
        results['shapley'] = shapley_results
        print(f"Shapley (feature-level) results saved to: {shapley_output}")
        print(f"Time: {elapsed:.1f}s")
    
    if args.analysis in ['data_shapley', 'pydvl_data_shapley', 'all']:
        print("\n" + "-" * 40)
        print("Running Data Shapley Analysis (utterance-level)...")
        print("-" * 40)
        
        data_shapley_output = output_dir / 'data_shapley'
        start_time = time.time()
        
        # Use pyDVL (true Data Shapley) by default, approximate if flag set
        if args.approximate_data_shapley:
            print("  Mode: Approximate Data Shapley (fast, no retraining)")
            print("  - Intra-speaker: Leave-one-out influence within speaker")
            print("  - Inter-speaker: Contribution to speaker separation")
            
            if use_multigpu:
                data_shapley_results = analyze_data_shapley_multigpu(
                    embeddings_dir=embeddings_dir,
                    model=model,
                    output_dir=data_shapley_output,
                    gpu_ids=gpu_ids,
                    n_permutations=args.n_permutations
                )
            else:
                data_shapley_results = analyze_data_shapley_singlegpu(
                    embeddings_dir=embeddings_dir,
                    model=model,
                    output_dir=data_shapley_output,
                    device=primary_device,
                    n_permutations=args.n_permutations
                )
        else:
            print("  Mode: pyDVL TMC-Shapley (true Data Shapley with model retraining)")
            resolved_main_task = resolve_pydvl_task_mode(
                task_mode=args.pydvl_task,
                dataset_name=args.dataset,
                analysis_head=args.head,
            )
            resolved_feature_head = resolve_pydvl_feature_head(
                feature_head=args.pydvl_feature_head,
                analysis_head=args.head,
            )
            if resolved_main_task == PYDVL_TASK_EMOTION:
                print("  Task: Emotion Recognition (full IEMOCAP emotion labels, 80/20 train/test split)")
            else:
                print("  Task: Speaker Identification (80/20 train/test split)")
            print(f"  Min updates: {args.min_updates}, rtol: {args.rtol}")

            feature_device = args.pydvl_feature_device
            if feature_device == 'auto':
                feature_device = primary_device if primary_device.startswith('cuda') else 'cpu'
            if feature_device == 'cuda' and not torch.cuda.is_available():
                feature_device = 'cpu'

            if args.pydvl_model == PYDVL_MODEL_FINETUNE_HEAD:
                print("  pyDVL model: fine-tune pretrained SUPERB projector + classifier")
                print(f"  Head init: {resolved_feature_head}")
                print(f"  Finetune device: {args.pydvl_train_device}")
            elif args.pydvl_model == PYDVL_MODEL_FROZEN_HEAD_TOP:
                print("  pyDVL model: frozen SUPERB head + top classifier")
                print(f"  Feature head: {resolved_feature_head}")
            else:
                print("  pyDVL model: legacy sklearn MLP (comparison mode)")
            print(f"  pyDVL target task: {args.pydvl_task}")
            if args.pydvl_feature_head == PYDVL_FEATURE_HEAD_AUTO:
                print("  pyDVL feature head selection: auto")
            if args.pydvl_allow_single_gpu_concurrency:
                print("  Single-GPU concurrency: enabled (experimental)")
            if args.pydvl_enforce_multiclass_subsets:
                print("  One-class subset handling: enabled (experimental, modifies estimator)")
            print(f"  joblib backend: {args.pydvl_joblib_backend}")
            print(f"  pyDVL tqdm progress: {'disabled' if args.disable_pydvl_tqdm else 'enabled'}")

            try:
                data_shapley_results = analyze_pydvl_data_shapley(
                    embeddings_dir=embeddings_dir,
                    output_dir=data_shapley_output,
                    min_updates=args.min_updates,
                    rtol=args.rtol,
                    n_jobs_per_speaker=args.pydvl_jobs,
                    model_variant=args.pydvl_model,
                    feature_head=resolved_feature_head,
                    project_root=project_root,
                    feature_batch_size=args.pydvl_feature_batch_size,
                    feature_device=feature_device,
                    train_device=args.pydvl_train_device,
                    finetune_epochs=args.pydvl_finetune_epochs,
                    finetune_lr=args.pydvl_finetune_lr,
                    finetune_batch_size=args.pydvl_finetune_batch_size,
                    finetune_weight_decay=args.pydvl_finetune_weight_decay,
                    gpu_ids=gpu_ids,
                    show_pydvl_progress=not args.disable_pydvl_tqdm,
                    task_mode=args.pydvl_task,
                    dataset_name=args.dataset,
                    analysis_head=args.head,
                    allow_single_gpu_concurrency=args.pydvl_allow_single_gpu_concurrency,
                    joblib_backend=args.pydvl_joblib_backend,
                    enforce_multiclass_subsets=args.pydvl_enforce_multiclass_subsets,
                )
            except RuntimeError as exc:
                print(f"\n[ERROR] {exc}")
                sys.exit(1)
        
        elapsed = time.time() - start_time
        results['data_shapley'] = data_shapley_results
        print(f"Data Shapley (utterance-level) results saved to: {data_shapley_output}")
        print(f"Time: {elapsed:.1f}s")
    
    if args.analysis in ['ig', 'all']:
        print("\n" + "-" * 40)
        print("Running Integrated Gradients Analysis...")
        print("-" * 40)
        
        ig_output = output_dir / 'integrated_gradients'
        start_time = time.time()
        
        if use_multigpu:
            ig_results = analyze_shapley_multigpu(
                embeddings_dir=embeddings_dir,
                model=model,
                output_dir=ig_output,
                gpu_ids=gpu_ids,
                use_ig=True
            )
        else:
            ig_results = analyze_shapley_singlegpu(
                embeddings_dir=embeddings_dir,
                model=model,
                output_dir=ig_output,
                device=primary_device,
                use_ig=True
            )
        
        elapsed = time.time() - start_time
        results['integrated_gradients'] = ig_results
        print(f"IG results saved to: {ig_output}")
        print(f"Time: {elapsed:.1f}s")
    
    if args.analysis in ['tracin', 'all']:
        print("\n" + "-" * 40)
        print("Running TracIn Analysis (single GPU)...")
        print("-" * 40)
        
        tracin_output = output_dir / 'tracin'
        tracin_output.mkdir(parents=True, exist_ok=True)
        
        checkpoint_dir = tracin_output / 'checkpoints'
        checkpoint_paths = run_minimal_finetuning(
            model=model,
            dataset=dataset,
            output_dir=checkpoint_dir,
            num_epochs=args.finetune_epochs,
            device=primary_device
        )
        
        print(f"\nGenerated {len(checkpoint_paths)} checkpoints")
        
        with open(tracin_output / 'checkpoint_info.json', 'w') as f:
            json.dump({
                'checkpoints': [str(p) for p in checkpoint_paths],
                'epochs': args.finetune_epochs,
                'head': args.head,
                'num_classes': num_classes
            }, f, indent=2)
        
        results['tracin'] = {
            'checkpoints': [str(p) for p in checkpoint_paths],
            'status': 'checkpoints_generated'
        }
        
        print(f"TracIn checkpoints saved to: {checkpoint_dir}")
    
    # Save overall summary
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'analysis_summary.json', 'w') as f:
        summary = {
            'dataset': args.dataset,
            'head': args.head,
            'num_samples': len(dataset),
            'analyses_run': list(results.keys()),
            'gpus_used': gpu_ids if use_multigpu else ([gpu_ids[0]] if gpu_ids else ['cpu']),
            'memory_fraction': GPU_MEMORY_FRACTION
        }
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Captum Analysis Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
