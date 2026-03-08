#!/usr/bin/env python3
"""
WaveCommit Commitment Tree Builder

Builds cryptographic commitments for speaker embedding pools using TensorCommitment.
Supports both TIMIT and IEMOCAP datasets with per-speaker commitments.

Usage:
    python scripts/build_commitments.py --dataset timit
    python scripts/build_commitments.py --dataset iemocap
    python scripts/build_commitments.py --dataset all
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm
import numpy as np

# Import TensorCommitment library
import tensor_commitment_lib


@dataclass
class CommitmentMetadata:
    """Metadata for a speaker commitment."""
    speaker_id: str
    dataset: str
    num_utterances: int
    embedding_dim: int
    total_elements: int
    num_variables: int
    degree_bound: int
    commitment_hex: str
    commitment_bytes: int


def find_optimal_hypercube_shape(total_elements: int, 
                                  min_dim: int = 4, 
                                  max_dim: int = 32) -> Tuple[int, int, int]:
    """
    Find optimal hypercube shape for the given number of elements.
    
    For small datasets (TIMIT ~10K elements): uses smaller degrees (4-16)
    For large datasets (IEMOCAP ~1M elements): uses larger degrees (up to 32)
    
    Args:
        total_elements: Total number of elements to commit
        min_dim: Minimum dimension size
        max_dim: Maximum dimension size (32 for large datasets)
        
    Returns:
        Tuple of (num_variables, degree_bound, total_capacity)
    """
    best_shape = None
    best_waste = float('inf')
    
    # Adjust max_dim based on data size
    if total_elements > 500000:
        # Large dataset (IEMOCAP): allow larger degrees
        effective_max = max_dim
    elif total_elements > 50000:
        # Medium dataset
        effective_max = min(max_dim, 24)
    else:
        # Small dataset (TIMIT)
        effective_max = min(max_dim, 16)
    
    for degree in range(min_dim, effective_max + 1):
        # Find minimum number of variables needed
        num_vars = 1
        while degree ** num_vars < total_elements:
            num_vars += 1
        
        capacity = degree ** num_vars
        waste = capacity - total_elements
        waste_pct = waste / capacity
        
        # Prefer shapes with less waste and reasonable number of variables
        # Also prefer fewer variables for efficiency
        if waste_pct < 0.5 and num_vars <= 6:
            score = waste + (num_vars * 10000)  # Penalize more variables
            if score < best_waste:
                best_waste = score
                best_shape = (num_vars, degree, capacity)
    
    if best_shape is None:
        # Fallback: use larger degree with reasonable variables
        degree = effective_max
        num_vars = 1
        while degree ** num_vars < total_elements:
            num_vars += 1
        best_shape = (num_vars, degree, degree ** num_vars)
    
    return best_shape


def flatten_embeddings_to_integers(embeddings: np.ndarray, 
                                   scale_factor: int = 8) -> List[int]:
    """
    Convert float embeddings to integers for commitment.
    
    Args:
        embeddings: Float embeddings array
        scale_factor: Decimal places to preserve (10^scale_factor)
        
    Returns:
        List of integers
    """
    # Flatten embeddings
    flat = embeddings.flatten()
    
    # Scale to integers
    scale = 10 ** scale_factor
    integers = (flat * scale).astype(np.int64).tolist()
    
    return integers


def pad_to_capacity(values: List[int], capacity: int, seed: int = 42) -> List[int]:
    """
    Pad values to match hypercube capacity.
    
    Args:
        values: Original values
        capacity: Target capacity
        seed: Random seed for padding values
        
    Returns:
        Padded list of values
    """
    if len(values) >= capacity:
        return values[:capacity]
    
    # Pad with random samples from existing values
    np.random.seed(seed)
    padding_needed = capacity - len(values)
    padding_indices = np.random.choice(len(values), padding_needed)
    padding = [values[i] for i in padding_indices]
    
    return values + padding


def build_speaker_commitment(
    speaker_id: str,
    embeddings_dir: Path,
    output_dir: Path,
    use_utterance_level: bool = True,
    scale_factor: int = 8
) -> Optional[CommitmentMetadata]:
    """
    Build commitment for a single speaker's embeddings.
    
    Args:
        speaker_id: Speaker identifier
        embeddings_dir: Directory containing speaker embeddings
        output_dir: Output directory for commitments
        use_utterance_level: Use utterance-level (pooled) embeddings
        scale_factor: Integer scaling factor
        
    Returns:
        CommitmentMetadata or None if failed
    """
    speaker_emb_dir = embeddings_dir / speaker_id
    
    # Load embeddings
    if use_utterance_level:
        emb_file = speaker_emb_dir / 'utterance_embeddings.npy'
    else:
        emb_file = speaker_emb_dir / 'frame_embeddings.npy'
    
    if not emb_file.exists():
        print(f"  Warning: Embeddings not found for {speaker_id}")
        return None
    
    embeddings = np.load(emb_file)
    
    # Load metadata
    with open(speaker_emb_dir / 'metadata.json') as f:
        emb_meta = json.load(f)
    
    # Convert to integers
    integers = flatten_embeddings_to_integers(embeddings, scale_factor)
    total_elements = len(integers)
    
    # Find optimal shape
    num_vars, degree_bound, capacity = find_optimal_hypercube_shape(total_elements)
    
    # Pad to capacity
    padded_values = pad_to_capacity(integers, capacity)
    
    # Create commitment
    try:
        wrapper = tensor_commitment_lib.PSTWrapper(num_vars, degree_bound)
        commitment_hex = wrapper.commit(padded_values)
        
    except Exception as e:
        print(f"  Error creating commitment for {speaker_id}: {e}")
        return None
    
    # Create output directory
    speaker_output = output_dir / speaker_id
    speaker_output.mkdir(parents=True, exist_ok=True)
    
    # Save commitment
    with open(speaker_output / 'commitment.txt', 'w') as f:
        f.write(commitment_hex)
    
    # Create metadata
    metadata = CommitmentMetadata(
        speaker_id=speaker_id,
        dataset=emb_meta.get('dataset', 'unknown'),
        num_utterances=emb_meta['num_utterances'],
        embedding_dim=emb_meta['embedding_dim'],
        total_elements=total_elements,
        num_variables=num_vars,
        degree_bound=degree_bound,
        commitment_hex=commitment_hex,
        commitment_bytes=len(commitment_hex) // 2  # hex to bytes
    )
    
    # Save metadata
    with open(speaker_output / 'commitment_metadata.json', 'w') as f:
        json.dump({
            **asdict(metadata),
            'capacity': capacity,
            'padding': capacity - total_elements,
            'scale_factor': scale_factor,
            'use_utterance_level': use_utterance_level
        }, f, indent=2)
    
    return metadata


def build_timit_commitments(
    embeddings_dir: Path,
    output_dir: Path,
    use_utterance_level: bool = True
) -> Dict[str, CommitmentMetadata]:
    """
    Build commitments for all TIMIT speakers.
    
    Args:
        embeddings_dir: Directory containing TIMIT embeddings
        output_dir: Output directory for commitments
        use_utterance_level: Use utterance-level embeddings
        
    Returns:
        Dictionary of speaker_id -> CommitmentMetadata
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all speaker directories (TIMIT: dr{1-8}-{speaker})
    speaker_dirs = sorted([d for d in embeddings_dir.iterdir() 
                          if d.is_dir() and d.name.startswith('dr')])
    
    print(f"\nBuilding commitments for {len(speaker_dirs)} TIMIT speakers...")
    
    all_metadata = {}
    
    for speaker_dir in tqdm(speaker_dirs, desc="TIMIT Commitments"):
        speaker_id = speaker_dir.name
        
        metadata = build_speaker_commitment(
            speaker_id=speaker_id,
            embeddings_dir=embeddings_dir,
            output_dir=output_dir,
            use_utterance_level=use_utterance_level
        )
        
        if metadata:
            all_metadata[speaker_id] = metadata
    
    return all_metadata


def build_iemocap_commitments(
    embeddings_dir: Path,
    output_dir: Path,
    use_utterance_level: bool = True
) -> Dict[str, CommitmentMetadata]:
    """
    Build commitments for all IEMOCAP speakers.
    
    IEMOCAP has ~860-1000 utterances per speaker (much larger than TIMIT).
    Uses larger hypercubes to accommodate the data.
    
    Args:
        embeddings_dir: Directory containing IEMOCAP embeddings
        output_dir: Output directory for commitments
        use_utterance_level: Use utterance-level embeddings
        
    Returns:
        Dictionary of speaker_id -> CommitmentMetadata
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all speaker directories (IEMOCAP: Ses{01-05}{F/M})
    speaker_dirs = sorted([d for d in embeddings_dir.iterdir() 
                          if d.is_dir() and d.name.startswith('Ses')])
    
    print(f"\nBuilding commitments for {len(speaker_dirs)} IEMOCAP speakers...")
    print("Note: IEMOCAP has ~1000 utterances/speaker - this may take longer")
    
    all_metadata = {}
    
    for speaker_dir in tqdm(speaker_dirs, desc="IEMOCAP Commitments"):
        speaker_id = speaker_dir.name
        
        metadata = build_speaker_commitment(
            speaker_id=speaker_id,
            embeddings_dir=embeddings_dir,
            output_dir=output_dir,
            use_utterance_level=use_utterance_level
        )
        
        if metadata:
            all_metadata[speaker_id] = metadata
    
    return all_metadata


def save_summary_and_print(results: Dict[str, CommitmentMetadata], 
                           output_dir: Path, 
                           dataset_name: str,
                           use_utterance_level: bool):
    """Save summary JSON and print results table."""
    if not results:
        return
    
    summary = {
        'dataset': dataset_name,
        'num_speakers': len(results),
        'embedding_level': 'utterance' if use_utterance_level else 'frame',
        'speakers': {
            spk: {
                'commitment_hex': meta.commitment_hex[:32] + '...',
                'commitment_bytes': meta.commitment_bytes,
                'num_variables': meta.num_variables,
                'degree_bound': meta.degree_bound,
                'total_elements': meta.total_elements
            }
            for spk, meta in results.items()
        }
    }
    
    with open(output_dir / 'commitment_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{dataset_name.upper()}: Built commitments for {len(results)} speakers")
    print(f"\nCommitment Summary:")
    print(f"{'Speaker':<15} {'Elements':>12} {'Vars':>6} {'Degree':>8} {'Bytes':>8}")
    print("-" * 55)
    for spk, meta in sorted(results.items()):
        print(f"{spk:<15} {meta.total_elements:>12,} {meta.num_variables:>6} {meta.degree_bound:>8} {meta.commitment_bytes:>8}")


def main():
    parser = argparse.ArgumentParser(description='Build commitment trees for speaker embeddings')
    parser.add_argument('--dataset', type=str, choices=['timit', 'iemocap', 'all'], default='timit',
                       help='Dataset to process (timit, iemocap, or all)')
    parser.add_argument('--embeddings', type=str, default=None,
                       help='Path to embeddings directory (default: embeddings/{dataset})')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for commitments (default: commitments/{dataset})')
    parser.add_argument('--utterance-level', action='store_true', default=True,
                       help='Use utterance-level (pooled) embeddings')
    parser.add_argument('--frame-level', action='store_true',
                       help='Use frame-level embeddings (larger commitments)')
    
    args = parser.parse_args()
    
    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    use_utterance_level = not args.frame_level
    
    print("=" * 60)
    print("WaveCommit Commitment Tree Builder")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Level: {'utterance' if use_utterance_level else 'frame'}")
    
    datasets_to_process = []
    if args.dataset == 'all':
        datasets_to_process = ['timit', 'iemocap']
    else:
        datasets_to_process = [args.dataset]
    
    all_results = {}
    
    for dataset in datasets_to_process:
        # Resolve paths for this dataset
        if args.embeddings:
            embeddings_dir = project_root / args.embeddings
        else:
            embeddings_dir = project_root / 'embeddings' / dataset
        
        if args.output:
            output_dir = project_root / args.output
        else:
            output_dir = project_root / 'commitments' / dataset
        
        print(f"\n{'='*40}")
        print(f"Processing: {dataset.upper()}")
        print(f"Embeddings: {embeddings_dir}")
        print(f"Output: {output_dir}")
        
        if not embeddings_dir.exists():
            print(f"Warning: Embeddings directory not found: {embeddings_dir}")
            continue
        
        if dataset == 'timit':
            results = build_timit_commitments(
                embeddings_dir=embeddings_dir,
                output_dir=output_dir,
                use_utterance_level=use_utterance_level
            )
        elif dataset == 'iemocap':
            results = build_iemocap_commitments(
                embeddings_dir=embeddings_dir,
                output_dir=output_dir,
                use_utterance_level=use_utterance_level
            )
        else:
            print(f"Unknown dataset: {dataset}")
            continue
        
        save_summary_and_print(results, output_dir, dataset, use_utterance_level)
        all_results[dataset] = results
    
    print("\n" + "=" * 60)
    print("Commitment Building Complete!")
    print("=" * 60)
    
    for dataset, results in all_results.items():
        print(f"  {dataset}: {len(results)} speakers -> commitments/{dataset}/")


if __name__ == '__main__':
    main()
