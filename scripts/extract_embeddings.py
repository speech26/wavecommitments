#!/usr/bin/env python3
"""
WaveCommit Speaker Embedding Extraction Script (Multi-GPU)

Extracts HuBERT-Large embeddings for all speakers in TIMIT and IEMOCAP datasets.
Saves embeddings per speaker for later commitment tree construction.

Supports multi-GPU parallelization: each GPU processes different speakers in parallel.

Usage:
    python scripts/extract_embeddings.py --dataset timit --output embeddings/timit
    python scripts/extract_embeddings.py --dataset iemocap --output embeddings/iemocap
    python scripts/extract_embeddings.py --dataset all --output embeddings/
    python scripts/extract_embeddings.py --dataset iemocap --num-gpus 8  # Use 8 GPUs
"""

import argparse
import os
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm
import numpy as np
import torch
import torch.multiprocessing as mp
import soundfile as sf
import librosa
from transformers import HubertModel, Wav2Vec2FeatureExtractor

# GPU memory configuration
GPU_MEMORY_FRACTION = 0.90  # Use 90% of available GPU memory
MIN_FREE_MEMORY_MB = 2000   # Minimum 2GB free memory required


def get_gpu_memory_info() -> List[Dict]:
    """Get memory info for all GPUs using nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.free,memory.total', 
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True
        )
        
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 4:
                    gpus.append({
                        'gpu_id': int(parts[0]),
                        'name': parts[1],
                        'free_mb': int(parts[2]),
                        'total_mb': int(parts[3])
                    })
        return gpus
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []


def select_available_gpus(max_gpus: int = 8, min_free_mb: int = MIN_FREE_MEMORY_MB) -> List[int]:
    """Select GPUs with sufficient free memory."""
    gpu_info = get_gpu_memory_info()
    
    # Filter GPUs with enough free memory
    available = [g for g in gpu_info if g['free_mb'] >= min_free_mb]
    
    # Sort by free memory (most free first)
    available.sort(key=lambda x: x['free_mb'], reverse=True)
    
    # Limit to max_gpus
    selected = [g['gpu_id'] for g in available[:max_gpus]]
    
    return selected


@dataclass
class EmbeddingMetadata:
    """Metadata for extracted embeddings."""
    speaker_id: str
    dataset: str
    num_utterances: int
    embedding_dim: int
    total_frames: int
    model_name: str
    extraction_layer: str  # 'last_hidden_state' or specific layer


def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class HuBERTEmbeddingExtractor:
    """Extract embeddings using HuBERT-Large model."""
    
    def __init__(self, model_path: str, device: Optional[torch.device] = None):
        """
        Initialize the embedding extractor.
        
        Args:
            model_path: Path to local HuBERT model or HuggingFace model ID
            device: Torch device to use
        """
        self.device = device or get_device()
        print(f"Loading HuBERT model from {model_path}...")
        print(f"Using device: {self.device}")
        
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        self.model = HubertModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Model info
        self.hidden_size = self.model.config.hidden_size
        self.num_layers = self.model.config.num_hidden_layers
        print(f"Model loaded: hidden_size={self.hidden_size}, num_layers={self.num_layers}")
    
    @torch.no_grad()
    def extract(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Extract embeddings from audio.
        
        Args:
            audio: Audio waveform (1D numpy array)
            sample_rate: Audio sample rate (should be 16000 for HuBERT)
            
        Returns:
            Embeddings of shape (num_frames, hidden_size)
        """
        # Resample if needed
        if sample_rate != 16000:
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sample_rate, target_sr=16000)
        
        # Prepare input
        inputs = self.feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        input_values = inputs.input_values.to(self.device)
        
        # Extract embeddings
        outputs = self.model(input_values, output_hidden_states=True)
        
        # Use last hidden state (shape: [1, num_frames, hidden_size])
        embeddings = outputs.last_hidden_state.squeeze(0).cpu().numpy()
        
        return embeddings
    
    @torch.no_grad()
    def extract_pooled(self, audio: np.ndarray, sample_rate: int = 16000, 
                       pooling: str = 'mean') -> np.ndarray:
        """
        Extract pooled embeddings (single vector per utterance).
        
        Args:
            audio: Audio waveform
            sample_rate: Audio sample rate
            pooling: 'mean', 'max', or 'last'
            
        Returns:
            Pooled embedding of shape (hidden_size,)
        """
        embeddings = self.extract(audio, sample_rate)
        
        if pooling == 'mean':
            return embeddings.mean(axis=0)
        elif pooling == 'max':
            return embeddings.max(axis=0)
        elif pooling == 'last':
            return embeddings[-1]
        else:
            raise ValueError(f"Unknown pooling: {pooling}")


def load_timit_audio(wav_path: str) -> Tuple[np.ndarray, int]:
    """Load TIMIT audio file."""
    audio, sr = sf.read(wav_path)
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), sr


def load_iemocap_audio(sample: dict) -> Tuple[np.ndarray, int]:
    """Load IEMOCAP audio from HuggingFace dataset sample."""
    audio_data = sample['audio']
    return np.array(audio_data['array']), audio_data['sampling_rate']


def load_speech_commands_audio(path: str) -> Tuple[np.ndarray, int]:
    """Load Speech Commands audio from a WAV file path."""
    audio, sr = sf.read(path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), sr


# =============================================================================
# Multi-GPU Worker Functions
# =============================================================================

def process_timit_speaker_worker(args: Tuple) -> Dict:
    """
    Worker function to process a single TIMIT speaker on a specific GPU.
    
    Args:
        args: Tuple of (speaker_dir, output_dir, model_path, pooling, gpu_id)
        
    Returns:
        Dictionary with speaker metadata or error info
    """
    speaker_dir, output_dir, model_path, pooling, gpu_id = args
    speaker_id = Path(speaker_dir).name
    
    try:
        # Set up GPU for this worker
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(gpu_id)
        torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, gpu_id)
        
        # Load model on this GPU (each worker loads its own copy)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        model = HubertModel.from_pretrained(model_path)
        model.to(device)
        model.eval()
        hidden_size = model.config.hidden_size
        
        # Create output directory
        speaker_output = Path(output_dir) / speaker_id
        speaker_output.mkdir(parents=True, exist_ok=True)
        
        # Find all wav files
        wav_files = sorted(Path(speaker_dir).glob('*.wav'))
        
        frame_embeddings = []
        utterance_embeddings = []
        utterance_info = []
        
        for wav_file in wav_files:
            try:
                audio, sr = load_timit_audio(str(wav_file))
                
                # Resample if needed
                if sr != 16000:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                
                # Extract embeddings
                inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
                input_values = inputs.input_values.to(device)
                
                with torch.no_grad():
                    outputs = model(input_values, output_hidden_states=True)
                    frames = outputs.last_hidden_state.squeeze(0).cpu().numpy()
                
                frame_embeddings.append(frames)
                
                # Pooled embedding
                if pooling == 'mean':
                    pooled = frames.mean(axis=0)
                elif pooling == 'max':
                    pooled = frames.max(axis=0)
                else:
                    pooled = frames[-1]
                    
                utterance_embeddings.append(pooled)
                
                utterance_info.append({
                    'file': wav_file.name,
                    'num_frames': frames.shape[0],
                    'duration_sec': len(audio) / 16000
                })
                
            except Exception as e:
                continue
        
        if not utterance_embeddings:
            return {'speaker_id': speaker_id, 'success': False, 'error': 'No embeddings extracted'}
        
        # Save embeddings
        utterance_embeddings = np.stack(utterance_embeddings)
        all_frames = np.concatenate(frame_embeddings, axis=0)
        
        np.save(speaker_output / 'utterance_embeddings.npy', utterance_embeddings)
        np.save(speaker_output / 'frame_embeddings.npy', all_frames)
        
        # Create metadata
        metadata = {
            'speaker_id': speaker_id,
            'dataset': 'timit',
            'num_utterances': len(utterance_embeddings),
            'embedding_dim': hidden_size,
            'total_frames': all_frames.shape[0],
            'model_name': 'facebook/hubert-large-ll60k',
            'extraction_layer': 'last_hidden_state',
            'utterances': utterance_info,
            'pooling': pooling
        }
        
        with open(speaker_output / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Clean up GPU memory
        del model, feature_extractor
        torch.cuda.empty_cache()
        
        return {
            'speaker_id': speaker_id,
            'success': True,
            'num_utterances': len(utterance_embeddings),
            'total_frames': all_frames.shape[0],
            'gpu_id': gpu_id
        }
        
    except Exception as e:
        return {'speaker_id': speaker_id, 'success': False, 'error': str(e)}


def process_iemocap_speaker_worker(args: Tuple) -> Dict:
    """
    Worker function to process a single IEMOCAP speaker on a specific GPU.
    
    Args:
        args: Tuple of (speaker_id, sample_indices, iemocap_dir, output_dir, model_path, pooling, gpu_id)
        
    Returns:
        Dictionary with speaker metadata or error info
    """
    speaker_id, sample_indices, iemocap_dir, output_dir, model_path, pooling, gpu_id = args
    
    try:
        from datasets import load_from_disk
        
        # Set up GPU for this worker
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(gpu_id)
        torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, gpu_id)
        
        # Load model on this GPU
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        model = HubertModel.from_pretrained(model_path)
        model.to(device)
        model.eval()
        hidden_size = model.config.hidden_size
        
        # Load dataset
        dataset = load_from_disk(str(iemocap_dir))
        train = dataset['train']
        
        # Create output directory
        speaker_output = Path(output_dir) / speaker_id
        speaker_output.mkdir(parents=True, exist_ok=True)
        
        frame_embeddings = []
        utterance_embeddings = []
        utterance_info = []
        
        for idx in sample_indices:
            try:
                sample = train[idx]
                audio_data = sample['audio']
                audio = np.array(audio_data['array'])
                sr = audio_data['sampling_rate']
                
                # Resample if needed
                if sr != 16000:
                    audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
                
                # Extract embeddings
                inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
                input_values = inputs.input_values.to(device)
                
                with torch.no_grad():
                    outputs = model(input_values, output_hidden_states=True)
                    frames = outputs.last_hidden_state.squeeze(0).cpu().numpy()
                
                frame_embeddings.append(frames)
                
                # Pooled embedding
                if pooling == 'mean':
                    pooled = frames.mean(axis=0)
                elif pooling == 'max':
                    pooled = frames.max(axis=0)
                else:
                    pooled = frames[-1]
                    
                utterance_embeddings.append(pooled)
                
                utterance_info.append({
                    'file': sample['file'],
                    'num_frames': frames.shape[0],
                    'duration_sec': len(audio) / 16000,
                    'emotion': sample.get('major_emotion', 'unknown'),
                    'gender': sample.get('gender', 'unknown')
                })
                
            except Exception as e:
                continue
        
        if not utterance_embeddings:
            return {'speaker_id': speaker_id, 'success': False, 'error': 'No embeddings extracted'}
        
        # Save embeddings
        utterance_embeddings = np.stack(utterance_embeddings)
        all_frames = np.concatenate(frame_embeddings, axis=0)
        
        np.save(speaker_output / 'utterance_embeddings.npy', utterance_embeddings)
        np.save(speaker_output / 'frame_embeddings.npy', all_frames)
        
        # Create metadata
        metadata = {
            'speaker_id': speaker_id,
            'dataset': 'iemocap',
            'num_utterances': len(utterance_embeddings),
            'embedding_dim': hidden_size,
            'total_frames': all_frames.shape[0],
            'model_name': 'facebook/hubert-large-ll60k',
            'extraction_layer': 'last_hidden_state',
            'utterances': utterance_info,
            'pooling': pooling
        }
        
        with open(speaker_output / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Clean up GPU memory
        del model, feature_extractor, dataset
        torch.cuda.empty_cache()
        
        return {
            'speaker_id': speaker_id,
            'success': True,
            'num_utterances': len(utterance_embeddings),
            'total_frames': all_frames.shape[0],
            'gpu_id': gpu_id
        }
        
    except Exception as e:
        return {'speaker_id': speaker_id, 'success': False, 'error': str(e)}


def process_speech_commands_speaker_worker(args: Tuple) -> Dict:
    """
    Worker function to process a single Speech Commands speaker on a specific GPU.

    Args:
        args: Tuple of (speaker_id, wav_files, output_dir, model_path, pooling, gpu_id)
    """
    speaker_id, wav_files, output_dir, model_path, pooling, gpu_id = args

    try:
        # Set up device
        device = torch.device(f"cuda:{gpu_id}") if torch.cuda.is_available() else get_device()
        if device.type == "cuda":
            torch.cuda.set_device(gpu_id)
            torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, gpu_id)

        # Load model on this worker's device
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        model = HubertModel.from_pretrained(model_path)
        model.to(device)
        model.eval()
        hidden_size = model.config.hidden_size

        speaker_output = Path(output_dir) / speaker_id
        speaker_output.mkdir(parents=True, exist_ok=True)

        frame_embeddings: List[np.ndarray] = []
        utterance_embeddings: List[np.ndarray] = []
        utterance_info: List[Dict] = []

        for wav_path in wav_files:
            try:
                audio, sr = load_speech_commands_audio(str(wav_path))
                if sr != 16000:
                    audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)

                inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
                input_values = inputs.input_values.to(device)

                with torch.no_grad():
                    outputs = model(input_values, output_hidden_states=True)
                    frames = outputs.last_hidden_state.squeeze(0).cpu().numpy()

                frame_embeddings.append(frames)

                if pooling == "mean":
                    pooled = frames.mean(axis=0)
                elif pooling == "max":
                    pooled = frames.max(axis=0)
                elif pooling == "last":
                    pooled = frames[-1]
                else:
                    raise ValueError(f"Unknown pooling: {pooling}")

                utterance_embeddings.append(pooled)

                rel = wav_path.name
                utterance_info.append(
                    {
                        "file": rel,
                        "num_frames": frames.shape[0],
                        "duration_sec": len(audio) / 16000.0,
                    }
                )
            except Exception as e:
                print(f"  Warning: Failed to process Speech Commands file {wav_path}: {e}")
                continue

        if not utterance_embeddings:
            return {"speaker_id": speaker_id, "success": False, "error": "No utterances processed"}

        all_frames = np.concatenate(frame_embeddings, axis=0)
        utterance_embeddings_arr = np.stack(utterance_embeddings, axis=0)

        np.save(speaker_output / "frame_embeddings.npy", all_frames)
        np.save(speaker_output / "utterance_embeddings.npy", utterance_embeddings_arr)

        metadata = {
            "speaker_id": speaker_id,
            "dataset": "speech_commands",
            "num_utterances": len(utterance_embeddings_arr),
            "embedding_dim": hidden_size,
            "total_frames": all_frames.shape[0],
            "model_name": "facebook/hubert-large-ll60k",
            "extraction_layer": "last_hidden_state",
            "utterances": utterance_info,
            "pooling": pooling,
        }

        with open(speaker_output / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        del model, feature_extractor
        if device.type == "cuda":
            torch.cuda.empty_cache()

        return {
            "speaker_id": speaker_id,
            "success": True,
            "num_utterances": len(utterance_embeddings_arr),
            "total_frames": all_frames.shape[0],
            "gpu_id": gpu_id,
        }

    except Exception as e:
        return {"speaker_id": speaker_id, "success": False, "error": str(e)}


def extract_timit_embeddings_multigpu(
    timit_dir: Path,
    output_dir: Path,
    model_path: str,
    pooling: str = 'mean',
    gpu_ids: List[int] = None
) -> Dict[str, dict]:
    """
    Extract embeddings for all TIMIT speakers using multiple GPUs.
    
    Args:
        timit_dir: Path to TIMIT dataset
        output_dir: Output directory for embeddings
        model_path: Path to HuBERT model
        pooling: Pooling strategy
        gpu_ids: List of GPU IDs to use
        
    Returns:
        Dictionary of speaker_id -> metadata
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all speaker directories
    speaker_dirs = sorted([d for d in timit_dir.iterdir() 
                          if d.is_dir() and d.name.startswith('dr')])
    
    num_speakers = len(speaker_dirs)
    num_gpus = len(gpu_ids) if gpu_ids else 1
    
    print(f"\nExtracting TIMIT embeddings for {num_speakers} speakers using {num_gpus} GPUs...")
    
    if not gpu_ids or num_gpus == 1:
        # Single GPU fallback - use original sequential approach
        gpu_id = gpu_ids[0] if gpu_ids else 0
        results = []
        for speaker_dir in tqdm(speaker_dirs, desc="TIMIT Speakers"):
            result = process_timit_speaker_worker(
                (str(speaker_dir), str(output_dir), model_path, pooling, gpu_id)
            )
            results.append(result)
    else:
        # Multi-GPU parallel processing
        mp_context = mp.get_context('spawn')
        
        # Prepare arguments - assign GPUs round-robin
        process_args = []
        for i, speaker_dir in enumerate(speaker_dirs):
            gpu_id = gpu_ids[i % num_gpus]
            process_args.append((str(speaker_dir), str(output_dir), model_path, pooling, gpu_id))
        
        # Process in parallel
        with mp_context.Pool(processes=num_gpus) as pool:
            results = list(tqdm(
                pool.imap(process_timit_speaker_worker, process_args),
                total=num_speakers,
                desc="TIMIT Speakers (Multi-GPU)"
            ))
    
    # Compile results
    all_metadata = {}
    for result in results:
        if result['success']:
            all_metadata[result['speaker_id']] = result
        else:
            print(f"  Warning: Failed for {result['speaker_id']}: {result.get('error', 'Unknown')}")
    
    return all_metadata


def extract_timit_embeddings(
    extractor: HuBERTEmbeddingExtractor,
    timit_dir: Path,
    output_dir: Path,
    pooling: str = 'mean'
) -> Dict[str, EmbeddingMetadata]:
    """
    Extract embeddings for all TIMIT speakers (single GPU, legacy).
    
    Args:
        extractor: HuBERT embedding extractor
        timit_dir: Path to TIMIT dataset
        output_dir: Output directory for embeddings
        pooling: Pooling strategy for utterance-level embeddings
        
    Returns:
        Dictionary of speaker_id -> metadata
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all speaker directories
    speaker_dirs = sorted([d for d in timit_dir.iterdir() 
                          if d.is_dir() and d.name.startswith('dr')])
    
    print(f"\nExtracting TIMIT embeddings for {len(speaker_dirs)} speakers...")
    
    all_metadata = {}
    
    for speaker_dir in tqdm(speaker_dirs, desc="TIMIT Speakers"):
        speaker_id = speaker_dir.name
        speaker_output = output_dir / speaker_id
        speaker_output.mkdir(exist_ok=True)
        
        # Find all wav files
        wav_files = sorted(speaker_dir.glob('*.wav'))
        
        frame_embeddings = []  # All frames for this speaker
        utterance_embeddings = []  # Pooled per utterance
        utterance_info = []
        
        for wav_file in wav_files:
            try:
                audio, sr = load_timit_audio(str(wav_file))
                
                # Extract frame-level embeddings
                frames = extractor.extract(audio, sr)
                frame_embeddings.append(frames)
                
                # Extract pooled embedding
                pooled = extractor.extract_pooled(audio, sr, pooling)
                utterance_embeddings.append(pooled)
                
                utterance_info.append({
                    'file': wav_file.name,
                    'num_frames': frames.shape[0],
                    'duration_sec': len(audio) / sr
                })
                
            except Exception as e:
                print(f"  Warning: Failed to process {wav_file.name}: {e}")
                continue
        
        if not utterance_embeddings:
            print(f"  Warning: No embeddings extracted for {speaker_id}")
            continue
        
        # Save embeddings
        utterance_embeddings = np.stack(utterance_embeddings)
        all_frames = np.concatenate(frame_embeddings, axis=0)
        
        np.save(speaker_output / 'utterance_embeddings.npy', utterance_embeddings)
        np.save(speaker_output / 'frame_embeddings.npy', all_frames)
        
        # Create metadata
        metadata = EmbeddingMetadata(
            speaker_id=speaker_id,
            dataset='timit',
            num_utterances=len(utterance_embeddings),
            embedding_dim=extractor.hidden_size,
            total_frames=all_frames.shape[0],
            model_name='facebook/hubert-large-ll60k',
            extraction_layer='last_hidden_state'
        )
        
        # Save metadata and utterance info
        with open(speaker_output / 'metadata.json', 'w') as f:
            json.dump({
                **asdict(metadata),
                'utterances': utterance_info,
                'pooling': pooling
            }, f, indent=2)
        
        all_metadata[speaker_id] = metadata
    
    return all_metadata


def get_iemocap_speaker_id(filename: str) -> str:
    """
    Extract the correct speaker ID from IEMOCAP filename.
    
    IEMOCAP filenames follow the pattern: Ses01F_impro01_F000.wav
    - Ses01F/Ses01M = Session 01, Female/Male recording perspective (NOT the speaker)
    - _F000/_M000 = The actual speaker gender (F=Female, M=Male)
    
    The correct speaker ID is: Session + Actual Gender
    Example: Ses01F_impro01_F000.wav -> Ses01_F (Session 01, Female speaker)
             Ses01F_impro01_M000.wav -> Ses01_M (Session 01, Male speaker)
             Ses01M_impro01_F000.wav -> Ses01_F (Same female speaker as above)
    
    Args:
        filename: IEMOCAP filename (e.g., "Ses01F_impro01_F000.wav")
        
    Returns:
        Correct speaker ID (e.g., "Ses01_F")
    """
    # Remove extension
    base = filename.replace('.wav', '')
    
    # Extract session number (Ses01, Ses02, etc.)
    if 'Ses' not in base:
        return None
    
    start = base.find('Ses')
    session = base[start:start+5]  # "Ses01", "Ses02", etc.
    
    # Extract actual speaker gender from the suffix (_F### or _M###)
    # The pattern is: ..._GXXX where G is gender (F/M) and XXX is utterance number
    parts = base.split('_')
    if len(parts) >= 3:
        last_part = parts[-1]  # e.g., "F000" or "M000"
        if last_part and last_part[0] in ['F', 'M']:
            actual_gender = last_part[0]
            return f"{session}_{actual_gender}"
    
    return None


def extract_iemocap_embeddings_multigpu(
    iemocap_dir: Path,
    output_dir: Path,
    model_path: str,
    pooling: str = 'mean',
    gpu_ids: List[int] = None,
    max_samples_per_speaker: Optional[int] = None
) -> Dict[str, dict]:
    """
    Extract embeddings for all IEMOCAP speakers using multiple GPUs.
    
    Args:
        iemocap_dir: Path to IEMOCAP HuggingFace dataset
        output_dir: Output directory for embeddings
        model_path: Path to HuBERT model
        pooling: Pooling strategy
        gpu_ids: List of GPU IDs to use
        max_samples_per_speaker: Limit samples per speaker (for testing)
        
    Returns:
        Dictionary of speaker_id -> metadata
    """
    from datasets import load_from_disk
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nLoading IEMOCAP dataset from {iemocap_dir}...")
    dataset = load_from_disk(str(iemocap_dir))
    train = dataset['train']
    
    # Group samples by ACTUAL speaker (Session + Gender)
    print("Grouping samples by actual speaker (Session + Gender)...")
    speaker_samples: Dict[str, List[int]] = {}
    
    for idx, sample in enumerate(tqdm(train, desc="Indexing")):
        filename = sample['file']
        speaker_id = get_iemocap_speaker_id(filename)
        
        if speaker_id:
            if speaker_id not in speaker_samples:
                speaker_samples[speaker_id] = []
            speaker_samples[speaker_id].append(idx)
    
    speakers = sorted(speaker_samples.keys())
    num_speakers = len(speakers)
    num_gpus = len(gpu_ids) if gpu_ids else 1
    
    print(f"Found {num_speakers} actual speakers (expected: 10)")
    print(f"Using {num_gpus} GPUs for extraction...")
    
    # Apply max_samples limit
    if max_samples_per_speaker:
        for spk in speakers:
            speaker_samples[spk] = speaker_samples[spk][:max_samples_per_speaker]
    
    if not gpu_ids or num_gpus == 1:
        # Single GPU fallback
        gpu_id = gpu_ids[0] if gpu_ids else 0
        results = []
        for speaker_id in tqdm(speakers, desc="IEMOCAP Speakers"):
            result = process_iemocap_speaker_worker(
                (speaker_id, speaker_samples[speaker_id], str(iemocap_dir), 
                 str(output_dir), model_path, pooling, gpu_id)
            )
            results.append(result)
    else:
        # Multi-GPU parallel processing
        mp_context = mp.get_context('spawn')
        
        # Prepare arguments - assign GPUs round-robin
        process_args = []
        for i, speaker_id in enumerate(speakers):
            gpu_id = gpu_ids[i % num_gpus]
            process_args.append(
                (speaker_id, speaker_samples[speaker_id], str(iemocap_dir),
                 str(output_dir), model_path, pooling, gpu_id)
            )
        
        # Process in parallel
        with mp_context.Pool(processes=num_gpus) as pool:
            results = list(tqdm(
                pool.imap(process_iemocap_speaker_worker, process_args),
                total=num_speakers,
                desc="IEMOCAP Speakers (Multi-GPU)"
            ))
    
    # Compile results
    all_metadata = {}
    for result in results:
        if result['success']:
            all_metadata[result['speaker_id']] = result
        else:
            print(f"  Warning: Failed for {result['speaker_id']}: {result.get('error', 'Unknown')}")
    
    return all_metadata


def extract_iemocap_embeddings(
    extractor: HuBERTEmbeddingExtractor,
    iemocap_dir: Path,
    output_dir: Path,
    pooling: str = 'mean',
    max_samples_per_speaker: Optional[int] = None
) -> Dict[str, EmbeddingMetadata]:
    """
    Extract embeddings for all IEMOCAP speakers (single GPU, legacy).
    
    IMPORTANT: Speaker ID is determined by Session + actual speaker gender,
    NOT by the recording folder (Ses01F/Ses01M).
    
    Each session has 2 speakers (1 Female, 1 Male):
    - Ses01_F = Session 01 Female speaker
    - Ses01_M = Session 01 Male speaker
    - etc. for Ses02-Ses05
    
    Total: 10 unique speakers (5 sessions × 2 genders)
    
    Args:
        extractor: HuBERT embedding extractor
        iemocap_dir: Path to IEMOCAP HuggingFace dataset
        output_dir: Output directory for embeddings
        pooling: Pooling strategy
        max_samples_per_speaker: Limit samples per speaker (for testing)
        
    Returns:
        Dictionary of speaker_id -> metadata
    """
    from datasets import load_from_disk
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nLoading IEMOCAP dataset from {iemocap_dir}...")
    dataset = load_from_disk(str(iemocap_dir))
    train = dataset['train']
    
    # Group samples by ACTUAL speaker (Session + Gender)
    print("Grouping samples by actual speaker (Session + Gender)...")
    speaker_samples: Dict[str, List[int]] = {}
    
    for idx, sample in enumerate(tqdm(train, desc="Indexing")):
        filename = sample['file']
        speaker_id = get_iemocap_speaker_id(filename)
        
        if speaker_id:
            if speaker_id not in speaker_samples:
                speaker_samples[speaker_id] = []
            speaker_samples[speaker_id].append(idx)
    
    print(f"Found {len(speaker_samples)} actual speakers (expected: 10)")
    
    all_metadata = {}
    
    for speaker_id in tqdm(sorted(speaker_samples.keys()), desc="IEMOCAP Speakers"):
        speaker_output = output_dir / speaker_id
        speaker_output.mkdir(exist_ok=True)
        
        sample_indices = speaker_samples[speaker_id]
        if max_samples_per_speaker:
            sample_indices = sample_indices[:max_samples_per_speaker]
        
        frame_embeddings = []
        utterance_embeddings = []
        utterance_info = []
        
        for idx in tqdm(sample_indices, desc=f"  {speaker_id}", leave=False):
            sample = train[idx]
            
            try:
                audio_data = sample['audio']
                audio = np.array(audio_data['array'])
                sr = audio_data['sampling_rate']
                
                # Extract embeddings
                frames = extractor.extract(audio, sr)
                frame_embeddings.append(frames)
                
                pooled = extractor.extract_pooled(audio, sr, pooling)
                utterance_embeddings.append(pooled)
                
                utterance_info.append({
                    'file': sample['file'],
                    'num_frames': frames.shape[0],
                    'duration_sec': len(audio) / sr,
                    'emotion': sample.get('major_emotion', 'unknown'),
                    'gender': sample.get('gender', 'unknown')
                })
                
            except Exception as e:
                print(f"  Warning: Failed to process sample {idx}: {e}")
                continue
        
        if not utterance_embeddings:
            print(f"  Warning: No embeddings extracted for {speaker_id}")
            continue
        
        # Save embeddings
        utterance_embeddings = np.stack(utterance_embeddings)
        all_frames = np.concatenate(frame_embeddings, axis=0)
        
        np.save(speaker_output / 'utterance_embeddings.npy', utterance_embeddings)
        np.save(speaker_output / 'frame_embeddings.npy', all_frames)
        
        # Create metadata
        metadata = EmbeddingMetadata(
            speaker_id=speaker_id,
            dataset='iemocap',
            num_utterances=len(utterance_embeddings),
            embedding_dim=extractor.hidden_size,
            total_frames=all_frames.shape[0],
            model_name='facebook/hubert-large-ll60k',
            extraction_layer='last_hidden_state'
        )
        
        with open(speaker_output / 'metadata.json', 'w') as f:
            json.dump({
                **asdict(metadata),
                'utterances': utterance_info,
                'pooling': pooling
            }, f, indent=2)
        
        all_metadata[speaker_id] = metadata
    
    return all_metadata


def extract_speech_commands_embeddings_multigpu(
    sc_dir: Path,
    output_dir: Path,
    model_path: str,
    pooling: str = "mean",
    gpu_ids: List[int] = None,
    max_samples_per_speaker: Optional[int] = None,
) -> Dict[str, dict]:
    """
    Extract embeddings for all Speech Commands speakers using multiple GPUs.

    We treat each unique `speaker_id` (parsed from filenames) as a speaker.
    Background-noise files under `_silence_` are ignored.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect WAV files from all splits
    print(f"\nIndexing Speech Commands WAV files under {sc_dir}...")
    split_names = ["train", "validation", "test"]
    speaker_samples: Dict[str, List[Path]] = {}

    for split in split_names:
        split_dir = sc_dir / split
        if not split_dir.exists():
            continue
        for wav_path in split_dir.rglob("*.wav"):
            rel = wav_path.relative_to(split_dir)
            parts = rel.parts
            if len(parts) < 2:
                continue
            word = parts[0]
            filename = parts[-1]

            # Skip background noise / silence
            if word == "_silence_":
                continue

            stem = filename.replace(".wav", "")
            # Pattern: {speaker_id}_nohash_{utterance_id}
            pieces = stem.split("_")
            if len(pieces) < 3:
                continue
            speaker_id = pieces[0]

            speaker_samples.setdefault(speaker_id, []).append(wav_path)

    speakers = sorted(speaker_samples.keys())
    num_speakers = len(speakers)
    num_gpus = len(gpu_ids) if gpu_ids else 1

    print(f"Found {num_speakers} Speech Commands speakers")
    print(f"Using {num_gpus} GPUs for extraction...")

    if max_samples_per_speaker:
        for spk in speakers:
            speaker_samples[spk] = speaker_samples[spk][:max_samples_per_speaker]

    if not gpu_ids or num_gpus == 1:
        gpu_id = gpu_ids[0] if gpu_ids else 0
        results = []
        for spk in tqdm(speakers, desc="Speech Commands Speakers"):
            result = process_speech_commands_speaker_worker(
                (spk, speaker_samples[spk], str(output_dir), model_path, pooling, gpu_id)
            )
            results.append(result)
    else:
        mp_context = mp.get_context("spawn")
        process_args = []
        for i, spk in enumerate(speakers):
            gpu_id = gpu_ids[i % num_gpus]
            process_args.append(
                (spk, speaker_samples[spk], str(output_dir), model_path, pooling, gpu_id)
            )
        with mp_context.Pool(processes=num_gpus) as pool:
            results = list(
                tqdm(
                    pool.imap(process_speech_commands_speaker_worker, process_args),
                    total=num_speakers,
                    desc="Speech Commands Speakers (Multi-GPU)",
                )
            )

    all_metadata: Dict[str, dict] = {}
    for result in results:
        spk = result.get("speaker_id", None)
        if not spk:
            continue
        if result.get("success", False):
            all_metadata[spk] = result
        else:
            print(f"  Warning: Failed for {spk}: {result.get('error', 'Unknown')}")

    return all_metadata


def main():
    parser = argparse.ArgumentParser(description='Extract speaker embeddings using HuBERT (Multi-GPU)')
    parser.add_argument('--dataset', type=str, choices=['timit', 'iemocap', 'speech_commands', 'all'],
                       default='all', help='Dataset to process')
    parser.add_argument('--output', type=str, default='embeddings/',
                       help='Output directory for embeddings')
    parser.add_argument('--model', type=str, 
                       default='models/embedding_extractor/hubert-large-ll60k',
                       help='Path to HuBERT model')
    parser.add_argument('--pooling', type=str, choices=['mean', 'max', 'last'],
                       default='mean', help='Pooling strategy for utterance embeddings')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Max samples per speaker (for testing)')
    parser.add_argument('--timit-dir', type=str, default='datasets/timit',
                       help='Path to TIMIT dataset')
    parser.add_argument('--iemocap-dir', type=str, default='datasets/iemocap/hf_dataset',
                       help='Path to IEMOCAP HuggingFace dataset')
    parser.add_argument('--speech-commands-dir', type=str, default='datasets/speech_commands/v0.02',
                       help='Path to Speech Commands v0.02 root directory')
    parser.add_argument('--num-gpus', type=int, default=8,
                       help='Maximum number of GPUs to use (default: 8, auto-detects available)')
    parser.add_argument('--single-gpu', action='store_true',
                       help='Force single GPU mode (no multiprocessing)')
    
    args = parser.parse_args()
    
    # Set multiprocessing start method
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    # Get project root (script is in scripts/)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Resolve paths relative to project root
    model_path = project_root / args.model
    output_dir = project_root / args.output
    timit_dir = project_root / args.timit_dir
    iemocap_dir = project_root / args.iemocap_dir
    sc_dir = project_root / args.speech_commands_dir
    
    print("=" * 60)
    print("WaveCommit Speaker Embedding Extraction (Multi-GPU)")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    print(f"Pooling: {args.pooling}")
    
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
            print(f"Memory fraction: {GPU_MEMORY_FRACTION * 100:.0f}%")
        else:
            print("\nNo GPUs with sufficient memory, using CPU")
    
    all_results = {}
    
    # Extract TIMIT
    if args.dataset in ['timit', 'all']:
        if timit_dir.exists():
            timit_output = output_dir / 'timit'
            
            if gpu_ids and len(gpu_ids) > 1:
                # Multi-GPU extraction
                results = extract_timit_embeddings_multigpu(
                    timit_dir, timit_output, str(model_path), args.pooling, gpu_ids
                )
            else:
                # Single GPU fallback
                extractor = HuBERTEmbeddingExtractor(str(model_path))
                results = extract_timit_embeddings(
                    extractor, timit_dir, timit_output, args.pooling
                )
            
            all_results['timit'] = results
            print(f"\nTIMIT: Extracted embeddings for {len(results)} speakers")
        else:
            print(f"\nWarning: TIMIT directory not found: {timit_dir}")
    
    # Extract IEMOCAP
    if args.dataset in ['iemocap', 'all']:
        if iemocap_dir.exists():
            iemocap_output = output_dir / 'iemocap'
            
            if gpu_ids and len(gpu_ids) > 1:
                # Multi-GPU extraction
                results = extract_iemocap_embeddings_multigpu(
                    iemocap_dir, iemocap_output, str(model_path), args.pooling,
                    gpu_ids, args.max_samples
                )
            else:
                # Single GPU fallback
                if 'extractor' not in dir():
                    extractor = HuBERTEmbeddingExtractor(str(model_path))
                results = extract_iemocap_embeddings(
                    extractor, iemocap_dir, iemocap_output, 
                    args.pooling, args.max_samples
                )
            
            all_results['iemocap'] = results
            print(f"\nIEMOCAP: Extracted embeddings for {len(results)} speakers")
        else:
            print(f"\nWarning: IEMOCAP directory not found: {iemocap_dir}")

    # Extract Speech Commands
    if args.dataset in ["speech_commands", "all"]:
        if sc_dir.exists():
            sc_output = output_dir / "speech_commands"

            if gpu_ids and len(gpu_ids) > 1:
                results = extract_speech_commands_embeddings_multigpu(
                    sc_dir,
                    sc_output,
                    str(model_path),
                    args.pooling,
                    gpu_ids,
                    args.max_samples,
                )
            else:
                # Reuse single-GPU extractor
                if "extractor" not in dir():
                    extractor = HuBERTEmbeddingExtractor(str(model_path))
                # Simple single-GPU loop using the multi-GPU helper with one GPU
                results = extract_speech_commands_embeddings_multigpu(
                    sc_dir,
                    sc_output,
                    str(model_path),
                    args.pooling,
                    gpu_ids or [0],
                    args.max_samples,
                )

            all_results["speech_commands"] = results
            print(f"\nSpeech Commands: Extracted embeddings for {len(results)} speakers")
        else:
            print(f"\nWarning: Speech Commands directory not found: {sc_dir}")
    
    # Save summary
    summary = {
        'datasets': {},
        'model': str(model_path),
        'pooling': args.pooling,
        'embedding_dim': 1024,  # HuBERT-Large hidden size
        'num_gpus_used': len(gpu_ids) if gpu_ids else 1
    }
    
    for dataset_name, results in all_results.items():
        summary['datasets'][dataset_name] = {
            'num_speakers': len(results),
            'speakers': {
                spk: {
                    'num_utterances': meta.get('num_utterances', meta.num_utterances if hasattr(meta, 'num_utterances') else 0),
                    'total_frames': meta.get('total_frames', meta.total_frames if hasattr(meta, 'total_frames') else 0)
                }
                for spk, meta in results.items()
            }
        }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'extraction_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Extraction Complete!")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")
    
    # Print summary
    for dataset_name, results in all_results.items():
        total_utterances = sum(
            m.get('num_utterances', m.num_utterances if hasattr(m, 'num_utterances') else 0) 
            for m in results.values()
        )
        total_frames = sum(
            m.get('total_frames', m.total_frames if hasattr(m, 'total_frames') else 0) 
            for m in results.values()
        )
        print(f"\n{dataset_name.upper()}:")
        print(f"  Speakers: {len(results)}")
        print(f"  Total utterances: {total_utterances}")
        print(f"  Total frames: {total_frames:,}")


if __name__ == '__main__':
    main()
