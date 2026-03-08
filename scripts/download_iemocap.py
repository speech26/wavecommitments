#!/usr/bin/env python3
"""
Download IEMOCAP dataset from HuggingFace.
Source: https://huggingface.co/datasets/AbstractTTS/IEMOCAP
"""

import os
from pathlib import Path
from datasets import load_dataset

def main():
    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    iemocap_dir = project_root / 'datasets' / 'iemocap'
    
    print("=" * 60)
    print("IEMOCAP Dataset Downloader")
    print("=" * 60)
    
    print(f"\nTarget directory: {iemocap_dir}")
    print("Source: huggingface.co/datasets/AbstractTTS/IEMOCAP")
    
    # Download dataset
    print("\nDownloading IEMOCAP dataset...")
    try:
        dataset = load_dataset("AbstractTTS/IEMOCAP", trust_remote_code=True)
        
        print(f"\nDataset loaded successfully!")
        print(f"Splits: {list(dataset.keys())}")
        
        for split_name, split_data in dataset.items():
            print(f"  {split_name}: {len(split_data)} samples")
            if len(split_data) > 0:
                print(f"    Features: {list(split_data.features.keys())}")
        
        # Save dataset to disk
        save_path = iemocap_dir / 'hf_dataset'
        print(f"\nSaving to {save_path}...")
        dataset.save_to_disk(str(save_path))
        
        print("\n✓ IEMOCAP dataset downloaded and saved!")
        
        # Get size
        size = sum(f.stat().st_size for f in save_path.rglob('*') if f.is_file())
        print(f"  Total size: {size / (1024 * 1024):.1f} MB")
        
    except Exception as e:
        print(f"\n✗ Failed to download IEMOCAP: {e}")
        print("\nNote: The IEMOCAP dataset may require special access.")
        print("Please check: https://huggingface.co/datasets/AbstractTTS/IEMOCAP")
        return 1
    
    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)
    
    return 0

if __name__ == '__main__':
    exit(main())
