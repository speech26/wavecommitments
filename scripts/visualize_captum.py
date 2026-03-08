#!/usr/bin/env python3
"""
WaveCommit Captum Visualization

Generates visualizations for Shapley and Integrated Gradients attributions.

Usage:
    python scripts/visualize_captum.py --analysis-dir analysis/captum/er
"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


def load_pydvl_data_shapley(analysis_dir: Path) -> dict:
    """Load pyDVL Data Shapley outputs if present."""
    ds_dir = analysis_dir / 'data_shapley'
    summary_file = ds_dir / 'pydvl_data_shapley_summary.json'

    if not summary_file.exists():
        return None

    with open(summary_file) as f:
        summary = json.load(f)

    speaker_values = {}
    groups = summary.get('groups') or summary.get('speakers') or summary.get('emotions') or {}
    for group_id in groups.keys():
        value_file = ds_dir / group_id / 'pydvl_data_shapley.npy'
        if value_file.exists():
            speaker_values[group_id] = np.load(value_file)

    global_values_file = ds_dir / 'pydvl_data_shapley_values.npy'
    global_values = np.load(global_values_file) if global_values_file.exists() else None

    return {
        'summary': summary,
        'speaker_values': speaker_values,
        'global_values': global_values,
        'data_dir': ds_dir,
    }


def _data_shapley_group_label(summary: dict) -> str:
    group_type = summary.get('group_type', 'speaker')
    return 'Emotion' if group_type == 'emotion' else 'Speaker'


def load_attributions(analysis_dir: Path, method: str) -> dict:
    """Load attribution data from saved files."""
    method_dir = analysis_dir / method
    summary_file = method_dir / f'{method}_summary.json'
    
    if not summary_file.exists():
        return None
    
    with open(summary_file) as f:
        summary = json.load(f)
    
    # Load full attributions
    attributions = {}
    for speaker_id in summary.keys():
        attr_file = method_dir / speaker_id / f'{method}_attributions.npy'
        if attr_file.exists():
            attributions[speaker_id] = np.load(attr_file)
    
    return {'summary': summary, 'attributions': attributions}


def load_attributions_v2(analysis_dir: Path, method_dir: str, method_file: str) -> dict:
    """Load attribution data with separate dir/file naming."""
    dir_path = analysis_dir / method_dir
    summary_file = dir_path / f'{method_file}_summary.json'
    
    if not summary_file.exists():
        return None
    
    with open(summary_file) as f:
        summary = json.load(f)
    
    # Load full attributions
    attributions = {}
    for speaker_id in summary.keys():
        attr_file = dir_path / speaker_id / f'{method_file}_attributions.npy'
        if attr_file.exists():
            attributions[speaker_id] = np.load(attr_file)
    
    return {'summary': summary, 'attributions': attributions}


def plot_feature_importance_heatmap(data: dict, output_path: Path, 
                                    method_name: str, top_k: int = 50):
    """Plot heatmap of top feature attributions across speakers."""
    summary = data['summary']
    speakers = sorted(summary.keys())
    
    # Aggregate mean attributions
    all_means = []
    for spk in speakers:
        mean_attr = np.array(summary[spk]['mean_attribution'])
        all_means.append(mean_attr)
    
    all_means = np.stack(all_means)  # (num_speakers, 1024)
    
    # Get top features by absolute attribution
    global_importance = np.abs(all_means).mean(axis=0)
    top_indices = global_importance.argsort()[-top_k:][::-1]
    
    # Create heatmap data
    heatmap_data = all_means[:, top_indices]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Custom colormap (blue-white-red)
    cmap = LinearSegmentedColormap.from_list('attribution', 
                                              ['#2166ac', '#f7f7f7', '#b2182b'])
    
    # Normalize around zero
    vmax = np.abs(heatmap_data).max()
    
    sns.heatmap(heatmap_data, 
                ax=ax,
                cmap=cmap,
                center=0,
                vmin=-vmax, vmax=vmax,
                xticklabels=[f'F{i}' for i in top_indices],
                yticklabels=speakers,
                cbar_kws={'label': 'Attribution'})
    
    ax.set_xlabel(f'Top {top_k} Feature Dimensions')
    ax.set_ylabel('Speaker')
    ax.set_title(f'{method_name} Feature Attributions by Speaker')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def plot_speaker_comparison(data: dict, output_path: Path, method_name: str):
    """Plot bar chart comparing attribution magnitudes across speakers."""
    summary = data['summary']
    speakers = sorted(summary.keys())
    
    # Compute mean absolute attribution per speaker
    mean_abs_attr = []
    for spk in speakers:
        mean_attr = np.array(summary[spk]['mean_attribution'])
        mean_abs_attr.append(np.abs(mean_attr).mean())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.bar(range(len(speakers)), mean_abs_attr, color='steelblue')
    
    ax.set_xticks(range(len(speakers)))
    ax.set_xticklabels(speakers, rotation=45, ha='right')
    ax.set_xlabel('Speaker')
    ax.set_ylabel('Mean |Attribution|')
    ax.set_title(f'{method_name}: Mean Absolute Attribution by Speaker')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def plot_attribution_distribution(data: dict, output_path: Path, method_name: str):
    """Plot distribution of attributions across all samples."""
    attributions = data['attributions']
    
    # Flatten all attributions
    all_attr = []
    for spk, attr in attributions.items():
        if attr is None:
            continue
        flat_attr = attr.flatten()
        if flat_attr.size == 0:
            continue
        all_attr.append(flat_attr)

    if not all_attr:
        print(f"Skipping: no non-empty attribution arrays for {method_name} distribution plot")
        return
    
    all_attr = np.concatenate(all_attr)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(all_attr, bins=100, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=1)
    axes[0].set_xlabel('Attribution Value')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'{method_name}: Attribution Distribution')
    
    # Box plot per speaker (sample)
    speaker_data = []
    speaker_labels = []
    for spk in sorted(attributions.keys())[:8]:  # First 8 speakers
        attr = attributions[spk].flatten()
        if attr.size == 0:
            continue
        # Sample for visualization
        if len(attr) > 1000:
            attr = np.random.choice(attr, 1000, replace=False)
        speaker_data.append(attr)
        speaker_labels.append(spk)

    if not speaker_data:
        print(f"Skipping: no speaker attribution samples for {method_name} box plot")
        return
    
    bp = axes[1].boxplot(speaker_data, labels=speaker_labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightsteelblue')
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[1].set_xlabel('Speaker')
    axes[1].set_ylabel('Attribution Value')
    axes[1].set_title(f'{method_name}: Attribution by Speaker')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def plot_top_features(data: dict, output_path: Path, method_name: str, top_k: int = 20):
    """Plot top influential features."""
    summary = data['summary']
    speakers = sorted(summary.keys())
    
    # Aggregate mean attributions
    all_means = []
    for spk in speakers:
        mean_attr = np.array(summary[spk]['mean_attribution'])
        all_means.append(mean_attr)
    
    all_means = np.stack(all_means)
    global_importance = np.abs(all_means).mean(axis=0)
    
    # Get top features
    top_indices = global_importance.argsort()[-top_k:][::-1]
    top_values = global_importance[top_indices]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.barh(range(len(top_indices)), top_values[::-1], color='steelblue')
    ax.set_yticks(range(len(top_indices)))
    ax.set_yticklabels([f'Feature {i}' for i in top_indices[::-1]])
    ax.set_xlabel('Mean |Attribution|')
    ax.set_title(f'{method_name}: Top {top_k} Most Influential Features')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def plot_data_shapley_speaker_stats(data: dict, output_path: Path):
    """Plot mean and std Data Shapley values by speaker."""
    speaker_summary = data['summary'].get('groups') or data['summary'].get('speakers') or data['summary'].get('emotions') or {}
    speakers = sorted(speaker_summary.keys())
    if not speakers:
        return
    group_label = _data_shapley_group_label(data['summary'])

    means = [speaker_summary[s]['shapley_mean'] for s in speakers]
    stds = [speaker_summary[s]['shapley_std'] for s in speakers]
    colors = ['#b2182b' if v < 0 else '#2166ac' for v in means]

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(range(len(speakers)), means, yerr=stds, capsize=3, color=colors, alpha=0.9)
    ax.axhline(y=0.0, color='black', linewidth=1.0, linestyle='--')
    ax.set_xticks(range(len(speakers)))
    ax.set_xticklabels(speakers, rotation=45, ha='right')
    ax.set_ylabel('Mean Data Shapley')
    ax.set_xlabel(group_label)
    ax.set_title(f'pyDVL Data Shapley Mean +/- Std by {group_label}')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_data_shapley_distribution(data: dict, output_path: Path):
    """Plot per-speaker Data Shapley distributions."""
    speaker_values = data['speaker_values']
    speakers = sorted(speaker_values.keys())
    if not speakers:
        return
    group_label = _data_shapley_group_label(data['summary'])

    values = [speaker_values[s] for s in speakers]

    fig, ax = plt.subplots(figsize=(13, 6))
    try:
        bp = ax.boxplot(values, tick_labels=speakers, patch_artist=True, showfliers=True)
    except TypeError:
        bp = ax.boxplot(values, labels=speakers, patch_artist=True, showfliers=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#9ecae1')
    ax.axhline(y=0.0, color='black', linewidth=1.0, linestyle='--')
    ax.set_ylabel('Data Shapley Value')
    ax.set_xlabel(group_label)
    ax.set_title(f'pyDVL Data Shapley Distribution by {group_label}')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_data_shapley_ranked_samples(data: dict, output_path: Path, top_k: int = 20):
    """Plot top and bottom training samples by Data Shapley value."""
    global_values = data.get('global_values')
    if global_values is None or len(global_values) == 0:
        speaker_values = data['speaker_values']
        if not speaker_values:
            return
        global_values = np.concatenate([speaker_values[s] for s in sorted(speaker_values.keys())])

    order = np.argsort(global_values)
    k = min(top_k, len(global_values) // 2)
    if k == 0:
        return

    low_idx = order[:k]
    high_idx = order[-k:]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].barh(range(k), global_values[low_idx], color='#b2182b')
    axes[0].set_yticks(range(k))
    axes[0].set_yticklabels([str(i) for i in low_idx])
    axes[0].set_xlabel('Data Shapley')
    axes[0].set_title(f'Bottom {k} Samples')

    axes[1].barh(range(k), global_values[high_idx], color='#2166ac')
    axes[1].set_yticks(range(k))
    axes[1].set_yticklabels([str(i) for i in high_idx])
    axes[1].set_xlabel('Data Shapley')
    axes[1].set_title(f'Top {k} Samples')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_data_shapley_heatmap(data: dict, output_path: Path):
    """Plot speaker x utterance-index Data Shapley heatmap."""
    speaker_values = data['speaker_values']
    speakers = sorted(speaker_values.keys())
    if not speakers:
        return
    group_label = _data_shapley_group_label(data['summary'])

    max_len = max(len(speaker_values[s]) for s in speakers)
    heatmap = np.full((len(speakers), max_len), np.nan, dtype=np.float32)

    for i, spk in enumerate(speakers):
        vals = speaker_values[spk]
        heatmap[i, :len(vals)] = vals

    vmax = np.nanmax(np.abs(heatmap)) if np.any(np.isfinite(heatmap)) else 1.0
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.heatmap(
        heatmap,
        ax=ax,
        cmap=LinearSegmentedColormap.from_list('shapley', ['#b2182b', '#f7f7f7', '#2166ac']),
        center=0.0,
        vmin=-vmax,
        vmax=vmax,
        xticklabels=[f'U{i}' for i in range(max_len)],
        yticklabels=speakers,
        cbar_kws={'label': 'Data Shapley'},
    )
    ax.set_xlabel('Train Utterance Index (per speaker split)')
    ax.set_ylabel(group_label)
    ax.set_title(f'pyDVL Data Shapley Heatmap ({group_label} x Utterance)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize Captum analysis results')
    parser.add_argument('--analysis-dir', type=str, default='analysis/captum/er',
                       help='Path to analysis directory')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for visualizations')
    parser.add_argument('--data-shapley-only', action='store_true',
                       help='Generate only pyDVL Data Shapley visualizations')
    
    args = parser.parse_args()
    
    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    analysis_dir = project_root / args.analysis_dir
    output_dir = Path(args.output_dir) if args.output_dir else analysis_dir / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("WaveCommit Captum Visualization")
    print("=" * 60)
    print(f"Analysis: {analysis_dir}")
    print(f"Output: {output_dir}")
    
    # Set global seaborn style for all plots.
    sns.set_theme(style='darkgrid')
    
    # Process feature-attribution methods unless explicitly disabled.
    if args.data_shapley_only:
        print("\nFeature-level Shapley/IG plotting disabled (--data-shapley-only).")
    else:
        methods = [
            ('shapley', 'shapley', 'Shapley'),
            ('integrated_gradients', 'integratedgradients', 'Integrated Gradients')
        ]
        
        for method_dir, method_file, method_name in methods:
            data = load_attributions_v2(analysis_dir, method_dir, method_file)
            
            if data is None:
                print(f"\nNo {method_dir} data found, skipping...")
                continue
            
            print(f"\nProcessing {method_name} results...")
            
            # Generate visualizations
            plot_feature_importance_heatmap(
                data, output_dir / f'{method_dir}_heatmap.png', method_name
            )
            
            plot_speaker_comparison(
                data, output_dir / f'{method_dir}_speaker_comparison.png', method_name
            )
            
            plot_attribution_distribution(
                data, output_dir / f'{method_dir}_distribution.png', method_name
            )
            
            plot_top_features(
                data, output_dir / f'{method_dir}_top_features.png', method_name
            )

    data_shapley_data = load_pydvl_data_shapley(analysis_dir)
    if data_shapley_data is None:
        print("\nNo data_shapley pyDVL data found, skipping...")
    else:
        print("\nProcessing Data Shapley (pyDVL) results...")
        plot_data_shapley_speaker_stats(
            data_shapley_data,
            output_dir / 'data_shapley_speaker_stats.png',
        )
        plot_data_shapley_distribution(
            data_shapley_data,
            output_dir / 'data_shapley_distribution.png',
        )
        plot_data_shapley_ranked_samples(
            data_shapley_data,
            output_dir / 'data_shapley_top_bottom_samples.png',
        )
        plot_data_shapley_heatmap(
            data_shapley_data,
            output_dir / 'data_shapley_heatmap.png',
        )
    
    print("\n" + "=" * 60)
    print("Visualization Complete!")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
