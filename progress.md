# WaveCommit Project Progress

## Current Stage (2026-02-19): Data Shapley Model Alignment + SID Head Integrity
### Status: COMPLETE (finetune-head default integrated + validated)

### Completed in this stage
- [x] Fixed model downloader to fetch prediction heads via `AutoModelForAudioClassification` (instead of base `AutoModel`)
- [x] Added strict prediction-head artifact validation for required keys:
  - `classifier.projector.weight`
  - `classifier.classifier.weight`
- [x] Added hard-fail guard in Data Shapley default path when head weights are incomplete
- [x] Implemented pyDVL finetune-head mode:
  - starts from pretrained SUPERB projector + classifier initialization
  - fine-tunes projector+classifier on each pyDVL subset fit
- [x] Switched pyDVL default model to `--pydvl-model finetune_head`
- [x] Kept old pyDVL MLP path for reliability comparison:
  - `--pydvl-model legacy_sklearn_mlp`
- [x] Kept frozen-head top-classifier path for comparison:
  - `--pydvl-model frozen_head_top`
- [x] Added unit tests for new behavior and downloader/model routing
- [x] Fixed prediction-head key validation bug:
  - accepted key format: `projector.weight` + `classifier.weight`
  - backward-compatible format retained: `classifier.projector.weight` + `classifier.classifier.weight`
- [x] Fixed frozen-head feature extraction path to use `model.projector -> model.classifier`
- [x] Added safe fallback for unfitted pyDVL subset models (avoids repeated `Model not fitted` warnings)
- [x] Added optional multi-GPU DataParallel for pyDVL finetune-head mode
- [x] Verified TIMIT SID Data Shapley smoke run in finetune-head mode (passes)
- [x] Updated `README.md`, `tests.md`, and parser docs for new defaults/flags
- [x] Extended `visualize_captum.py` to support pyDVL Data Shapley outputs
  - speaker mean/std bar chart
  - per-speaker distribution plot
  - top/bottom sample ranking plot
  - speaker x sample heatmap
- [x] Verified visualization generation for TIMIT SID Data Shapley outputs
- [x] Switched visualization theme to seaborn `darkgrid` for all plots
- [x] Added technical/scientific report for Data Shapley visualizations:
  - `analysis/captum/sid/visualizations/README.md`
  - structured as question -> hypothesis -> observation -> conclusion per plot
- [x] Added pyDVL tqdm progress/ETA in `captum_analysis.py` (enabled by default)
- [x] Added `full_process.sh --step 6.2.6`:
  - runs Data Shapley for TIMIT(SID labels) + IEMOCAP(ER full emotion labels)
  - shows outer tqdm progress/ETA across both datasets
- [x] Added full-emotion IEMOCAP label mode for pyDVL Data Shapley:
  - `--pydvl-task emotion` uses metadata `utterances[].emotion`
  - `--pydvl-task auto` resolves to emotion mode for IEMOCAP+ER and speaker mode otherwise
- [x] Added matching-head default for pyDVL feature initialization:
  - `--pydvl-feature-head auto` (default) resolves to current `--head`
  - explicit `--pydvl-feature-head sid|er|ks|ic` override remains available for reliability comparisons
- [x] Regenerated latest ER/SID visualization outputs from current artifacts:
  - `analysis/captum/er/visualizations/*` updated to include ER Data Shapley plots
  - `analysis/captum/sid/visualizations/*` refreshed from latest SID Data Shapley summary
- [x] Added robustness in `visualize_captum.py`:
  - attribution distribution plotting now skips empty arrays instead of raising `ValueError`
- [x] Added isolated latest Data Shapley visualization mode:
  - new CLI flag: `--data-shapley-only` in `scripts/visualize_captum.py`
  - skips SHAP/IG plotting and generates only pyDVL Data Shapley figures
  - used to produce:
    - `analysis/captum/er/visualizations/latest_data_shapley/*`
    - `analysis/captum/sid/visualizations/latest_data_shapley/*`
- [x] Added unit-test coverage for data-shapley-only mode:
  - `test_main_data_shapley_only_skips_feature_methods`
- [x] Added experimental single-GPU pyDVL concurrency controls (opt-in):
  - new flag: `--pydvl-allow-single-gpu-concurrency`
  - new flag: `--pydvl-joblib-backend auto|loky|threading`
  - multi-GPU DataParallel behavior preserved (`pydvl-jobs` forced to 1)
  - single-GPU `pydvl-jobs > 1` now explicitly gated as experimental
- [x] Added benchmark utility for concurrency sweeps:
  - new script: `scripts/benchmark_pydvl_single_gpu_concurrency.py`
  - compares jobs sweep (e.g., 1/2/4) and records runtime summary JSON
- [x] Added unit tests for parallelism resolution + benchmark command parsing:
  - `tests/test_data_shapley_pipeline.py` (parallelism/backend resolver tests)
  - `tests/test_benchmark_single_gpu_concurrency.py`
- [x] Added pyDVL profile-grid benchmark utility:
  - new script: `scripts/benchmark_pydvl_profiles.py`
  - profile sets: `safe`, `aggressive`, `all`
  - orchestrates 8-GPU baseline + single-GPU overlap runs and records runtime/effective-jobs metadata
- [x] Added unit tests for profile-grid runner:
  - `tests/test_benchmark_pydvl_profiles.py`
- [x] Expanded README command matrix for benchmark/profile execution:
  - documented `iemocap+sid`, `timit+sid`, and `timit+er` runnable examples
  - retained explicit `iemocap+er` full-emotion examples
- [x] Added speaker-clustered pyDVL Data Shapley aggregation (both task modes):
  - keeps existing per-group outputs unchanged
  - stores `train_speaker_ids.npy` for train-split sample alignment
  - writes per-(label,speaker) vectors under:
    - `data_shapley/speaker_label_contributions/{label_or_speaker}/{speaker_id}/pydvl_data_shapley.npy`
  - stores summary stats in `pydvl_data_shapley_summary.json` field:
    - `speaker_label_contributions`
- [x] Switched pyDVL TMC-Shapley defaults to lower-variance settings (standard method unchanged):
  - default `--min-updates`: `200` (was `100`)
  - default `--rtol`: `0.03` (was `0.05`)
  - old behavior remains available by explicitly passing prior values
- [x] Added optional one-class subset mitigation for pyDVL finetune mode:
  - new flag: `--pydvl-enforce-multiclass-subsets`
  - behavior: augments one-class subset fits with anchor samples from another class
  - goal: reduce degenerate one-class subset training events/logs
  - note: this modifies standard TMC-Shapley estimator behavior (opt-in only)

### Future Tasks Backlog (from current findings)
- [ ] Add a dedicated script/report to compare Data Shapley reliability between:
  - `finetune_head` vs `frozen_head_top` vs `legacy_sklearn_mlp`
- [ ] Add optional normalization views for speaker-clustered contributions
  - example: per-label speaker shares that sum to 1
- [ ] Add visualization support for approximate Data Shapley outputs (`intra`, `inter`)
- [ ] Separate analysis summaries by dataset to avoid mixed-state overwrite in shared output folders
- [ ] Extend TracIn from checkpoint generation to full influence score computation
- [ ] Add end-to-end GPU test for pyDVL finetune-head DataParallel behavior

## Phase 1: Repository Exploration & Setup

### Status: COMPLETE ✓

### Tasks:
- [x] Explore WaveCommit folder structure
- [x] Read all README files
- [x] Check conda environment (wavecommit exists but nearly empty)
- [x] Verify Rust/Cargo availability (cargo 1.90.0)
- [x] Identify missing TensorCommitment components (CleanPegasus, merkle, terkleLib missing)
- [x] Sync complete TensorCommitment library to WaveCommit/libs
- [x] Install all required Python packages
- [x] Build TensorCommitment Rust libraries (4 libs: tensorcommitments, terkle, pegasus_verkle, multibranch_merkle)
- [x] Verify all imports and GPU availability (8 GPUs)
- [x] Download HuggingFace models (HuBERT backbone + 4 prediction heads, ~6GB total)
- [x] Download IEMOCAP dataset (10,039 samples, 1.37GB)

### Observations:
- TIMIT dataset: Present with 16 speakers (dr1-dr8, male/female)
- IEMOCAP dataset: Folder exists, needs HuggingFace download
- Models folder: Empty, needs HuBERT models downloaded
- TensorCommitment in WaveCommit/libs is incomplete copy

### Required HuggingFace Models:
- Embedding: `facebook/hubert-large-ll60k`
- Prediction Heads:
  - `superb/hubert-large-superb-er` (emotion recognition)
  - `superb/hubert-large-superb-sid` (speaker identification)
  - `superb/hubert-large-superb-ks` (keyword spotting)
  - `superb/hubert-large-superb-ic` (intent classification)

---

### Verification Tests:
- All Python imports: PASS
- TensorCommitment commit/prove/verify: PASS
- CUDA available: 8 GPUs
- HuBERT config: hidden_size=1024, num_layers=24

---

## Phase 2: Dataset Structure Extraction
### Status: COMPLETE ✓

### Tasks:
- [x] Analyze TIMIT folder structure (16 speakers, 10 utterances each)
- [x] Analyze IEMOCAP HuggingFace dataset (10 speakers, 10,039 samples)
- [x] Document speaker naming conventions
- [x] Document file formats and features
- [x] Create speaker ID mapping for embedding extraction
- [x] Save to `datasets/DATASET_STRUCTURE.md`

### Key Findings:
- **TIMIT**: 16 speakers from 8 dialect regions, 160 total utterances
- **IEMOCAP**: 10 speakers across 5 sessions, 10,039 emotional speech samples
- Both datasets have balanced gender distribution
- IEMOCAP includes emotion labels (frustrated, excited, neutral, angry, sad, etc.)

---

---

## Phase 3: Speaker Embedding Extraction
### Status: COMPLETE ✓

### Tasks:
- [x] Load TIMIT, parse speakers from folder names (16 speakers)
- [x] Load IEMOCAP, parse speakers from filenames (10 speakers)
- [x] Load HuBERT-Large backbone to GPU
- [x] Build embedding pools per speaker (both datasets)
- [x] Create extraction script (`scripts/extract_embeddings.py`)
- [x] Save embeddings to speaker folders
- [x] Add multi-GPU support for faster extraction

### Multi-GPU Support (Added Feb 17, 2026):
- **Parallelization**: Per-speaker (each GPU processes different speakers)
- **Auto-detection**: Uses all available GPUs with sufficient memory
- **Configuration**: `--num-gpus N` argument (default: 8, auto-detects)
- **Memory**: 90% GPU memory fraction per worker
- **TIMIT speedup**: ~33s with 8 GPUs (vs ~2 min single GPU)
- **IEMOCAP speedup**: Proportional to number of GPUs used

### Results:
| Dataset | Speakers | Utterances | Total Frames | Size |
|---------|----------|------------|--------------|------|
| TIMIT | 16 | 160 | 24,385 | 97 MB |
| IEMOCAP | 10 | 10,039 | 2,231,241 | 8.6 GB |
| **Total** | **26** | **10,199** | **2,255,626** | **8.7 GB** |

### Output Structure:
```
embeddings/
├── timit/
│   └── dr{1-8}-{speaker}/
│       ├── utterance_embeddings.npy  # (num_utterances, 1024)
│       ├── frame_embeddings.npy       # (total_frames, 1024)
│       └── metadata.json
├── iemocap/
│   └── Ses{01-05}{F/M}/
│       ├── utterance_embeddings.npy
│       ├── frame_embeddings.npy
│       └── metadata.json
└── extraction_summary.json
```

---

---

## Phase 4: TensorCommitment Integration
### Status: COMPLETE (TIMIT only) ✓

### Tasks:
- [x] Build commitment trees for TIMIT (16 speakers)
- [ ] Build commitment trees for IEMOCAP (deferred to post-Phase 5)
- [ ] Test all verification operations (deferred to post-Phase 5)
- [ ] Validate embedding pool quality (deferred to post-Phase 5)

### TIMIT Commitment Results:
| Speaker | Elements | Variables | Degree | Commitment Size |
|---------|----------|-----------|--------|-----------------|
| All 16 speakers | 10,240 | 4 | 11 | 32 bytes |

### Commitment Parameters:
- **Embedding level**: Utterance (pooled mean)
- **Scale factor**: 10^8 (8 decimal places preserved)
- **Hypercube shape**: 4 variables, degree bound 11
- **Capacity**: 14,641 (11^4)
- **Padding**: ~4,400 elements per speaker

### Output Structure:
```
commitments/timit/
├── dr{1-8}-{speaker}/
│   ├── commitment.txt           # 32-byte commitment (hex)
│   └── commitment_metadata.json # Parameters and stats
└── commitment_summary.json
```

### Deferred to Post-Phase 5:
- IEMOCAP commitment building
- Proof generation and verification tests
- Embedding quality validation

---

---

## Phase 5: Captum Analysis (TracIn & Shapley)
### Status: COMPLETE (TIMIT + ER Head) ✓

### Tasks:
- [x] Run Integrated Gradients analysis (16 speakers, 160 samples)
- [x] Run Shapley value analysis (16 speakers, 160 samples)
- [x] Analyze per recording and speaker for ER head
- [x] Prepare minimal exploration dataset (TIMIT)
- [x] Generate checkpoints via short fine-tuning (5 epochs)
- [x] Generate exploration visualizations (8 plots)

### Analysis Results (Multi-GPU: 8 GPUs):
| Analysis | Method | Samples | Single GPU | 8 GPUs |
|----------|--------|---------|------------|--------|
| Feature Attribution | Integrated Gradients | 160 | ~2s | ~18s* |
| Feature Attribution | Shapley Values (n=10) | 160 | ~13 min | ~2.4 min |
| Checkpoint Generation | TracIn (5 epochs) | 160 | ~9s | N/A |

*Multi-GPU adds overhead for fast methods; best for Shapley

### Multi-GPU Configuration:
- **Memory fraction**: 90% per GPU
- **Parallelization**: 1 speaker per GPU (parallel across speakers)
- **Auto-detection**: Uses only GPUs with >2GB free memory

### Output Structure:
```
analysis/captum/er/
├── integrated_gradients/
│   ├── dr{1-8}-{speaker}/
│   │   └── integratedgradients_attributions.npy
│   └── integratedgradients_summary.json
├── shapley/
│   ├── dr{1-8}-{speaker}/
│   │   └── shapley_attributions.npy
│   └── shapley_summary.json
├── tracin/
│   ├── checkpoints/
│   │   └── checkpoint_epoch_{1-5}.pt
│   └── checkpoint_info.json
├── visualizations/
│   ├── shapley_heatmap.png
│   ├── shapley_distribution.png
│   ├── shapley_speaker_comparison.png
│   ├── shapley_top_features.png
│   ├── integrated_gradients_heatmap.png
│   ├── integrated_gradients_distribution.png
│   ├── integrated_gradients_speaker_comparison.png
│   └── integrated_gradients_top_features.png
└── analysis_summary.json
```

### Key Findings:
- Both Shapley and IG show consistent feature importance patterns across speakers
- Top features identified for emotion recognition task
- Checkpoints generated for future TracIn influence analysis

---

## Phase 6: IEMOCAP Integration
### Status: COMPLETE ✓

### Bug Fix (Feb 17, 2026):
**Problem**: Initial IEMOCAP parsing incorrectly used folder names (`Ses01F`, `Ses01M`) as speaker IDs.
These represent recording perspectives, NOT actual speakers. The same female/male speaker pair
appears in both `Ses01F` and `Ses01M` folders.

**Solution**: Parse actual speaker identity from filename suffix (`_F###` = Female, `_M###` = Male).
Correct speaker IDs are now `Ses01_F`, `Ses01_M`, etc. (Session + Gender).

**Impact**: 
- Before: 10 "speakers" (Ses01F, Ses01M, ..., Ses05F, Ses05M) - incorrectly grouped by recording folder
- After: 10 speakers (Ses01_F, Ses01_M, ..., Ses05_F, Ses05_M) - correctly grouped by actual identity

### Tasks:
- [x] Fix IEMOCAP speaker parsing in `extract_embeddings.py`
- [x] Add multi-GPU support to embedding extraction
- [x] Re-extract IEMOCAP embeddings (10 speakers, 10,039 samples, ~170s with 8 GPUs)
- [x] Rebuild commitment trees for IEMOCAP (10 speakers, ~21s)
- [x] Re-run Integrated Gradients on IEMOCAP (~30s with 8 GPUs)
- [x] Generate TracIn checkpoints for IEMOCAP (~18s)
- [x] Implement Data Shapley (utterance-level attribution)
- [x] Run Data Shapley on TIMIT (~18s with 8 GPUs)
- [x] Run Data Shapley on IEMOCAP (~110s with 8 GPUs)
- [ ] Regenerate visualizations with Data Shapley

### Data Shapley Implementation (Added Feb 18, 2026):

**Previous (Feature-level Shapley):**
- `ShapleyValueSampling` computes importance of each feature (1024 dimensions)
- Output: 1024 Shapley values per utterance
- Answers: "Which embedding dimensions are important for prediction?"

**New (Data Shapley - Utterance-level):**
- Computes importance of each utterance as a whole data point
- Output: 1 Shapley value per utterance
- Two types implemented:
  1. **Intra-speaker**: Leave-one-out influence within the speaker (utility = avg confidence)
  2. **Inter-speaker**: Contribution to speaker separation (utility = centroid distance)

### Data Shapley Performance (8 GPUs):
| Dataset | Speakers | Samples | Time | Permutations |
|---------|----------|---------|------|--------------|
| TIMIT | 16 | 160 | ~18s | 100 |
| IEMOCAP | 10 | 10,039 | ~110s | 100 |

### Data Shapley Output:
```
analysis/captum/er/data_shapley/
├── {speaker_id}/
│   ├── data_shapley_intra.npy  # (N,) values - within-speaker importance
│   └── data_shapley_inter.npy  # (N,) values - cross-speaker importance
└── data_shapley_summary.json
```

### Multi-GPU Embedding Extraction (Added Feb 18, 2026):
| Dataset | Speakers | Samples | Time (8 GPUs) |
|---------|----------|---------|---------------|
| TIMIT | 16 | 160 | ~33s |
| IEMOCAP | 10 | 10,039 | ~170s |

### IEMOCAP Commitment Results (Corrected):
| Speaker | Elements | Variables | Degree | Commitment Size |
|---------|----------|-----------|--------|-----------------|
| Ses01_F | 893,952 | 4 | 31 | 32 bytes |
| Ses01_M | 968,704 | 6 | 10 | 32 bytes |
| Ses02_F | 879,616 | 4 | 31 | 32 bytes |
| Ses02_M | 974,848 | 6 | 10 | 32 bytes |
| Ses03_F | 1,073,152 | 5 | 17 | 32 bytes |
| Ses03_M | 1,114,112 | 5 | 17 | 32 bytes |
| Ses04_F | 1,010,688 | 4 | 32 | 32 bytes |
| Ses04_M | 1,142,784 | 5 | 17 | 32 bytes |
| Ses05_F | 1,057,792 | 5 | 17 | 32 bytes |
| Ses05_M | 1,164,288 | 5 | 17 | 32 bytes |

---

## Deferred Tasks (Post-Phase 6)
### Status: PENDING

### Extended Verification:
- [ ] Proof generation and verification tests for TIMIT
- [ ] Proof generation and verification tests for IEMOCAP
- [ ] Tamper detection validation
- [ ] Cross-dataset commitment comparison

### Additional Prediction Heads:
- [ ] SID (Speaker Identification) analysis
- [ ] KS (Keyword Spotting) analysis  
- [ ] IC (Intent Classification) analysis

### TracIn Influence Analysis:
- [ ] Compute training influence scores
- [ ] Identify most influential samples per prediction

---

## Last Updated: 2026-02-18
