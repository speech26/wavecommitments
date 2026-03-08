#!/bin/bash
#===============================================================================
# WaveCommit Full Process Script
#
# This script reproduces the entire WaveCommit pipeline from scratch.
# It includes all steps from environment setup to final analysis.
#
# Prerequisites:
#   - Conda installed
#   - CUDA-capable GPU(s)
#   - Rust toolchain (cargo, maturin)
#   - ~20GB disk space for models, embeddings, and analysis
#
# Usage:
#   ./full_process.sh              # Run all steps
#   ./full_process.sh --phase 3    # Run from Phase 3 onwards
#   ./full_process.sh --step 3.1   # Run specific step only
#
# Author: WaveCommit Team
# Last Updated: 2026-02-17
#===============================================================================

set -e  # Exit on error

# Configuration
CONDA_ENV="wavecommit"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NUM_GPUS=8
SHAPLEY_N_SAMPLES=10
TRACIN_EPOCHS=5
SID_PYDVL_JOBS=8

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo ""
    echo "============================================================"
    echo -e "${GREEN}STEP $1: $2${NC}"
    echo "============================================================"
}

# Parse arguments
PHASE=""
STEP=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --phase)
            PHASE="$2"
            shift 2
            ;;
        --step)
            STEP="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--phase N] [--step N.N]"
            echo "  --phase N    Start from phase N (1-6)"
            echo "  --step N.N   Run only step N.N"
            echo ""
            echo "Phases:"
            echo "  1: Environment Setup"
            echo "  2: Dataset Structure"
            echo "  3: Embedding Extraction"
            echo "  4: Commitment Building"
            echo "  5: TIMIT Captum Analysis"
            echo "  6: IEMOCAP Captum Analysis"
            echo ""
            echo "Useful step for this combined setting:"
            echo "  6.2.6: Data Shapley for TIMIT(SID) + IEMOCAP(ER full emotions) with tqdm ETA"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

should_run_step() {
    local step_num=$1
    local step_phase=${step_num%%.*}
    
    if [[ -n "$STEP" ]]; then
        [[ "$STEP" == "$step_num" ]]
    elif [[ -n "$PHASE" ]]; then
        [[ "$step_phase" -ge "$PHASE" ]]
    else
        return 0  # Run all
    fi
}

cd "$PROJECT_ROOT"

#===============================================================================
# PHASE 1: ENVIRONMENT SETUP
#===============================================================================

if should_run_step "1.1"; then
    log_step "1.1" "Activate Conda Environment"
    
    # Source conda
    if [[ -f ~/miniconda3/etc/profile.d/conda.sh ]]; then
        source ~/miniconda3/etc/profile.d/conda.sh
    elif [[ -f ~/anaconda3/etc/profile.d/conda.sh ]]; then
        source ~/anaconda3/etc/profile.d/conda.sh
    else
        log_error "Conda not found. Please install conda first."
        exit 1
    fi
    
    # Create environment if it doesn't exist
    if ! conda env list | grep -q "^${CONDA_ENV} "; then
        log_info "Creating conda environment: $CONDA_ENV"
        conda create -n "$CONDA_ENV" python=3.10 -y
    fi
    
    conda activate "$CONDA_ENV"
    log_success "Conda environment activated: $CONDA_ENV"
fi

# Always activate conda for subsequent steps
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

if should_run_step "1.2"; then
    log_step "1.2" "Install Python Dependencies"
    
    pip install torch torchaudio transformers captum librosa soundfile jiwer \
                scipy numpy pandas matplotlib seaborn tqdm datasets maturin torchcodec
    
    log_success "Python dependencies installed"
fi

if should_run_step "1.3"; then
    log_step "1.3" "Build TensorCommitment Library"
    
    cd "$PROJECT_ROOT/libs/TensorCommitment"
    
    if [[ -f install.sh ]]; then
        chmod +x install.sh
        ./install.sh
    else
        log_warning "install.sh not found, building manually..."
        
        # Build tensorCommitmentLib
        if [[ -d tensorCommitmentLib ]]; then
            cd tensorCommitmentLib
            maturin develop --release
            cd ..
        fi
        
        # Build terkleLib
        if [[ -d terkleLib ]]; then
            cd terkleLib
            maturin develop --release
            cd ..
        fi
        
        # Build CleanPegasus
        if [[ -d CleanPegasus ]]; then
            cd CleanPegasus
            maturin develop --release
            cd ..
        fi
        
        # Build merkle
        if [[ -d merkle ]]; then
            cd merkle
            maturin develop --release
            cd ..
        fi
    fi
    
    cd "$PROJECT_ROOT"
    log_success "TensorCommitment library built"
fi

if should_run_step "1.4"; then
    log_step "1.4" "Verify Environment Setup"
    
    python scripts/verify_setup.py
    
    log_success "Environment verification complete"
fi

if should_run_step "1.5"; then
    log_step "1.5" "Download HuggingFace Models"
    
    python scripts/download_models.py
    
    log_success "Models downloaded to models/"
fi

if should_run_step "1.6"; then
    log_step "1.6" "Download IEMOCAP Dataset"
    
    python scripts/download_iemocap.py
    
    log_success "IEMOCAP dataset downloaded to datasets/iemocap/"
fi

#===============================================================================
# PHASE 2: DATASET STRUCTURE EXTRACTION
#===============================================================================

if should_run_step "2.1"; then
    log_step "2.1" "Dataset Structure Documentation"
    
    log_info "Dataset structure documentation is in: datasets/DATASET_STRUCTURE.md"
    
    if [[ -f datasets/DATASET_STRUCTURE.md ]]; then
        log_success "Dataset structure documentation exists"
    else
        log_warning "datasets/DATASET_STRUCTURE.md not found"
    fi
fi

#===============================================================================
# PHASE 3: EMBEDDING EXTRACTION
#===============================================================================

if should_run_step "3.1"; then
    log_step "3.1" "Extract TIMIT Embeddings (Multi-GPU)"
    
    python scripts/extract_embeddings.py \
        --dataset timit \
        --output embeddings/ \
        --num-gpus $NUM_GPUS
    
    log_success "TIMIT embeddings extracted to embeddings/timit/ (~97 MB)"
fi

if should_run_step "3.2"; then
    log_step "3.2" "Extract IEMOCAP Embeddings (Multi-GPU)"
    
    python scripts/extract_embeddings.py \
        --dataset iemocap \
        --output embeddings/ \
        --num-gpus $NUM_GPUS
    
    log_success "IEMOCAP embeddings extracted to embeddings/iemocap/ (~8.6 GB)"
fi

#===============================================================================
# PHASE 4: TENSOR COMMITMENT BUILDING
#===============================================================================

if should_run_step "4.1"; then
    log_step "4.1" "Build TIMIT Commitments"
    
    python scripts/build_commitments.py \
        --dataset timit
    
    log_success "TIMIT commitments built (16 speakers, 32 bytes each)"
fi

if should_run_step "4.2"; then
    log_step "4.2" "Build IEMOCAP Commitments"
    
    python scripts/build_commitments.py \
        --dataset iemocap
    
    log_success "IEMOCAP commitments built (10 speakers, 32 bytes each)"
fi

#===============================================================================
# PHASE 5: TIMIT CAPTUM ANALYSIS (MULTI-GPU)
#===============================================================================

if should_run_step "5.1"; then
    log_step "5.1" "TIMIT: Integrated Gradients Analysis"
    
    python scripts/captum_analysis.py \
        --dataset timit \
        --head er \
        --analysis ig \
        --num-gpus "$NUM_GPUS"
    
    log_success "TIMIT IG analysis complete (~18s with 8 GPUs)"
fi

if should_run_step "5.2"; then
    log_step "5.2" "TIMIT: Shapley Value Analysis (Feature-level)"
    
    python scripts/captum_analysis.py \
        --dataset timit \
        --head er \
        --analysis shapley \
        --n-samples "$SHAPLEY_N_SAMPLES" \
        --num-gpus "$NUM_GPUS"
    
    log_success "TIMIT feature Shapley analysis complete (~2.4 min with 8 GPUs)"
fi

if should_run_step "5.2.5"; then
    log_step "5.2.5" "TIMIT: Data Shapley Analysis (Utterance-level)"
    
    python scripts/captum_analysis.py \
        --dataset timit \
        --head er \
        --analysis data_shapley \
        --n-permutations 100 \
        --num-gpus "$NUM_GPUS"
    
    log_success "TIMIT Data Shapley analysis complete (~18s with 8 GPUs)"
fi

if should_run_step "5.3"; then
    log_step "5.3" "TIMIT: TracIn Checkpoints"
    
    python scripts/captum_analysis.py \
        --dataset timit \
        --head er \
        --analysis tracin \
        --finetune-epochs "$TRACIN_EPOCHS"
    
    log_success "TIMIT TracIn checkpoints generated"
fi

if should_run_step "5.4"; then
    log_step "5.4" "TIMIT: Visualizations"
    
    python scripts/visualize_captum.py \
        --analysis-dir analysis/captum/er
    
    log_success "TIMIT visualizations saved"
fi

#===============================================================================
# PHASE 6: IEMOCAP CAPTUM ANALYSIS (MULTI-GPU)
#===============================================================================

if should_run_step "6.1"; then
    log_step "6.1" "IEMOCAP: Integrated Gradients Analysis"
    
    python scripts/captum_analysis.py \
        --dataset iemocap \
        --head er \
        --analysis ig \
        --num-gpus "$NUM_GPUS"
    
    log_success "IEMOCAP IG analysis complete (~32s with 8 GPUs)"
fi

if should_run_step "6.2"; then
    log_step "6.2" "IEMOCAP: Shapley Value Analysis (Feature-level)"
    
    log_warning "This step takes ~3.5 hours with 8 GPUs (10,039 samples)"
    
    python scripts/captum_analysis.py \
        --dataset iemocap \
        --head er \
        --analysis shapley \
        --n-samples "$SHAPLEY_N_SAMPLES" \
        --num-gpus "$NUM_GPUS"
    
    log_success "IEMOCAP feature Shapley analysis complete"
fi

if should_run_step "6.2.5"; then
    log_step "6.2.5" "IEMOCAP: Data Shapley Analysis (Utterance-level)"
    
    python scripts/captum_analysis.py \
        --dataset iemocap \
        --head er \
        --analysis data_shapley \
        --n-permutations 100 \
        --num-gpus "$NUM_GPUS"
    
    log_success "IEMOCAP Data Shapley analysis complete (~2 min with 8 GPUs)"
fi

if should_run_step "6.2.6"; then
    log_step "6.2.6" "Data Shapley (SID+ER full emotions) for TIMIT + IEMOCAP with tqdm ETA"

    python - <<PY
import subprocess
import sys
import time
from pathlib import Path
from tqdm import tqdm

project_root = Path(r"${PROJECT_ROOT}")
num_gpus = "${NUM_GPUS}"
pydvl_jobs = "${SID_PYDVL_JOBS}"

runs = [
    (
        "timit",
        [
            "python", "scripts/captum_analysis.py",
            "--dataset", "timit",
            "--head", "sid",
            "--analysis", "data_shapley",
            "--num-gpus", num_gpus,
            "--pydvl-model", "finetune_head",
            "--pydvl-feature-head", "sid",
            "--pydvl-train-device", "cuda",
            "--pydvl-jobs", pydvl_jobs,
        ],
    ),
    (
        "iemocap",
        [
            "python", "scripts/captum_analysis.py",
            "--dataset", "iemocap",
            "--head", "er",
            "--analysis", "data_shapley",
            "--num-gpus", num_gpus,
            "--pydvl-model", "finetune_head",
            "--pydvl-feature-head", "er",
            "--pydvl-train-device", "cuda",
            "--pydvl-jobs", pydvl_jobs,
            "--pydvl-task", "emotion",
        ],
    ),
]

overall_start = time.time()
for dataset_name, cmd in tqdm(runs, desc="Data Shapley (both datasets)", unit="dataset"):
    start = time.time()
    tqdm.write(f"Starting {dataset_name}...")
    return_code = subprocess.call(cmd, cwd=str(project_root))
    if return_code != 0:
        sys.exit(return_code)
    elapsed = time.time() - start
    tqdm.write(f"Finished {dataset_name} in {elapsed / 60.0:.2f} min")

total_elapsed = time.time() - overall_start
print(f"Total elapsed: {total_elapsed / 60.0:.2f} min")
PY

    log_success "Data Shapley complete for TIMIT(SID) + IEMOCAP(ER full emotions)"
fi

if should_run_step "6.3"; then
    log_step "6.3" "IEMOCAP: TracIn Checkpoints"
    
    python scripts/captum_analysis.py \
        --dataset iemocap \
        --head er \
        --analysis tracin \
        --finetune-epochs "$TRACIN_EPOCHS"
    
    log_success "IEMOCAP TracIn checkpoints generated"
fi

if should_run_step "6.4"; then
    log_step "6.4" "Update Visualizations (Both Datasets)"
    
    python scripts/visualize_captum.py \
        --analysis-dir analysis/captum/er
    
    log_success "Visualizations updated with IEMOCAP data"
fi

#===============================================================================
# OPTIONAL: ADDITIONAL PREDICTION HEADS
#===============================================================================

if should_run_step "7.1"; then
    log_step "7.1" "Additional Heads (Optional)"
    
    log_warning "Additional head analysis not run by default"
    log_info "Available heads: sid (Speaker ID), ks (Keyword Spotting), ic (Intent Classification)"
    log_info ""
    log_info "To analyze other heads:"
    log_info "  python scripts/captum_analysis.py --dataset timit --head sid --analysis all --num-gpus $NUM_GPUS"
    log_info "  python scripts/captum_analysis.py --dataset iemocap --head sid --analysis all --num-gpus $NUM_GPUS"
fi

#===============================================================================
# SUMMARY
#===============================================================================

echo ""
echo "============================================================"
echo -e "${GREEN}FULL PROCESS COMPLETE${NC}"
echo "============================================================"
echo ""
echo "Output locations:"
echo "  - Embeddings:"
echo "      embeddings/timit/    (16 speakers, 97 MB)"
echo "      embeddings/iemocap/  (10 speakers, 8.6 GB)"
echo ""
echo "  - Commitments:"
echo "      commitments/timit/   (16 speakers, 32 bytes each)"
echo "      commitments/iemocap/ (10 speakers, 32 bytes each)"
echo ""
echo "  - Analysis:"
echo "      analysis/captum/er/shapley/"
echo "      analysis/captum/er/integrated_gradients/"
echo "      analysis/captum/er/tracin/"
echo "      analysis/captum/er/visualizations/"
echo ""
echo "Documentation:"
echo "  - progress.md          - Implementation progress"
echo "  - tests.md             - Test documentation"
echo "  - datasets/DATASET_STRUCTURE.md"
echo ""
echo "Total runtime estimate:"
echo "  - Without IEMOCAP Shapley: ~30 minutes"
echo "  - With IEMOCAP Shapley:    ~4 hours"
echo ""
