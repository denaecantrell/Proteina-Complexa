#!/usr/bin/env bash
# Table 2: base generation without optimization, 200 samples per target.
#
# generate -> evaluate -> analyze, one target at a time on one pinned GPU.
#
# usage: run_table2.sh <gpu-uuid> <nsamples> <target> [target...]

set -uo pipefail

GPU_UUID="${1:?usage: run_table2.sh <gpu-uuid> <nsamples> <target>...}"
NSAMPLES="${2:?missing nsamples}"
shift 2
TARGETS=("$@")

REPO="$HOME/Proteina-Complexa"
cd "$REPO" || exit 1
export PATH="$HOME/miniconda3/envs/complexa/bin:$PATH"
set +u; source env.sh >/dev/null 2>&1; set -u
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_UUID"

# JAX grabs ~75% of the card by default and starves PyTorch; 04_IFNAR2 OOMs at
# batch 8 of 13 without this, with PyTorch holding 7.6 GB of 47.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for T in "${TARGETS[@]}"; do
    RUN="table2_$T"
    DIR="./inference/search_binder_local_pipeline_${T}_${RUN}"

    # One run of N samples, not N/4 repeats: the config pins seed: 5, so repeats
    # regenerate identical structures. Samples within a single run do vary.
    echo "=== $(date -u +%H:%M:%S) $T generate ==="
    complexa generate configs/search_binder_local_pipeline.yaml \
        ++run_name="$RUN" \
        ++generation.task_name="$T" \
        ++generation.search.algorithm=single-pass \
        ++generation.dataloader.dataset.nres.nsamples="$NSAMPLES" \
        || { echo "GENERATE FAILED $T"; continue; }

    # Refolds each design; this is where scRMSD comes from.
    echo "=== $(date -u +%H:%M:%S) $T evaluate ==="
    complexa evaluate configs/evaluate.yaml \
        ++run_name="$RUN" \
        ++dataset.task_name="$T" \
        ++sample_storage_path="$DIR" \
        ++eval_njobs=1 \
        || { echo "EVALUATE FAILED $T"; continue; }

    # base_config_name=evaluate is required: analyze.py:509 builds its input
    # filename as binder_results_{config_name}_ and the CLI injects "analyze",
    # so it looks for a file evaluate never wrote.
    echo "=== $(date -u +%H:%M:%S) $T analyze ==="
    complexa analyze configs/analyze.yaml \
        ++run_name="$RUN" \
        ++results_dir="./evaluation_results/binder_eval_$RUN" \
        ++base_config_name=evaluate \
        || echo "ANALYZE FAILED $T"

    echo "=== $(date -u +%H:%M:%S) $T done ==="
done

echo "TABLE 2 SWEEP DONE on $GPU_UUID"
