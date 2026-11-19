#!/usr/bin/env bash
# Matched-budget sweep for the Figure 7 subset.
#
# Each (target, algorithm) pair gets the same wall-clock budget and repeats its
# search until that budget is spent, which is how the paper normalises compute.
# Each repeat is one point on the scaling curve.
#
# The pipeline config pins seed: 5, so every repeat must override it. Without
# that, repeats regenerate byte-identical structures and the unique-success
# curve goes flat while the raw count climbs.
#
# usage: run_figure7.sh <gpu-uuid> <budget-seconds> <target> [target...]

set -uo pipefail

GPU_UUID="${1:?usage: run_figure7.sh <gpu-uuid> <budget-seconds> <target>...}"
BUDGET="${2:?missing budget in seconds}"
shift 2
TARGETS=("$@")

REPO="$HOME/Proteina-Complexa"
ALGOS=(${ALGOS_OVERRIDE:-best-of-n beam-search fk-steering})
LEDGER="$REPO/${LEDGER_NAME:-fig7_ledger.csv}"

cd "$REPO" || exit 1
export PATH="$HOME/miniconda3/envs/complexa/bin:$PATH"
set +u; source env.sh >/dev/null 2>&1; set -u
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_UUID"

# JAX preallocates about 75% of the device by default, which starves PyTorch.
# The generator and the AF2 reward share one card here.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

[ -f "$LEDGER" ] || echo "target,algorithm,repeat,seed,elapsed_s,cumulative_s,run_name,gpu_uuid" > "$LEDGER"

SEED_BASE="${SEED_BASE:-$((RANDOM % 1000 * 100000))}"
BON_N0="${BON_N0:-50}"
seed_n=0

for TARGET in "${TARGETS[@]}"; do
    for ALGO in "${ALGOS[@]}"; do
        cum=0
        rep=0
        nsamples="$BON_N0"
        while [ "$cum" -lt "$BUDGET" ]; do
            rep=$((rep + 1))
            seed_n=$((seed_n + 1))
            SEED=$((SEED_BASE + seed_n))
            RUN="f7_${TARGET}_${ALGO//-/_}_s${SEED}"
            # Best-of-N is scaled by raising N within a run (Appendix H.1); the
            # branching methods are scaled by repeating the whole search (H.2).
            if [ "$ALGO" = best-of-n ]; then
                NS_OVERRIDE="++generation.dataloader.dataset.nres.nsamples=$nsamples"
            else
                NS_OVERRIDE=""
            fi
            start=$(date +%s)
            complexa generate configs/search_binder_local_pipeline.yaml \
                ++run_name="$RUN" \
                ++generation.task_name="$TARGET" \
                ++generation.search.algorithm="$ALGO" \
                ++seed="$SEED" \
                $NS_OVERRIDE \
                >/dev/null 2>&1
            rc=$?
            end=$(date +%s)
            elapsed=$((end - start))
            cum=$((cum + elapsed))
            echo "$TARGET,$ALGO,$rep,$SEED,$elapsed,$cum,$RUN,$GPU_UUID" >> "$LEDGER"
            printf '[%s] %s %s rep%-3d rc=%d %5ds cum=%6ds/%ds\n' \
                "$(date -u +%H:%M:%S)" "$TARGET" "$ALGO" "$rep" "$rc" "$elapsed" "$cum" "$BUDGET"
            if [ "$rc" -ne 0 ]; then
                echo "  non-zero exit, stopping this pair"
                break
            fi
            # Do not start a repeat that would clearly overrun the budget.
            if [ $((cum + elapsed)) -gt $((BUDGET + BUDGET / 4)) ]; then
                echo "  next repeat would overrun, stopping this pair"
                break
            fi
            [ "$ALGO" = best-of-n ] && nsamples=$((nsamples * 2))
        done
    done
done

echo "SWEEP DONE on $GPU_UUID"
