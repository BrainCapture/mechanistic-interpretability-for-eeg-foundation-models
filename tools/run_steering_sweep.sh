#!/usr/bin/env bash
# Run the steering experiment across an SAE expansion sweep.
#
# Picks all experiment dirs under results/experiments/ matching $PATTERN,
# symlinks the shared SleepFM-finetuned steering cache (encoder is identical
# across experiments), runs plot_concept_steering_xae_all.py with the
# requested fractions, and renames the resulting steering_metrics.json into
# paper/concept_steering_figures/steering_metrics_{exp}.json.
#
# Defaults to the kscaled sweep (k/n=6.25%):
#   bash tools/run_steering_sweep.sh
# Override the pattern, concept, fractions, or random-baseline draws:
#   bash tools/run_steering_sweep.sh \
#       --pattern 'sleepfm_finetuned_exp*_kscaled_layer2' \
#       --concept age \
#       --fracs 0.25 0.5 1.0 \
#       --rand 100
# Always includes the E=1 baseline experiment (sleepfm_finetuned_layer2).

set -euo pipefail
cd "$(dirname "$0")/.."

PATTERN='sleepfm_finetuned_exp*_kscaled_layer2'
CONCEPT='age'
FRACS=(0.25 0.5 1.0)
RAND=100
BASELINE_EXP='sleepfm_finetuned_layer2'   # E=1; auto-prepended to MATCHES; suppress with --no-baseline
SHARED_FROM=''                            # source experiment for the shared steering cache; defaults to BASELINE_EXP
OUT_BASE='paper/concept_steering_figures'

EXPLICIT=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pattern)       PATTERN="$2"; shift 2 ;;
        --experiments)   shift; EXPLICIT=(); while [[ $# -gt 0 && "$1" != --* ]]; do EXPLICIT+=("$1"); shift; done ;;
        --concept)       CONCEPT="$2"; shift 2 ;;
        --fracs)         shift; FRACS=(); while [[ $# -gt 0 && "$1" != --* ]]; do FRACS+=("$1"); shift; done ;;
        --rand)          RAND="$2"; shift 2 ;;
        --baseline-exp)  BASELINE_EXP="$2"; shift 2 ;;
        --shared-from)   SHARED_FROM="$2"; shift 2 ;;
        --no-baseline)   BASELINE_EXP=""; shift ;;
        *) echo "Unknown flag: $1"; exit 2 ;;
    esac
done

# Resolve shared steering cache source: explicit --shared-from beats baseline.
if [[ -z "$SHARED_FROM" ]]; then
    SHARED_FROM="$BASELINE_EXP"
fi
if [[ -z "$SHARED_FROM" ]]; then
    echo "ERROR: cannot resolve shared steering cache; pass --shared-from <exp> when using --no-baseline" >&2
    exit 2
fi
SHARED_CACHE="$(pwd)/results/steering_cache/${SHARED_FROM}/steering_cache.pt"

if [[ ! -f "$SHARED_CACHE" ]]; then
    echo "ERROR: shared steering cache missing at $SHARED_CACHE" >&2
    echo "Build it first:  uv run tools/build_steering_cache.py --experiment $SHARED_FROM" >&2
    exit 1
fi

if [[ ${#EXPLICIT[@]} -gt 0 ]]; then
    MATCHES=("${EXPLICIT[@]}")
else
    mapfile -t MATCHES < <(cd results/experiments && ls -d $PATTERN 2>/dev/null | sort)
fi

if [[ -n "$BASELINE_EXP" ]]; then
    HAS_BASELINE=0
    for e in "${MATCHES[@]}"; do
        [[ "$e" == "$BASELINE_EXP" ]] && HAS_BASELINE=1
    done
    if [[ "$HAS_BASELINE" -eq 0 ]]; then
        MATCHES=("$BASELINE_EXP" "${MATCHES[@]}")
    fi
fi

if [[ ${#MATCHES[@]} -eq 0 ]]; then
    echo "No experiment dirs matched pattern: $PATTERN"
    exit 1
fi

echo "=== Sweep config ==="
echo "  pattern : $PATTERN"
echo "  concept : $CONCEPT"
echo "  fracs   : ${FRACS[*]}"
echo "  rand    : $RAND"
echo "  matches : ${MATCHES[*]}"
echo

for EXP in "${MATCHES[@]}"; do
    echo ">>> [$EXP]"

    EXP_DIR="results/experiments/$EXP"
    if [[ ! -f "$EXP_DIR/app_cache.pt" ]]; then
        echo "    [skip] $EXP_DIR/app_cache.pt missing — build it first"
        continue
    fi

    SC_DIR="results/steering_cache/$EXP"
    SC_LINK="$SC_DIR/steering_cache.pt"
    if [[ ! -f "$SC_LINK" && ! -L "$SC_LINK" ]]; then
        mkdir -p "$SC_DIR"
        ln -sf "$SHARED_CACHE" "$SC_LINK"
        echo "    symlinked steering cache: $SC_LINK -> $SHARED_CACHE"
    fi

    OUT_SUB="$OUT_BASE/sweep/$EXP"
    mkdir -p "$OUT_SUB"
    echo "    running plot_concept_steering_xae_all.py …"
    uv run tools/plot_concept_steering_xae_all.py \
        --experiment "$EXP" \
        --concepts "$CONCEPT" \
        --steps-frac "${FRACS[@]}" \
        --random-baseline "$RAND" \
        --out-dir "$OUT_SUB" 2>&1 | tail -30

    SRC="$OUT_SUB/steering_metrics.json"
    DST="$OUT_BASE/steering_metrics_${EXP}.json"
    if [[ -f "$SRC" ]]; then
        cp "$SRC" "$DST"
        echo "    saved → $DST"
    else
        echo "    [warn] $SRC not produced"
    fi
    echo
done

echo "=== Sweep done ==="
echo "Per-experiment metrics in $OUT_BASE/steering_metrics_<exp>.json"
echo "Now run:  uv run tools/plot_steering_vs_expansion.py --variant kscaled"
