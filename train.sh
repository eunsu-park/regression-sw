#!/bin/bash
# Parallel training for all experiment configs.
#
# Runs up to MAX_JOBS processes concurrently.
# When one finishes, the next config in the queue starts automatically.
#
# Usage (config groups — io × model cross product):
#   ./train.sh                                  # Run all io × model combos
#   ./train.sh --filter out12h                  # Only io configs matching "out12h"
#   ./train.sh --model transformer              # Only transformer model
#   ./train.sh --filter in2d --model gnn_tcn    # Specific io + model
#   ./train.sh --filter "in(6|12)h_out(6|12)h"  # Regex pattern supported
#   ./train.sh --max-jobs 4                     # Limit to 4 parallel jobs
#   ./train.sh --dry-run                        # Print configs without running
#   ./train.sh --config-name dev                # Use configs/dev.yaml (default: local)
#   ./train.sh --skip-existing                  # Skip completed runs (config-group
#                                               # mode only; a run counts as completed
#                                               # when {save_root}/{exp}/log/
#                                               # training_history.json exists)
#
# Usage (file-based):
#   ./train.sh --config-file list.txt           # Run configs from file

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Prevent non-matching globs from returning literal patterns
shopt -s nullglob

# =============================================================================
# Arguments
# =============================================================================
MAX_JOBS=8
CONFIG_FILE=""
FILTER=""
MODEL_FILTER=""
DRY_RUN=false
CONFIG_NAME="local"
SKIP_EXISTING=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --max-jobs)
            MAX_JOBS="$2"
            shift 2
            ;;
        --config-file)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --filter)
            FILTER="$2"
            shift 2
            ;;
        --model)
            MODEL_FILTER="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-existing)
            SKIP_EXISTING=true
            shift
            ;;
        --config-name)
            CONFIG_NAME="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./train.sh [--config-file FILE] [--max-jobs N] [--filter PATTERN] [--model MODEL] [--dry-run] [--config-name NAME] [--skip-existing]"
            exit 1
            ;;
    esac
done

# =============================================================================
# Experiment-name prefix (prevents ap/hp result collisions)
# Derived from the config profile: server -> ap_, server_hp -> hp_.
# Other profiles (local/dev/...) keep the legacy un-prefixed names.
# =============================================================================
case "$CONFIG_NAME" in
    server_ap|mac_ap)                     EXP_PREFIX="ap_" ;;
    server_ap_storm|mac_ap_storm)         EXP_PREFIX="ap_storm_" ;;
    server_ap_quiet|mac_ap_quiet)         EXP_PREFIX="ap_quiet_" ;;
    server_ap_recursive|mac_ap_recursive) EXP_PREFIX="ap_recursive_" ;;
    server_hp|mac_hp)                     EXP_PREFIX="hp_" ;;
    *)                                    EXP_PREFIX="" ;;
esac

# All 2026-08 ap sweeps (direct / storm / quiet / recursive, server or Mac)
# share the short-horizon io grid: input {6h,12h,18h,1d} x output {1h..6h}.
case "$CONFIG_NAME" in
    server_ap|mac_ap|server_ap_storm|mac_ap_storm|server_ap_quiet|mac_ap_quiet|server_ap_recursive|mac_ap_recursive)
        SHORT_GRID=true ;;
    *)  SHORT_GRID=false ;;
esac

# Default io filter for short-grid profiles (explicit --filter overrides).
if $SHORT_GRID && [[ -z "$FILTER" ]]; then
    FILTER="in(6h|12h|18h|1d)_out[1-6]h$"
fi

# =============================================================================
# Completed-run detection (--skip-existing)
# A run counts as completed when {save_root}/{exp}/log/training_history.json
# exists — scripts/train.py writes it only after fit() returns normally, so
# interrupted runs are re-queued. save_root is resolved by walking the config
# inheritance chain (defaults:), or taken from the SAVE_ROOT env var.
# =============================================================================
SKIPPED=0
if $SKIP_EXISTING; then
    if [[ -z "${SAVE_ROOT:-}" ]]; then
        cfg="$CONFIG_NAME"
        for _ in 1 2 3 4 5; do
            cfg_path="configs/${cfg}.yaml"
            [[ -f "$cfg_path" ]] || break
            SAVE_ROOT=$(grep -E "^[[:space:]]*save_root:" "$cfg_path" | head -1 \
                | sed -E 's/.*save_root:[[:space:]]*"?([^"]*)"?.*/\1/')
            [[ -n "$SAVE_ROOT" ]] && break
            cfg=$(awk '/^defaults:/{f=1;next}
                       f && /^[^ -]/{exit}
                       f && /^[[:space:]]*-[[:space:]]*[a-zA-Z]/{
                           sub(/^[[:space:]]*-[[:space:]]*/,"");
                           if ($1 != "override" && $1 != "_self_") {print $1; exit}
                       }' "$cfg_path")
            [[ -z "$cfg" ]] && break
        done
    fi
    if [[ -z "${SAVE_ROOT:-}" || ! -d "$SAVE_ROOT" ]]; then
        echo "ERROR: --skip-existing needs a valid save_root for '$CONFIG_NAME'" \
             "(resolved: '${SAVE_ROOT:-<none>}', not a directory)."
        echo "       Set the SAVE_ROOT env var explicitly."
        exit 1
    fi
fi

# hp uses SW + hp30 inputs, so the inherited ap30 GNN node must be dropped
# (Hydra deep-merges dicts, so it cannot be removed from within server_hp.yaml).
EXTRA_ARGS=()
if [[ "$CONFIG_NAME" == "server_hp" || "$CONFIG_NAME" == "mac_hp" ]]; then
    EXTRA_ARGS+=("~data.timeseries.gnn_variable_groups.ap30")
fi

# =============================================================================
# Collect configs
# =============================================================================
CONFIGS=()
DISPLAY_NAMES=()

if [[ -n "$CONFIG_FILE" ]]; then
    # Read from file (skip empty lines and comments)
    while IFS= read -r line; do
        line=$(echo "$line" | sed 's/#.*//' | xargs)
        [[ -z "$line" ]] && continue
        CONFIGS+=("$line")
        DISPLAY_NAMES+=("$line")
    done < "$CONFIG_FILE"
else
    # New config group mode: io × model cross product
    IO_CONFIGS=()
    for f in configs/io/*.yaml; do
        io_name=$(basename "$f" .yaml)
        if [[ -n "$FILTER" && ! "$io_name" =~ $FILTER ]]; then
            continue
        fi
        IO_CONFIGS+=("$io_name")
    done
    IO_CONFIGS=($(printf '%s\n' "${IO_CONFIGS[@]}" | sort))

    MODEL_CONFIGS=()
    for f in configs/model/*.yaml; do
        model_name=$(basename "$f" .yaml)
        if [[ -n "$MODEL_FILTER" && "$model_name" != "$MODEL_FILTER" ]]; then
            continue
        fi
        MODEL_CONFIGS+=("$model_name")
    done
    MODEL_CONFIGS=($(printf '%s\n' "${MODEL_CONFIGS[@]}" | sort))

    for io in "${IO_CONFIGS[@]}"; do
        for mdl in "${MODEL_CONFIGS[@]}"; do
            exp_name="${EXP_PREFIX}${io}_${mdl}"
            if $SKIP_EXISTING && [[ -f "$SAVE_ROOT/${exp_name}/log/training_history.json" ]]; then
                SKIPPED=$((SKIPPED + 1))
                continue
            fi
            CONFIGS+=("+io=${io} +model=${mdl} experiment.name=${exp_name}")
            DISPLAY_NAMES+=("${exp_name}")
        done
    done
fi

TOTAL=${#CONFIGS[@]}
if [[ $TOTAL -eq 0 ]]; then
    if [[ $SKIPPED -gt 0 ]]; then
        echo "All $SKIPPED matching runs are already completed. Nothing to do."
        exit 0
    fi
    echo "No configs found (config-file: '$CONFIG_FILE', filter: '$FILTER', model: '$MODEL_FILTER')"
    exit 1
fi

echo "========================================"
echo "Parallel Training Runner"
echo "========================================"
echo "Total configs: $TOTAL"
echo "Max parallel:  $MAX_JOBS"
if [[ -n "$CONFIG_FILE" ]]; then
    echo "Source:        $CONFIG_FILE"
else
    echo "Mode:          config groups (io × model)"
fi
echo "Filter:        ${FILTER:-none}"
echo "Model:         ${MODEL_FILTER:-all}"
echo "Config name:   $CONFIG_NAME"
echo "Exp prefix:    ${EXP_PREFIX:-<none>}"
if $SKIP_EXISTING; then
    echo "Skipped done:  $SKIPPED (save_root: $SAVE_ROOT)"
fi
echo "========================================"
echo ""

if $DRY_RUN; then
    echo "[DRY RUN] Configs to run:"
    for name in "${DISPLAY_NAMES[@]}"; do
        echo "  $name"
    done
    echo ""
    echo "Total: $TOTAL configs"
    exit 0
fi

# =============================================================================
# Parallel execution
# =============================================================================
LOG_DIR="$HOME/tmp/train_logs"
mkdir -p "$LOG_DIR"

RUNNING_PIDS=()
RUNNING_NAMES=()
COMPLETED=0
FAILED=0
STARTED=0

wait_for_slot() {
    while [[ ${#RUNNING_PIDS[@]} -ge $MAX_JOBS ]]; do
        NEW_PIDS=()
        NEW_NAMES=()
        for i in "${!RUNNING_PIDS[@]}"; do
            pid=${RUNNING_PIDS[$i]}
            name=${RUNNING_NAMES[$i]}
            if kill -0 "$pid" 2>/dev/null; then
                NEW_PIDS+=("$pid")
                NEW_NAMES+=("$name")
            else
                wait "$pid" && code=0 || code=$?
                COMPLETED=$((COMPLETED + 1))
                if [[ $code -eq 0 ]]; then
                    echo "[DONE]  $name  ($COMPLETED/$TOTAL completed)"
                else
                    echo "[FAIL]  $name  (exit $code) — see $LOG_DIR/${name}.log"
                    FAILED=$((FAILED + 1))
                fi
            fi
        done
        RUNNING_PIDS=("${NEW_PIDS[@]}")
        RUNNING_NAMES=("${NEW_NAMES[@]}")

        if [[ ${#RUNNING_PIDS[@]} -ge $MAX_JOBS ]]; then
            sleep 5
        fi
    done
}

for idx in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$idx]}"
    display_name="${DISPLAY_NAMES[$idx]}"

    wait_for_slot

    STARTED=$((STARTED + 1))
    echo "[START] $display_name  ($STARTED/$TOTAL, running: ${#RUNNING_PIDS[@]}+1)"

    if [[ -n "$CONFIG_FILE" ]]; then
        # File mode: config name passed as --config-name
        python scripts/train.py --config-name="$cfg" \
            > "$LOG_DIR/${display_name}.log" 2>&1 &
    else
        # Config group mode: overrides passed as positional args
        # shellcheck disable=SC2086
        python scripts/train.py --config-name="$CONFIG_NAME" $cfg "${EXTRA_ARGS[@]}" \
            > "$LOG_DIR/${display_name}.log" 2>&1 &
    fi

    RUNNING_PIDS+=($!)
    RUNNING_NAMES+=("$display_name")
done

for i in "${!RUNNING_PIDS[@]}"; do
    pid=${RUNNING_PIDS[$i]}
    name=${RUNNING_NAMES[$i]}
    wait "$pid" && code=0 || code=$?
    COMPLETED=$((COMPLETED + 1))
    if [[ $code -eq 0 ]]; then
        echo "[DONE]  $name  ($COMPLETED/$TOTAL completed)"
    else
        echo "[FAIL]  $name  (exit $code) — see $LOG_DIR/${name}.log"
        FAILED=$((FAILED + 1))
    fi
done

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "========================================"
echo "Training Complete"
echo "========================================"
echo "Total:     $TOTAL"
echo "Succeeded: $((COMPLETED - FAILED))"
echo "Failed:    $FAILED"
echo "Logs:      $LOG_DIR/"
echo "========================================"

if [[ $FAILED -gt 0 ]]; then
    echo ""
    echo "Failed configs:"
    grep -l "Error\|Exception\|Traceback" "$LOG_DIR"/*.log 2>/dev/null | while read f; do
        echo "  $(basename "$f" .log)"
    done
    exit 1
fi
