#!/bin/bash
# Run pending validation / attention jobs in parallel.
#
# Scans save_root for missing outputs across the io x model matrix and
# executes only the combinations whose expected artifact is absent.
# Safe to re-run — completed experiments are skipped automatically.
#
# MC-dropout uncertainty is folded into the validation pass (no separate mcd phase);
# the per-event npz under validation/ carries the predictive interval.
#
# Completion markers (per phase, under {save_root}/{experiment}/):
#   validation/{epoch}/validation_results.csv
#   attention/{epoch}/npz.zip   (only for transformer, patchtst,
#                                gnn_transformer, gnn_patchtst)
#
# Usage:
#   ./run_pending.sh                              # validation + attention
#   ./run_pending.sh --phases attention           # subset (comma-separated)
#   ./run_pending.sh --phases validation          # single phase
#   ./run_pending.sh --dry-run                    # print tasks, do not execute
#   ./run_pending.sh --max-jobs 4 --epoch best
#   ./run_pending.sh --config-name dev            # use configs/dev.yaml (default: local)
#   SAVE_ROOT=/custom/path ./run_pending.sh       # override results root

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# =============================================================================
# Arguments
# =============================================================================
MAX_JOBS=8
EPOCH="best"
DRY_RUN=false
PHASES="validation,attention"
CONFIG_NAME="local"

while [[ $# -gt 0 ]]; do
    case $1 in
        --max-jobs)    MAX_JOBS="$2"; shift 2 ;;
        --epoch)       EPOCH="$2"; shift 2 ;;
        --phases)      PHASES="$2"; shift 2 ;;
        --dry-run)     DRY_RUN=true; shift ;;
        --config-name) CONFIG_NAME="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Validate requested phases
IFS=',' read -r -a PHASE_LIST <<< "$PHASES"
for p in "${PHASE_LIST[@]}"; do
    case $p in
        validation|attention) ;;
        *) echo "Invalid phase: $p (allowed: validation, attention)"; exit 1 ;;
    esac
done

# =============================================================================
# Resolve save_root
# =============================================================================
# Profiles like server_ap_storm / server_ap_recursive / server_hp inherit
# save_root from a parent config, so walk the defaults chain (same logic
# as train.sh --skip-existing) instead of grepping one file.
if [[ -z "${SAVE_ROOT:-}" ]]; then
    if [[ ! -f "configs/${CONFIG_NAME}.yaml" ]]; then
        echo "ERROR: Config file not found: configs/${CONFIG_NAME}.yaml"
        exit 1
    fi
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
if [[ -z "$SAVE_ROOT" || ! -d "$SAVE_ROOT" ]]; then
    echo "ERROR: SAVE_ROOT not found or invalid: '$SAVE_ROOT'"
    echo "Set SAVE_ROOT env var or fix the config chain for '$CONFIG_NAME'."
    exit 1
fi

# =============================================================================
# Experiment-name prefix + hp GNN-node fix (server profiles)
#   server_ap -> "ap_" prefix ; server_hp -> "hp_" prefix + drop the inherited
#   ap30 GNN node. The prefix also scopes the save_root scan below so ap and hp
#   results (ap_*/hp_*) are detected independently. Other profiles: no prefix.
#   save_root is resolved through the config defaults chain, so profiles
#   that inherit it (server_hp, server_ap_storm, server_ap_recursive) work
#   without the SAVE_ROOT env var; the env var still overrides if set.
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
EXTRA_ARGS=()
if [[ "$CONFIG_NAME" == "server_hp" || "$CONFIG_NAME" == "mac_hp" ]]; then
    EXTRA_ARGS+=("~data.timeseries.gnn_variable_groups.ap30")
fi

# =============================================================================
# Matrix definition
# =============================================================================
IO_CONFIGS=(
    in6h_out6h  in6h_out12h  in6h_out18h  in6h_out24h
    in12h_out6h in12h_out12h in12h_out18h in12h_out24h
    in18h_out6h in18h_out12h in18h_out18h in18h_out24h
    in1d_out6h  in1d_out12h  in1d_out18h  in1d_out24h
    in2d_out6h  in2d_out12h  in2d_out18h  in2d_out24h
    in3d_out6h  in3d_out12h  in3d_out18h  in3d_out24h
)

# Both 2026-08 ap sweeps (direct and recursive) use the short-horizon grid:
# input {6h,12h,18h,1d} x output {1h..6h}. The legacy grid above stays for
# server_hp (hp results exist on it) and the local/dev profiles.
if $SHORT_GRID; then
    IO_CONFIGS=()
    for _in in in6h in12h in18h in1d; do
        for _out in out1h out2h out3h out4h out5h out6h; do
            IO_CONFIGS+=("${_in}_${_out}")
        done
    done
fi

ALL_MODELS=(linear transformer tcn patchtst timesnet lstm bilstm
            gnn_transformer gnn_tcn gnn_bilstm gnn_patchtst
            gnn_lstm gnn_timesnet gnn_linear)
ATTN_MODELS=(transformer patchtst gnn_transformer gnn_patchtst)

# Phase -> completion marker (relative to {save_root}/{experiment}/)
marker_for_phase() {
    case $1 in
        validation) echo "validation/${EPOCH}/validation_results.csv" ;;
        attention)  echo "attention/${EPOCH}/npz.zip" ;;
    esac
}

# Phase -> python entry script
runner_for_phase() {
    case $1 in
        validation) echo "scripts/validate.py" ;;
        attention)  echo "analysis/run_attention.py" ;;
    esac
}

# Phase -> applicable models
models_for_phase() {
    case $1 in
        attention) echo "${ATTN_MODELS[@]}" ;;
        *)         echo "${ALL_MODELS[@]}" ;;
    esac
}

# =============================================================================
# Detect pending tasks
# =============================================================================
TASKS=()
NOT_TRAINED=()

for phase in "${PHASE_LIST[@]}"; do
    marker=$(marker_for_phase "$phase")
    read -r -a models <<< "$(models_for_phase "$phase")"
    for io in "${IO_CONFIGS[@]}"; do
        for m in "${models[@]}"; do
            if [[ ! -f "$SAVE_ROOT/${EXP_PREFIX}${io}_${m}/${marker}" ]]; then
                # Guard: never analyze a checkpoint whose training did not
                # finish (log/training_history.json is written only on normal
                # completion). Interrupted trainings otherwise get silently
                # validated on a partial best-checkpoint.
                if [[ ! -f "$SAVE_ROOT/${EXP_PREFIX}${io}_${m}/log/training_history.json" ]]; then
                    NOT_TRAINED+=("${phase}:${EXP_PREFIX}${io}_${m}")
                    continue
                fi
                TASKS+=("${phase}:${io}:${m}")
            fi
        done
    done
done

TOTAL=${#TASKS[@]}

if [[ ${#NOT_TRAINED[@]} -gt 0 ]]; then
    echo "WARNING: skipping ${#NOT_TRAINED[@]} pending task(s) whose training is"
    echo "         incomplete (no log/training_history.json) — train them first:"
    for t in "${NOT_TRAINED[@]}"; do
        echo "  $t"
    done
    echo ""
fi

echo "========================================"
echo "Pending Analysis Runner"
echo "========================================"
echo "Save root:     $SAVE_ROOT"
echo "Config name:   $CONFIG_NAME"
echo "Phases:        $PHASES"
echo "Epoch:         $EPOCH"
echo "Max parallel:  $MAX_JOBS"
echo "Pending tasks: $TOTAL"
echo "========================================"

if [[ $TOTAL -eq 0 ]]; then
    echo "All targets already complete. Nothing to do."
    exit 0
fi

# Per-phase summary
for phase in "${PHASE_LIST[@]}"; do
    cnt=0
    for t in "${TASKS[@]}"; do
        [[ "${t%%:*}" == "$phase" ]] && cnt=$((cnt + 1))
    done
    printf "  %-11s %d\n" "$phase:" "$cnt"
done
echo ""

if $DRY_RUN; then
    echo "[DRY RUN] Pending tasks:"
    for t in "${TASKS[@]}"; do echo "  $t"; done
    exit 0
fi

# =============================================================================
# Parallel execution
# =============================================================================
LOG_DIR="$HOME/tmp/pending_analysis_logs"
mkdir -p "$LOG_DIR"

RUNNING_PIDS=()
RUNNING_NAMES=()
COMPLETED=0
FAILED=0
STARTED=0

reap_finished() {
    local new_pids=()
    local new_names=()
    for i in "${!RUNNING_PIDS[@]}"; do
        local pid=${RUNNING_PIDS[$i]}
        local name=${RUNNING_NAMES[$i]}
        if kill -0 "$pid" 2>/dev/null; then
            new_pids+=("$pid")
            new_names+=("$name")
        else
            local code=0
            wait "$pid" || code=$?
            COMPLETED=$((COMPLETED + 1))
            if [[ $code -eq 0 ]]; then
                echo "[DONE]  $name  ($COMPLETED/$TOTAL)"
            else
                echo "[FAIL]  $name  (exit $code) — $LOG_DIR/${name}.log"
                FAILED=$((FAILED + 1))
            fi
        fi
    done
    RUNNING_PIDS=("${new_pids[@]}")
    RUNNING_NAMES=("${new_names[@]}")
}

wait_for_slot() {
    while [[ ${#RUNNING_PIDS[@]} -ge $MAX_JOBS ]]; do
        reap_finished
        if [[ ${#RUNNING_PIDS[@]} -ge $MAX_JOBS ]]; then
            sleep 5
        fi
    done
}

for t in "${TASKS[@]}"; do
    phase=${t%%:*}
    rest=${t#*:}
    io=${rest%%:*}
    mdl=${rest#*:}
    exp="${EXP_PREFIX}${io}_${mdl}"
    name="${phase}__${exp}"

    wait_for_slot

    STARTED=$((STARTED + 1))
    echo "[START] $name  ($STARTED/$TOTAL, running: $((${#RUNNING_PIDS[@]} + 1)))"

    runner=$(runner_for_phase "$phase")
    python "$runner" --config-name="$CONFIG_NAME" \
        "+io=${io}" "+model=${mdl}" "experiment.name=${exp}" \
        "${phase}.epoch=${EPOCH}" "${EXTRA_ARGS[@]}" \
        > "$LOG_DIR/${name}.log" 2>&1 &
    RUNNING_PIDS+=($!)
    RUNNING_NAMES+=("$name")
done

# Drain remaining jobs
while [[ ${#RUNNING_PIDS[@]} -gt 0 ]]; do
    reap_finished
    if [[ ${#RUNNING_PIDS[@]} -gt 0 ]]; then
        sleep 5
    fi
done

echo ""
echo "========================================"
echo "Pending Analysis Complete"
echo "========================================"
echo "Total:     $TOTAL"
echo "Succeeded: $((COMPLETED - FAILED))"
echo "Failed:    $FAILED"
echo "Logs:      $LOG_DIR"
echo "========================================"

[[ $FAILED -eq 0 ]] || exit 1
