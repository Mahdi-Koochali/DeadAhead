#!/bin/bash
#SBATCH --job-name=videomae
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --open-mode=append
#SBATCH --output=%x_%A_%a.out
#
# One stage of the VideoMAE pipeline for one array index; one GPU per job.
# Set DEADAHEAD_DATA to the data folder and PYTHON to an interpreter with the [video] extras.
# Order (each stage needs the previous one finished):
#   sbatch --array=0-8  scripts/slurm/videomae.sh adapt      # 9 held-out groups
#   sbatch --array=0-35 scripts/slurm/videomae.sh grid       # 9 groups x 4 settings, seed 2601562
#   sbatch --array=0-35 scripts/slurm/videomae.sh score      # selection-group AUPRC of each grid run
#   sbatch --array=0-8  scripts/slurm/videomae.sh choose     # 9 groups: the best-scoring setting (CPU work)
#   sbatch --array=0-26 scripts/slurm/videomae.sh final      # 9 groups x 3 seeds, chosen setting
#   sbatch --array=0-26 scripts/slurm/videomae.sh predict
#   sbatch --array=0-26 scripts/slurm/videomae.sh features
#   sbatch --array=0-8  scripts/slurm/videomae.sh features-adapted   # adapted encoders, before fine-tuning
#   sbatch --array=0-26 scripts/slurm/videomae.sh test-time          # the four test-time conditions
# The choose stage needs no GPU; it also runs directly, one index at a time:
#   SLURM_ARRAY_TASK_ID=0 scripts/slurm/videomae.sh choose   # indexes 0 to 8
# The crop store and clip indexes are built first on a CPU, and the mask store before test-time:
#   python -m deadahead.video.crops && python -m deadahead.video.clips
#   python -m deadahead.video.test_time masks
#
# Training saves its state every 500 steps and when the job is about to reach its time limit
# (USR1, forwarded below); a stopped run exits with code 10 and this script requeues it, and the
# requeued job resumes from the saved state. Use --micro-batch 32 on GPUs with more than 80 GB.
set -uo pipefail

STAGE="${1:?stage: adapt | grid | score | choose | final | predict | features | features-adapted | test-time}"
PYTHON="${PYTHON:-python}"
I="${SLURM_ARRAY_TASK_ID:?submit as an array job}"
GROUPS_=(PNT1A_bp PNT1A_do PNT1A_st DU145_bp DU145_do DU145_st LNCaP_bp LNCaP_do LNCaP_st)
LRS=(5e-5 5e-5 2e-4 2e-4)
LDS=(0.75 0.9 0.75 0.9)
SEEDS=(2601562 2601563 2601564)
EXTRA=("${@:2}")

chosen() {   # backbone lr and layer decay chosen for a group
    "$PYTHON" - "$1" <<'PY'
import json, sys
from deadahead import paths
w = json.loads((paths.runs_dir() / "videomae" / "select" / f"selected__{sys.argv[1]}.json").read_text())["winner"]
print(f"{w['backbone_lr']:g} {w['layer_decay']:g}")
PY
}

case "$STAGE" in
    adapt)    G=${GROUPS_[$I]}; CMD=(-m deadahead.video.adapt --fold "$G") ;;
    grid)     G=${GROUPS_[$((I / 4))]}; C=$((I % 4))
              CMD=(-m deadahead.video.finetune --fold "$G" --backbone-lr "${LRS[$C]}" --layer-decay "${LDS[$C]}") ;;
    score)    G=${GROUPS_[$((I / 4))]}; C=$((I % 4))
              CMD=(-m deadahead.video.selection score --fold "$G" --backbone-lr "${LRS[$C]}" --layer-decay "${LDS[$C]}") ;;
    choose)   G=${GROUPS_[$I]}; CMD=(-m deadahead.video.selection choose --fold "$G") ;;
    features-adapted)
              G=${GROUPS_[$I]}; CMD=(-m deadahead.video.features --fold "$G" --adapted) ;;
    final|predict|features|test-time)
              G=${GROUPS_[$((I / 3))]}; S=${SEEDS[$((I % 3))]}
              read -r LR LD < <(chosen "$G") || { echo "no chosen setting for $G" >&2; exit 3; }
              case "$STAGE" in
                  final)    CMD=(-m deadahead.video.finetune --fold "$G" --backbone-lr "$LR" --layer-decay "$LD" --final --seed "$S") ;;
                  predict)  CMD=(-m deadahead.video.predict --fold "$G" --seed "$S" --backbone-lr "$LR" --layer-decay "$LD") ;;
                  features) CMD=(-m deadahead.video.features --fold "$G" --seed "$S" --backbone-lr "$LR" --layer-decay "$LD") ;;
                  test-time) CMD=(-m deadahead.video.test_time predict --fold "$G" --seed "$S" --backbone-lr "$LR" --layer-decay "$LD") ;;
              esac ;;
    *) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac

echo "$(date '+%F %T') $STAGE index $I: $PYTHON ${CMD[*]} ${EXTRA[*]}"
"$PYTHON" "${CMD[@]}" "${EXTRA[@]}" &
PID=$!
trap 'kill -USR1 "$PID" 2>/dev/null' USR1
wait "$PID"; RC=$?
while kill -0 "$PID" 2>/dev/null; do wait "$PID"; RC=$?; done   # wait again after a forwarded signal
if [ "$RC" -eq 10 ] && [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "$(date '+%F %T') state saved; requeueing"
    scontrol requeue "${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${I}" 2>/dev/null || scontrol requeue "$SLURM_JOB_ID"
fi
exit "$RC"
