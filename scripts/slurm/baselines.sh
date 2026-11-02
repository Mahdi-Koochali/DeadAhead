#!/bin/bash
#SBATCH --job-name=baselines
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=%x_%A_%a.out
#
# One stage of the survival heads and the comparison methods for one array index. Set DEADAHEAD_DATA to the data folder
# and PYTHON to an interpreter with the package installed (the [video] extras for the DINOv2 and DeepHit stages).
# The DeadAhead head and its cross-fitted inner heads need the VideoMAE features (scripts/slurm/videomae.sh):
#   sbatch --array=0-26  scripts/slurm/baselines.sh train-head        # 9 groups x 3 seeds
#   sbatch --array=0-215 scripts/slurm/baselines.sh train-head-inner  # 9 groups x 3 seeds x 8 inner groups
# Order (each stage needs the previous one finished; the DINOv2 stages need a GPU: add --gres=gpu:1):
#   sbatch --array=0-8 --gres=gpu:1 scripts/slurm/baselines.sh dinov2-adapt            # tile pool + adaptation, 9 groups
#   sbatch --array=0-8 --gres=gpu:1 scripts/slurm/baselines.sh dinov2-embed            # every movie, each group's encoder
#   sbatch --array=0-8 --gres=gpu:1 scripts/slurm/baselines.sh dinov2-finetune         # on the death labels
#   sbatch --array=0-8 --gres=gpu:1 scripts/slurm/baselines.sh dinov2-embed-finetuned
# The DINOv2 stages read 128-pixel crops every 5 frames, cut first on a CPU:
#   python -m deadahead.video.crops --stride 5 --out-dir "$DEADAHEAD_DATA/crops/frames128"
#   sbatch --array=0-35 scripts/slurm/baselines.sh heads             # 4 methods x 9 groups, 3 seeds each, both policies
#   sbatch --array=0-8  scripts/slurm/baselines.sh heads-inner       # DINOv2 + handcrafted: cross-fitted training scores
#   sbatch --array=0-26 scripts/slurm/baselines.sh dinov2-survival   # 9 groups x 3 seeds (DINOv2 + handcrafted, same head)
#   sbatch --array=0-26 scripts/slurm/baselines.sh deephit           # 9 groups x 3 seeds (needs the VideoMAE features)
# The handcrafted head needs only the tables; the image heads need the embedding stores of the DINOv2 stages.
# A CPU stage also runs directly, one index at a time:
#   SLURM_ARRAY_TASK_ID=0 scripts/slurm/baselines.sh heads
# Finished outputs are skipped by deephit (selection and final fits) and resumed from their per-epoch checkpoints.
set -uo pipefail

STAGE="${1:?stage: train-head | train-head-inner | dinov2-adapt | dinov2-embed | dinov2-finetune | dinov2-embed-finetuned | heads | heads-inner | dinov2-survival | deephit}"
PYTHON="${PYTHON:-python}"
I="${SLURM_ARRAY_TASK_ID:?submit as an array job}"
GROUPS_=(DU145_bp DU145_do DU145_st LNCaP_bp LNCaP_do LNCaP_st PNT1A_bp PNT1A_do PNT1A_st)
METHODS=(handcrafted dinov2 dinov2_handcrafted dinov2_finetuned_handcrafted)
SEEDS=(2601562 2601563 2601564)
THREADS="${SLURM_CPUS_PER_TASK:-8}"
EXTRA=("${@:2}")

case "$STAGE" in
    train-head)
              G=${GROUPS_[$((I / 3))]}; S=${SEEDS[$((I % 3))]}
              CMD=(-m deadahead.survival.train --group "$G" --seed "$S" --threads 8) ;;
    train-head-inner)
              G=${GROUPS_[$((I / 24))]}; S=${SEEDS[$((I / 8 % 3))]}
              INNER=(); for H in "${GROUPS_[@]}"; do [ "$H" = "$G" ] || INNER+=("$H"); done
              CMD=(-m deadahead.survival.train --group "$G" --seed "$S" --inner-group "${INNER[$((I % 8))]}" --threads 8) ;;
    dinov2-adapt)    CMD=(-m deadahead.baselines.dinov2_adapt --fold "${GROUPS_[$I]}") ;;
    dinov2-embed)    CMD=(-m deadahead.baselines.dinov2_embed --fold "${GROUPS_[$I]}" --encoder adapted) ;;
    dinov2-finetune) CMD=(-m deadahead.baselines.dinov2_finetune --fold "${GROUPS_[$I]}") ;;
    dinov2-embed-finetuned)
                     CMD=(-m deadahead.baselines.dinov2_embed --fold "${GROUPS_[$I]}" --encoder finetuned) ;;
    heads)    M=${METHODS[$((I / 9))]}; G=${GROUPS_[$((I % 9))]}
              CMD=(-m deadahead.baselines.heads --method "$M" --group "$G" --policy both --threads "$THREADS") ;;
    heads-inner)
              G=${GROUPS_[$I]}
              CMD=(-m deadahead.baselines.heads --method dinov2_handcrafted --group "$G" --policy refit --inner --threads "$THREADS") ;;
    dinov2-survival)
              G=${GROUPS_[$((I / 3))]}; S=${SEEDS[$((I % 3))]}
              CMD=(-m deadahead.baselines.dinov2_survival --group "$G" --seed "$S" --threads 8) ;;
    deephit)  G=${GROUPS_[$((I / 3))]}; S=${SEEDS[$((I % 3))]}
              CMD=(-m deadahead.baselines.deephit --group "$G" --seed "$S" --threads 8) ;;
    *) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac

echo "$(date '+%F %T') $STAGE index $I: $PYTHON ${CMD[*]} ${EXTRA[*]}"
exec "$PYTHON" "${CMD[@]}" "${EXTRA[@]}"
