# Models, data and training

[Back to the README](../README.md)

## Pretrained models

| Files | Number | Licence |
|---|:---:|---|
| Fine-tuned VideoMAE-L, one per held-out group and seed | 27 | CC BY-NC 4.0 |
| DeadAhead survival heads | 27 | MIT |

The [model card](MODEL_CARD.md) says what they are and how to use them; sizes and checksums are in
[`docs/WEIGHTS.md`](WEIGHTS.md). Download: https://zenodo.org/records/23883153.

## Data

CellDeathDetect is public on Zenodo (record 2601562, CC BY 4.0). Download the three archives, check them against
[`data/celldeathdetect_md5.txt`](../data/celldeathdetect_md5.txt), and extract them into
`$DEADAHEAD_DATA/raw/celldeathdetect/` (by default `data/` inside the repository). [`docs/DATA.md`](DATA.md), the data card, gives
the commands and the counts a correct build prints.

## Training from scratch

The full pipeline, in the order the paper ran it. Every command takes `--help`, and data locations come from
`DEADAHEAD_DATA`. The code is that of the paper's runs, reorganised: its CPU data stages were re-run on the real data
and give the same tables and handcrafted features (the per-cell measures were compared on three movies), the
released weights were checked against the stored predictions, and the code was compared with the paper's source,
where each difference has a recorded reason. The GPU training loops were run here only on small synthetic inputs, in
the tests. Training VideoMAE took about 187.4 GPU-hours in total over the 9 held-out groups
(adaptation, four settings on the inner group, three final seeds). The survival head trains on a CPU in minutes per
fold and seed; the stored inner heads took 95 to 270 s each at 8 threads, with a peak of about 12 GB.

| Step | Command | Hardware |
|---|---|---|
| List the movies of the extracted archives | `deadahead data-inventory` | CPU |
| Movies, tracks and events tables | `deadahead data-tables` | CPU |
| Per-cell, per-frame measures (about 1.5 h) | `deadahead data-cells` | CPU |
| Forecast rows and folds | `deadahead data-windows` | CPU |
| Handcrafted features (112 per forecast) | `deadahead history-features` | CPU |
| Crops of every tracked cell (128 px, every frame) | `deadahead video-crops` | CPU |
| Clip index (16 frames over the past 120 min) | `deadahead video-clips` | CPU |
| Adapt VideoMAE on the eight training groups | `deadahead video-adapt --fold <group>` | GPU |
| Fine-tune the four settings | `deadahead video-finetune --fold <group> --backbone-lr {5e-05,0.0002} --layer-decay {0.75,0.9}` | GPU |
| Pick the setting on the inner group | `deadahead video-select score ...`, then `deadahead video-select choose --fold <group>` | GPU, CPU |
| Final runs, three seeds | `deadahead video-finetune --fold <group> --backbone-lr <lr> --layer-decay <ld> --final --seed <seed>` | GPU |
| Held-out predictions and pooled features | `deadahead video-predict ...`, `deadahead video-features --fold <group> --seed <seed>` | GPU |
| Survival head (DeadAhead) | `deadahead train-head --group <group> --seed <seed>` | CPU |
| Cross-fitted inner heads (the training-only alarm thresholds), 8 per group and seed, 216 in all | `deadahead train-head --group <group> --seed <seed> --inner-group <other group>` | CPU |
| Adapted features (S4) | `deadahead video-features --fold <group> --adapted` writes `$DEADAHEAD_DATA/runs/videomae/features/adapted/<group>.npy` and `rows.parquet` (`sbatch --array=0-8 scripts/slurm/videomae.sh features-adapted`) | GPU |
| Survival head on the adapted features (S4) | `deadahead train-head --group <group> --seed <seed> --method videomae_adapted_survival --features $DEADAHEAD_DATA/runs/videomae/features/adapted/<group>.npy` | CPU |
| Test-time changes (S4) | `deadahead video-test-time masks`, then `deadahead video-test-time predict` (on SLURM: `python -m deadahead.video.test_time masks` on a CPU, then `sbatch --array=0-26 scripts/slurm/videomae.sh test-time`) | CPU, GPU |
| Censoring experiment (S6), one unit per group, regime and seed | `deadahead censoring-experiment --all` | CPU |
| Crops every 5 frames for the DINOv2 baselines | `deadahead video-crops --stride 5 --out-dir $DEADAHEAD_DATA/crops/frames128` | CPU |
| Baselines: DINOv2 adaptation, embeddings, label fine-tuning | `deadahead baseline-dinov2-adapt`, `baseline-dinov2-embed`, `baseline-dinov2-finetune` | GPU |
| Baselines: LightGBM heads, same-head DINOv2 + handcrafted | `deadahead baseline-heads`, `baseline-dinov2-survival` | CPU |
| Baseline: DeepHit | `deadahead baseline-deephit --group <group> --seed <seed>` | CPU (PyTorch, the `[video]` extra) |
| Table 1, the warnings, S1 to S6 and S8 from the released predictions | `deadahead reproduce --part all` | CPU |
| Every figure from that output | `deadahead plot-figures` | CPU |

On SLURM: `sbatch --array=0-26 scripts/slurm/baselines.sh train-head`, then `sbatch --array=0-215 scripts/slurm/baselines.sh train-head-inner`. Without SLURM, a loop over the folds does it:

```bash
for g in PNT1A_bp PNT1A_do PNT1A_st DU145_bp DU145_do DU145_st LNCaP_bp LNCaP_do LNCaP_st; do
  for s in 2601562 2601563 2601564; do
    deadahead train-head --group $g --seed $s
    for h in PNT1A_bp PNT1A_do PNT1A_st DU145_bp DU145_do DU145_st LNCaP_bp LNCaP_do LNCaP_st; do
      [ "$h" = "$g" ] || deadahead train-head --group $g --seed $s --inner-group $h
    done
  done
done
```

| Topic | Details |
|---|---|
| Inner heads | They read the features of the model that held out `<group>`, same seed, never those of the inner group. Outputs: `pred/deadahead__<group>__s<seed>__primary__h<inner group>.npz` and `boosters/deadahead_head__<group>__seed<seed>__h<inner group>.txt`; `p_<cause>_30` is the score column of `inner_predictions.parquet`, and `p_<cause>_60`, `120` and `240` are `F_<cause>_<h>` of `inner_cumulative_incidence.parquet`. Two refits agreed with the stored units to about 4e-16; the last bits differ with the thread count. |
| Adapted features | `--features` takes the array of `video-features --fold <group> --adapted`, one per group, with no seed. `video-features` needs `--seed` unless `--adapted` is given. |
| Refused restarts | A changed `--steps` or `--seed`, or a re-made adapted encoder for fine-tuning, is refused: exit 3 if the checkpoint exists, an error if only the state file is left. Delete the old `.pt`, `.json` and `.state.pt` to train again. |
| Disk | Each VideoMAE run leaves a 3.6 GB state file beside its 1.2 GB checkpoint; the state files of the 72 runs take about 260 GB. |
| Test-time masks | `video-test-time masks` exits 1 on every run, reruns included, while any of its five checks has failed. Delete the named movies' files under the mask store to build them again. |
| No GPU | The video commands warn and run in fp32 on the CPU; `--device cpu` silences the warning. |
| Released weights | `--weights` with a released file name must match `--fold` and `--seed`. `video-select --fold` takes only the nine groups. `video-crops --stride` other than 1 needs `--out-dir`. |
| Untyped deaths | `--untyped at_risk_to_cut` has its own file names, with `__at_risk_to_cut`, on `train-head` and `baseline-dinov2-survival`. |
| Baselines | `baseline-heads --inner` writes `<unit>__inner.parquet` with the bundle's `outer_group` and `inner_group` columns plus `example_id`, `horizon_min`, `label` and `window_weight`. `baseline-deephit` counts a unit as finished when `<unit>.json` exists; the record is written last. |
| SLURM logs | `videomae_<job>_<index>.out` and `baselines_<job>_<index>.out`; a requeued job appends to its log. |

`deadahead reproduce` scores the released predictions bundle, not the outputs of the steps above: to score your own
runs, arrange their predictions in the bundle layout documented in `deadahead/reproduce/bundle.py`, with a
`MANIFEST.json` that lists every file.

The yeast probe (S7) reads the DetecDiv archive (see [`docs/DATA.md`](DATA.md)):

| Step | Command | Hardware |
|---|---|---|
| Lifespan labels, landmark rows, pixel store and per-fold inputs | `deadahead yeast-data --zip <lifespanAnnotation.zip>` | CPU |
| Fine-tune VideoMAE on six positions from either start, then predict the held-out one | `deadahead yeast-finetune --start {videomae_finetuned,videomae_public} --heldout <position> --seed <seed>` | GPU |
| Predictions from saved yeast weights, or of a released model without yeast training | `deadahead yeast-predict --weights <file> (--heldout <position> \| --zero-shot)` | GPU |
| The survival layer on each run's embeddings | `deadahead yeast-survival --run <run>` | CPU |
| The simple rules on each held-out position (`--horizons` takes only 30, 60, 120, 240) | `deadahead yeast-rules` | CPU |
| The scores of S7 from the yeast bundle | `deadahead yeast-score` | CPU |

`yeast-finetune` removes `weights/<run>.state.pt` once the weights are saved, and returns 5 with no prediction file when a
value is not finite.

To use the released models instead of training, pass `--weights videomae_large_finetuned__<group>__seed<seed>.safetensors`
to `video-features`, then `--booster deadahead_head__<group>__seed<seed>.txt` to `train-head` to predict without
refitting. The 216 inner boosters are not released, so `--booster` with `--inner-group` needs your own. Two example SLURM
scripts, for the VideoMAE stages and for the baseline stages, are in `scripts/slurm/`.
