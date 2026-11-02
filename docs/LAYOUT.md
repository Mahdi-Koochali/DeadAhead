# Layout, licences and citation

[Back to the README](../README.md)

## Repository layout

```
deadahead/                    the Python package (below)
configs/                      settings of the windows, the survival head, VideoMAE and the evaluation (alarms, horizons)
splits/folds.json             the nine leave-one-group-out folds
data/celldeathdetect_md5.txt  checksums of the three CellDeathDetect archives
docs/SUPPLEMENT.md            the full results: Table 1, Figure 2, Table 2 and the supplement S1 to S8
docs/REPRODUCE.md             the 5-minute check, the claim table and the installation
docs/METHOD.md                what protects the result, the closest work and the method
docs/TRAINING.md              the pretrained models, the data and training from scratch
docs/LAYOUT.md                this page: the repository layout, the licences and the citation
docs/                         data card, model card, weights and figures (docs/figures/data/ holds the plotted values and the small tables behind them)
environment/                  the exact package versions of the paper's runs (CPU and GPU environments)
scripts/slurm/                two example SLURM scripts, for the VideoMAE and the baseline stages
tests/                        unit and end-to-end tests on synthetic data (pytest)
.github/workflows/tests.yml   the CPU tests, run on every push with the pinned versions
```

| Module | What it does |
|---|---|
| `deadahead/cli.py` | Command-line entry point: ``deadahead <command> [options]`` |
| `deadahead/paths.py` | Where the data and outputs live |
| `deadahead/data/inventory.py` | Movie table of the extracted CellDeathDetect archives: one row per movie with its phase stack, mask stack and label spreadsheet, found by file name |
| `deadahead/data/tables.py` | Movie, track and event tables of CellDeathDetect, from the movie table, the label spreadsheets and the stack headers |
| `deadahead/data/windows.py` | Forecast rows (``windows.parquet``) and the nine leave-one-group-out folds (``folds.json``) |
| `deadahead/features/frame_measures.py` | Per-cell, per-frame shape and phase measures from the curated masks (``cells.parquet``) |
| `deadahead/features/history.py` | Handcrafted history features: 14 per-cell QPI measures summarised over each cell's own past |
| `deadahead/video/adapt.py` | Masked-autoencoder adaptation of VideoMAE-large on one fold's unlabelled training clips |
| `deadahead/video/clips.py` | Clip index for the video model: 16 distinct frames over the past 120 min (40 frames) per landmark |
| `deadahead/video/crops.py` | All-frames crop store for the video model: one 128 x 128 crop per cell and frame, centred on the cell's centroid, raw phase in float16, zero-padded where the crop runs off the field of view |
| `deadahead/video/features.py` | Per-row features of VideoMAE (1,024 values) for every landmark row of the primary panel |
| `deadahead/video/finetune.py` | Fine-tune the whole VideoMAE-large end to end on 30-minute death labels, starting from the fold's adapted encoder |
| `deadahead/video/model.py` | VideoMAE-large for quantitative phase clips: the pinned checkpoint, model construction, attention biases, layer-wise learning-rate decay and the phase intensity map |
| `deadahead/video/predict.py` | Held-out predictions of a fine-tuned VideoMAE: three class probabilities per 30-minute row |
| `deadahead/video/selection.py` | Choose the fine-tuning setting of each fold on its selection group |
| `deadahead/video/test_time.py` | Test-time changes to the fine-tuned video model: the tracked cell hidden, its 16 frames shuffled, or the last frame repeated 16 times |
| `deadahead/video/training.py` | Shared parts of VideoMAE training: the phase map, augmentation, model input, tube masking, the learning-rate schedule and resumable state files |
| `deadahead/survival/boosters.py` | Cumulative incidence from a saved DeadAhead booster, and capture of the booster a fit trains |
| `deadahead/survival/censoring.py` | The injected-censoring experiment: does handling censored cells correctly matter when censoring is added? |
| `deadahead/survival/config.py` | Read the YAML settings in ``configs/`` |
| `deadahead/survival/frames.py` | The inputs of the DeadAhead head: landmark rows, training and evaluation labels, and the feature matrix |
| `deadahead/survival/head.py` | The DeadAhead survival head: discrete-time competing risks fitted with LightGBM |
| `deadahead/survival/oracle.py` | Evaluation labels: the outcome of every (landmark, horizon), read from the complete movie |
| `deadahead/survival/train.py` | Train the DeadAhead head for one held-out group and seed, and write its predictions and booster |
| `deadahead/baselines/deephit.py` | DeepHit (Lee et al., 2018) on DeadAhead's fixed inputs, one held-out group and seed at a time |
| `deadahead/baselines/dinov2_adapt.py` | Adapt DINOv2 ViT-B/14 to the phase images of the training groups of one fold, without labels |
| `deadahead/baselines/dinov2_common.py` | DINOv2 ViT-B/14 pieces shared by the DINOv2 baselines: loading the encoder, the phase map, the augmentation, the contrastive objective, and one forward pass over a crop store |
| `deadahead/baselines/dinov2_embed.py` | Per-frame DINOv2 embeddings of every crop of every movie, by the encoder of one fold |
| `deadahead/baselines/dinov2_finetune.py` | Fine-tune DINOv2 ViT-B/14 on the 30-minute death labels of the training groups of one fold |
| `deadahead/baselines/dinov2_survival.py` | DINOv2 + handcrafted through DeadAhead's survival head (the same-head comparison of the gap and fixed-horizon results), one held-out group and seed at a time |
| `deadahead/baselines/elapsed_time.py` | The elapsed-time baseline: a weighted logistic regression on the time since the movie started |
| `deadahead/baselines/heads.py` | LightGBM heads of the comparison methods of Table 1, one held-out group at a time |
| `deadahead/evaluation/aggregate.py` | Aggregation over held-out groups: seed means, the macro over groups, paired differences and the decision rule |
| `deadahead/evaluation/alarms.py` | Warnings: the warning rows and their roles, training-only alarm thresholds, sustained coverage before death |
| `deadahead/evaluation/censoring.py` | Scores of the injected-censoring experiment: oracle AUPRC per unit, the macro over groups and paired contrasts |
| `deadahead/evaluation/horizons.py` | Outcome sets at a fixed horizon from the survival outcomes (T, delta, C in minutes), alarm counts on them, and the per-unit and macro summaries of the fixed-horizon results |
| `deadahead/evaluation/metrics.py` | Ranking metrics for one cause at one horizon (AUPRC, AUROC), numpy only |
| `deadahead/evaluation/rows.py` | Forecast rows: row keys, score alignment, the evaluation cohort and the outcome sets P, Q, N, U |
| `deadahead/evaluation/survival_metrics.py` | Survival metrics of one held-out group at fixed horizons, numpy only: IPCW Brier score and IPA, Wolbers' concordance, expected over observed incidence and the IPCW calibration intercept and slope |
| `deadahead/evaluation/table1.py` | Table 1: AUPRC at 30 min per method, cause, held-out group and seed, the macros and the paired contrasts |
| `deadahead/reproduce/alarm_rate.py` | Reproduce the post hoc alarm-rate analysis: the share of deaths warned against the alarm rate, and one comparison at a matched achieved alarm rate |
| `deadahead/reproduce/bundle.py` | Read the predictions bundle: the forecast rows, the cell event types and the held-out predictions per method |
| `deadahead/reproduce/censoring.py` | Reproduce the injected-censoring experiment (supplementary S6) from the bundle's stored unit predictions |
| `deadahead/reproduce/compare.py` | Leaf-by-leaf comparison of two result trees, for the cross-checks between reproduce parts |
| `deadahead/reproduce/data.py` | Reproduce the data counts of the paper from the bundle's tables: the cells of the 63 movies and of the 62 scored movies by outcome, the same at the cut, and the forecasts and positives at 30 min |
| `deadahead/reproduce/decomposition.py` | Reproduce the decomposition results (supplementary S4) from the bundle, on Table 1's rows and cohort |
| `deadahead/reproduce/gap.py` | Reproduce the post hoc gap analysis: deaths 60 to 120 min ahead, with the forecasts whose cell died within the first 60 min left out |
| `deadahead/reproduce/gap_window.py` | Reproduce the post hoc window analysis: AUPRC of deaths g to H min ahead, with the forecasts whose cell died within the first g min left out, at H = 120 min (g = 0, 30, 60, 90) and H = 240 min (g = 0, 30, 60, 90, 120) |
| `deadahead/reproduce/horizons.py` | Reproduce the fixed-horizon results: at h = 30, 60, 120 and 240 min, DeadAhead and elapsed time at training-only alarm thresholds (precision, sensitivity, AUPRC, prevalence), the AUPRC of the other survival methods, and the survival estimators of the four survival methods (IPCW Brier score, IPA, Wolbers' concordance, E/O, the IPCW calibration intercept and slope) |
| `deadahead/reproduce/image_warnings.py` | Reproduce, post hoc, the warnings of the image baseline (DINOv2 image features + handcrafted features, refit head) under the same training-only alarm thresholds as DeadAhead's warnings |
| `deadahead/reproduce/main.py` | deadahead reproduce: recompute the paper's Table 1, the warnings and the supplementary results from a predictions bundle |
| `deadahead/reproduce/sensitivity.py` | Reproduce the sensitivity analyses that the paper quotes: Table 1 restricted to rows whose 30-min window ends inside the common cut (frame 438), and the warnings at a common landmark (rows t <= 358, follow-up truncated at frame 438, thresholds reselected on the restricted training reference rows) |
| `deadahead/reproduce/table1.py` | Reproduce Table 1 from a predictions bundle: AUPRC at 30 min on the common evaluation cohort, every method |
| `deadahead/reproduce/warning_times.py` | Reproduce the warnings from a predictions bundle: training-only alarm thresholds at targets of 0.2 and 0.4 alarms per event-free reference cell-hour, applied to DeadAhead's held-out F_c(30) and to the elapsed-time score |
| `deadahead/reproduce/within_time.py` | Reproduce the post hoc within-hour scoring: the 30-min ranking when a forecast competes only with forecasts of the same group in the same hour of imaging |
| `deadahead/report/bundle_tables.py` | Tables of the supplement read from the predictions bundle, not from the reproduce output: the weight of each outcome in the training rows, and two post hoc views of the warnings (the alarm share per 30-min bin before death, the match group by group) |
| `deadahead/report/figures.py` | Figures for the supplement: the warning curves of Figure 2, the overview of Table 1, the figures of S1, S2 and S4 and the post hoc figures of S8 |
| `deadahead/yeast/constants.py` | Constants and folds of the budding-yeast death probe (DetecDiv, Aspert et al. 2022, Zenodo 6078462, CC BY 4.0) |
| `deadahead/yeast/data.py` | Yeast data: landmark rows, 16-frame clips, the pixel store, the per-fold channel and intensity map, the picture measurements and the per-fold row files |
| `deadahead/yeast/finetune.py` | Short fine-tune of VideoMAE-large on yeast death within 30 min, from one of two starts, then the held-out pass |
| `deadahead/yeast/leadtime.py` | Lead time of a death forecast on one held-out position |
| `deadahead/yeast/lifespans.py` | Yeast lifespans from the human per-frame classes of DetecDiv, the landmark rows and their label features |
| `deadahead/yeast/metrics.py` | The ranking metrics of the yeast probe: AUPRC and AUROC, as in the main evaluation |
| `deadahead/yeast/model.py` | VideoMAE-large on yeast trap videos: the intensity map, the two-class model of each start, and the forward pass |
| `deadahead/yeast/predict.py` | Yeast predictions of a VideoMAE-large: a trained 2-class yeast model on its held-out position, or a released 3-class fine-tuned VideoMAE applied with no yeast training on every position |
| `deadahead/yeast/rules.py` | The simple rules on each held-out position, the best of them, and the one-cause survival head |
| `deadahead/yeast/score.py` | Scores of the yeast death probe from its stored predictions: VideoMAE fine-tuned on six positions (from two starting points) against the best simple rule on each of the 7 held-out positions, at 30 min |
| `deadahead/yeast/survival.py` | The survival layer of one fine-tuned yeast run: the one-cause survival head (``rules.survival_cif``: bins of 6 frames, 8 bins, 200 rounds, weight 1, the LightGBM seed is the run's seed) fitted on the training positions' landmarks embedded by that run's own model, scored on the predicted position (the held-out position; the threshold position for a threshold run) |
| `deadahead/yeast/survival_scores.py` | Threshold-free survival scores of the survival layer (each seed's survival head on its held-out position, the mean over seeds per position) and of the survival rules (``label_surv``, ``label_pic_surv``, ``pic_bright_surv``, ``pic_size_surv``), on each held-out position |

**Terms used in the code.**

| Term | Meaning |
|---|---|
| *landmark* | one forecast: a tracked cell at one forecast time, every 30 min |
| *held-out group* (`--fold`, `--group`) | the line-by-treatment group a model never saw |
| outcome sets P, Q, N, U | for one route and horizon: dies by that route within the horizon, dies by the other route, event-free through the horizon, or follow-up ends inside the horizon (never scored as a negative) |
| *reference forecasts* | event-free landmarks followed for 240 min; alarm thresholds are set on them |
| *complete-movie labels* (the *oracle* labels in the code) | labels from each cell's full follow-up, while training labels stop at the cut |
| *primary panel* (`primary_439`) | the forecast rows with every movie cut at frame 439 |

## Licences

| Material | Licence |
|---|---|
| Code | MIT ([`LICENSE`](../LICENSE)) |
| Tables and predictions derived from CellDeathDetect, including the predictions bundle | CC BY 4.0, with attribution to Vicar et al. (2020) |
| Fine-tuned VideoMAE weights | CC BY-NC 4.0, inherited from the VideoMAE checkpoint (MCG-NJU/videomae-large) |
| DeadAhead survival heads (`deadahead_head__*.txt`) | MIT; their inputs include the features of the fine-tuned VideoMAE weights (CC BY-NC 4.0) |
| Yeast predictions bundle, derived from DetecDiv (Aspert et al., 2022) | CC BY 4.0 |
| Figures and figure data in `docs/figures/` | CC BY 4.0; Figure 1 and the quiz show CellDeathDetect crops (Vicar et al., 2020) |

Third-party components and datasets are listed in [`NOTICE.md`](../NOTICE.md).

## Citation

If you use this code, please cite the paper:

```bibtex
@misc{koochali2026deadahead,
  title  = {{DeadAhead}: Forecasting when and how a cell will die from quantitative phase video},
  author = {Koochali, Mohammadmahdi and Khalid, Nabeel and Dengel, Andreas and Ahmed, Sheraz},
  year   = {2026},
  note   = {Under review}
}
```

and the CellDeathDetect dataset (Vicar et al., 2020) and VideoMAE (Tong et al., 2022).
