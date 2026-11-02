# Method and closest work

[Back to the README](../README.md)

## What protects the result

Leakage is what to look for in a forecast. Each line below is a test in this repository; none needs the data, and
together they take about 11 s (`pip install -e ".[dev]"` first):

```bash
pytest tests/test_history_features.py::test_no_future_frames tests/test_data_windows.py::test_fold_check_rejects_a_leak tests/test_data_windows.py::test_labels_and_landmarks tests/test_survival_head.py::test_no_training_landmark_is_at_risk_past_the_cut tests/test_survival_head.py::test_inner_group_refuses_the_held_out_group tests/test_image_warnings_reproduce.py::test_image_reference_scores_pool_the_training_groups_reference_rows tests/test_horizons.py::test_inner_reference_scores_read_the_table_of_their_horizon
```

| Risk | What the test proves | Test |
|---|---|---|
| Future frames reach the features | Changing or deleting every frame after a cut leaves the handcrafted features at or before that cut exactly as they were. This test covers the handcrafted features | [`tests/test_history_features.py`](../tests/test_history_features.py), `test_no_future_frames` |
| A held-out movie enters training | `check_folds` raises an error when a fold's held-out movie is added to its training movies | [`tests/test_data_windows.py`](../tests/test_data_windows.py), `test_fold_check_rejects_a_leak` |
| An unresolved outcome counts as a negative | A forecast whose follow-up ends inside the horizon is labelled `CENSORED_IN_HORIZON`, not `NO_EVENT` | [`tests/test_data_windows.py`](../tests/test_data_windows.py), `test_labels_and_landmarks` |
| Training reads past the cut | No training forecast is taught "no event" in a bin that ends after its movie's last observed frame, and `assert_inside_cut` raises an error when one is | [`tests/test_survival_head.py`](../tests/test_survival_head.py), `test_no_training_landmark_is_at_risk_past_the_cut` |
| A threshold head sees the held-out group | `train-head` exits with code 2 when the inner group is the held-out group, and `inner_split` refuses that group, training rows that contain it, and a missing training group | [`tests/test_survival_head.py`](../tests/test_survival_head.py), `test_inner_group_refuses_the_held_out_group` |
| A threshold reads held-out scores | The reference scores that set the image baseline's threshold for a held-out group are exactly the reference rows of its eight training groups | [`tests/test_image_warnings_reproduce.py`](../tests/test_image_warnings_reproduce.py), `test_image_reference_scores_pool_the_training_groups_reference_rows` |
| A fixed-horizon threshold reads held-out scores | The reference scores that set DeadAhead's threshold for a held-out group at each horizon are exactly the reference rows of its eight training groups, read from the table of that horizon | [`tests/test_horizons.py`](../tests/test_horizons.py), `test_inner_reference_scores_read_the_table_of_their_horizon` |

## Closest work

| Work | Task | Input | Output | Split | Censoring | How DeadAhead differs |
|---|---|---|---|---|---|---|
| [Okaniwa et al., 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12582640/) | Fate after heat, C1498 cells | Bright-field, 16-frame clips ending 1, 5, 10 or 15 h after heating | Survival, death or unknown; no horizon | Leave-one-out over 4 experiments | None; undecided cells are the class "unknown" | A forecast every 30 min at four horizons; two routes; unresolved cells censored |
| [Bürger et al., 2026](https://doi.org/10.1371/journal.pcbi.1014813) | Mitosis or apoptosis | Three fluorescent reporters | Fate, reliable "up to 8 h before the event"; no fixed horizon | Random 0.8/0.1/0.1 by trajectory | None; the main model keeps only annotated fates, and a side experiment makes undecided cells the class "neither" | Reads quantitative phase; undecided cells censored, not a class; whole groups held out |
| [Vicar et al., 2020](https://doi.org/10.1038/s41598-020-58474-w) | Find the death and its route | Quantitative phase, whole track | Death time and route | One field per experiment held out; SVM scheme not stated | None | Past frames only; whole groups held out |

Okaniwa T, Kryukov K, Shiroguchi K. Finding differentially expressed genes between cell fates predicted by image-based deep
learning. *Biophys. Physicobiol.* 22, e220022 (2025). Bürger F et al. Understanding cell fate decisions with temporal
attention. *PLoS Comput. Biol.* 22, e1014813 (2026). Vicar T et al. The quantitative-phase dynamics of apoptosis and lytic
cell death. *Sci. Rep.* 10, 1566 (2020).

Our citation scan: of the 99 works citing the dataset paper or the mitosis-or-apoptosis paper in the table above,
found on 18 October 2026, 18 detect or classify death after the fact and none forecasts it ahead of time.

## Method in brief

**Data.** The public CellDeathDetect movies (Vicar et al., 2020; doi:10.5281/zenodo.2601562): quantitative phase imaging
(QPI, Q-PHASE, Nikon Plan 10/0.30) of PNT1A, DU145 and LNCaP cells under black phosphorus (bp, 400 µg/mL), doxorubicin
(do, 0.1 µM) or staurosporine (st, 0.5 µM), with 7 fields of view for each of the 9
line-by-treatment groups. The 63 movies have 439 to 863 frames at 3 min per
frame (22.0 to 43.2 h). Cell outlines and tracks are the dataset's curated ones, seeded from
Hoechst-stained nuclei; only cells present in the whole movie without division or merging were kept, and one expert
marked each death's time and morphotype (route) from caspase-3/7 and
propidium iodide dyes and morphology. Routes are expert morphotypes. Our models read phase pixels and the shapes of these outlines, so results are conditional on this
curated cohort. The unit of analysis is the group (n = 9), not the movies or the cells: the fields of one group
have identical frame counts and may share an acquisition run.

One movie whose curated outlines failed a visual check of 9 sampled movies, blind to model output, was left
out of all labels and scores before any model was scored, so scores cover 62 of the 63 movies (the
left-out movie: DU145 st, field 2, 1 of 10 sampled tracks usable; its clips stay in the
unlabelled adaptation, see [`docs/DATA.md`](DATA.md)).

**Data table.** Cells by outcome, with their count at the cut (see Task below; recomputed by
`deadahead reproduce --part data`).

| Cells | All 63 movies | 62 scored movies |
|---|:---:|:---:|
| Annotated | 2,155 | 2,122 |
| Apoptosis (`APOPTOSIS` in `events.parquet`) | 635 | 608 |
| Lytic death (`LYTIC`) | 1,117 | 1,112 |
| Alive when their movie ends (`CENSORED`) | 378 | 378 |
| Death without a typed route (`UNTYPED_DEATH`) | 12 | 11 |
| Typed deaths dated after their movie ends, left out (`EXCLUDED`) | 13 | 13 |
| Apoptosis before the cut | 531 | 505 |
| Lytic death before the cut | 1,032 | 1,027 |
| Censored at the cut | 567 | 566 |
| Untyped deaths (before the cut) | 12 (6) | 11 (5) |

Of 2,155 annotated cells, 646 die by apoptosis and 1,119 by lytic death. These two totals count the
13 typed deaths dated after their movie ends (11 apoptotic, 2 lytic), which are left out,
so `events.parquet` holds 635 APOPTOSIS, 1,117 LYTIC and 13 EXCLUDED. Counts follow the
released annotations; Vicar et al. (2020) report 1,099 lytic deaths.

**Task.** A forecast is made every 30 min along each track once 120 min of history exist, while
the cell is alive, using only frames from the past 120 min, with one fixed intensity map and no smoothing. For
each route (apoptosis, lytic death) the model gives the probability that the cell dies by that route within h = 30, 60,
120 or 240 min. For a route, a death by it within the horizon is a positive; a death by the other route, or event-free
follow-up through the horizon, is a negative; a forecast whose follow-up ends inside the horizon with no death is never
a negative: it is left out of that horizon's AUPRC and precision, while IPA and calibration offset it through censoring
weights, as pre-specified. Forecast times and training labels end at the cut, after the first 439 frames (the shortest group's length, about
22.0 h; only Fine-tuned DINOv2 took labels past it); in training and in Table 1, deaths after the cut are censored
there. An untyped death ends follow-up at its death time (training: see Model). Every evaluation, for every method and
metric, uses one evaluation cohort that leaves out the 11 cells with an untyped death in the
62 scored movies (an outcome-based exclusion).

**Forecast table.** Forecasts at h = 30 min on the evaluation cohort. Count and pooled share: `--part data`; the fewest
to the most forecasts in one held-out group: `--part table1` (`P` of each unit).

| Forecasts at h = 30 min | Count | Pooled share | Fewest to most in one group |
|---|:---:|:---:|:---:|
| Evaluation cohort | 51,457 | | |
| Followed by lytic death within 30 min | 1,020 | 1.98% | 53 to 195 |
| Followed by apoptosis within 30 min | 504 | 0.98% | 11 to 131 |

**Model.** VideoMAE-L (Tong et al., 2022; ViT-L, pretrained without labels on Kinetics-400; licence CC BY-NC 4.0) reads 16
frames spread over the past 120 min as 128-pixel crops centred on the cell's curated outline in each frame,
resized to 224 pixels. In each fold it is first adapted with its masked-autoencoder objective on unlabelled clips of the
eight training groups, censored cells included, then fine-tuned end to end as a three-way classifier (no event, apoptosis,
lytic) at 30 min, with class weights; one of four learning-rate and layer-decay settings is chosen by lytic AUPRC on an
inner training group, and the winner is retrained on all eight training groups with three seeds (6,000 steps each). Its
pooled features, together with 112 handcrafted QPI features computed automatically inside the cell's outline
(14 shape and phase measures, each summarised by 8 statistics over the past 120 min),
feed a discrete-time competing-risks survival head: each forecast is expanded into eight 30-min bins (to 240 min), a
multiclass LightGBM with the bin index as an input gives the hazard of each route per bin, and the cumulative incidence
follows. A censored cell contributes the 30-min bins it was seen to survive in full, then leaves the risk set. This model
is DeadAhead.

As in every model of Table 1, its likelihood weights rows so that each cell, then each movie, counts equally, favouring
early deaths: a row weighs 1 / (forecasts of its cell), then the weights are normalised per movie. Over the 30-min rows with
a determined label, this raises the share of lytic rows from 1.97% to 4.72%
([`docs/figures/data/method_row_weights.csv`](figures/data/method_row_weights.csv)). At an untyped death the survival
head ends follow-up and does not count later time as survival, a pragmatic treatment of an unrecorded route; we do not
claim that ordinary censoring alone corrects for it.

**Baselines.** The first four, with no video model, are the image baselines; no image encoder saw the held-out group.
Their training code is in `deadahead/baselines/`, and their held-out predictions are in the predictions bundle, so every
comparison can be recomputed without retraining.

| Baseline | Inputs | Head |
|---|---|---|
| Handcrafted | the 112 features (no cell dynamic score; Vicar et al., 2020) | LightGBM |
| DINOv2 | ViT-B/14 adapted per fold without labels on random image tiles of the training groups; 9 single-frame embeddings over the past 120 min, pooled as last, mean and change | LightGBM |
| DINOv2 + handcrafted | DINOv2 and the 112 features | LightGBM; also DeadAhead's survival head where a table says "same head" |
| Fine-tuned DINOv2 + handcrafted | DINOv2 fine-tuned on the death labels (last two blocks and final norm, one fixed recipe) and the 112 features | LightGBM |
| DeepHit | DeadAhead's inputs, one of eight settings chosen on an inner training group | DeepHit |
| Elapsed time (control) | the forecast's time in the movie | a logistic fit in Table 1; later forecasts rank higher in the warnings |

**Evaluation.** Leave-one-group-out over the 9 groups. The main measure is macro AUPRC (area under the
precision-recall curve) at 30 min, the mean over held-out groups, for each route. Metrics are averaged over three seeds per
fold (video-model and survival-head seeds paired; the image encoders are adapted once per fold), then over the groups. We
report the paired macro difference (the lead) and the groups ahead, without p-values: leave-one-group-out folds share
training data, so per-group results are not independent. A lead meets our decision rule when it is at least 0.02
and has the same sign in at least 7 of 9 groups, a descriptive rule fixed before scoring, not a test.

An alarm fires when the predicted 30-min risk exceeds a threshold chosen on the training groups only. In each outer fold,
survival heads cross-fitted over the eight training groups (LightGBM heads for DINOv2 + handcrafted) score the training
forecasts; on these pooled scores the threshold is set for a target reference alarm rate of 0.2 (primary) or
0.4 alarms per event-free reference cell-hour, then applied once to the held-out group, where we report the
pooled rate achieved. Reference forecasts have 240 min of event-free follow-up and count as 30 min of exposure each; every
alarm counts.
A death is covered over its final X min when the alarm is on at every forecast in that interval, among deaths tracked that long; as forecasts are 30 min apart, the first of these alarms came X-30 to X min before death. The median coverage takes each death's largest such X, 0 if none.

Table 1 labels forecasts up to the cut; the survival metrics, the fixed-horizon results and the warnings use each movie's
complete follow-up, and a sensitivity analysis keeps only forecasts with complete follow-up inside the cut
([S6](SUPPLEMENT.md#s6-gap-analysis-sensitivity-analyses-censoring-and-label-audit)). The main analyses follow a dated protocol written before they ran. A dated amendment, written after
those results were seen, added the threshold, outcome-handling and sensitivity analyses: it replaced alarm thresholds that
had seen the test groups with training-only ones. The gap analyses, the within-hour scoring, the warnings of DINOv2 +
handcrafted and the matched alarm rate, whose target was chosen on the held-out groups, are post hoc.
