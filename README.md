# DeadAhead: Forecasting when and how a cell will die from quantitative phase video

**From quantitative phase video of held-out groups, DeadAhead forecasts when and how a cell will die: it ranks lytic deaths
at 13 times chance (macro AUPRC 0.322 against 0.024) and warns 66.2% of them 60 min ahead, with alarm
thresholds set on training groups only.**

Code, results and pretrained models for the paper *DeadAhead: Forecasting when and how a cell will die from quantitative phase video*
(M. Koochali, N. Khalid, A. Dengel, S. Ahmed; under review).

Forecasting when and how a cell will die would let biologists study cells while still alive, to learn why some cancer
cells survive a drug that kills their neighbours. Existing forecasts say whether a cell will die, not when. Every 30
min, DeadAhead, a fine-tuned VideoMAE with handcrafted features and a competing-risks survival head, forecasts each
cell's death by apoptosis or lytic death within 30 to 240 min from its past two hours of quantitative phase
video, keeping cells whose fate was never seen. On held-out line-by-treatment groups of three prostate cell lines, it
reached a 30-min lytic macro AUPRC of 0.322, thirteen times chance, against 0.254 for the best image
baseline, DINOv2 + handcrafted (ahead in 8 of 9 groups). With training-only thresholds,
DeadAhead's alarm was on at every forecast of the final hour before 66.2% of lytic deaths, far more than
an elapsed-time control. Both routes stayed far above chance out to 240 min.

![The three parts of DeadAhead, three example cells and their alarm timelines](docs/figures/fig1_overview.png)

**Figure 1.** DeadAhead. (a) Every 30 min, VideoMAE (fine-tuned) features of the past 2 h and handcrafted features
feed the survival head, giving the probability of lytic death, apoptosis or survival within h = 30 to 240 min.
(b) Input crops (80.5 µm wide) of three held-out LNCaP do cells: orange, warned early; blue, the group's
median warning; green, alive with no alarm. (c) Each dot is one
forecast, filled when its 30-min lytic risk exceeded the training-only threshold; the alive cell's time runs to its last
forecast. Cells LNCaP_do_4:49, LNCaP_do_3:10 and LNCaP_do_2:8, seed 2601562; the orange cell's alarm stayed on through its
last 270 min, which counts as warned 240 min ahead on the X grid of [Figure 2](#figure-2). Images:
CellDeathDetect (Vicar et al., 2020), CC BY 4.0.

## At a glance

| Method | Lytic AUPRC ([Table 1](#table-1)) | Apoptosis AUPRC ([Table 1](#table-1)) | Lytic deaths warned 60 min ahead ([Figure 2](#figure-2)) | Details |
|---|:---:|:---:|:---:|---|
| **DeadAhead (ours)** | **0.322** | **0.213** | **66.2%** (achieved rate 0.24) | [S1](#s1-fixed-horizon-forecasts), [S3](#s3-warnings-in-detail) |
| DeadAhead, target 0.26, at a matched alarm rate (post hoc) | | | **72.8%** (achieved rate 0.309) against 63.5% for DINOv2 + handcrafted (achieved rate 0.307); ahead in 7 of 9 groups | [S8](#s8-post-hoc-analyses) |
| VideoMAE (fine-tuned; ours) | 0.318 | 0.190 | | [S4](#s4-fine-tuning-and-test-time-changes), [S5](#s5-videomae-alone-the-same-head-comparison-and-elapsed-time) |
| DeepHit | 0.315 | 0.187 | | [S5](#s5-videomae-alone-the-same-head-comparison-and-elapsed-time) |
| DINOv2 + handcrafted | 0.254 | 0.180 | 63.5% (achieved rate 0.31; post hoc) | [S2](#s2-per-group-differences), [S8](#s8-post-hoc-analyses) |
| Fine-tuned DINOv2 + handcrafted | 0.217 | 0.130 | | [S5](#s5-videomae-alone-the-same-head-comparison-and-elapsed-time) |
| Handcrafted | 0.216 | 0.130 | | [S6](#s6-gap-analysis-sensitivity-analyses-censoring-and-label-audit) |
| DINOv2 | 0.159 | 0.086 | | [S5](#s5-videomae-alone-the-same-head-comparison-and-elapsed-time) |
| Elapsed time (control) | 0.020 | 0.068 | 10.5% (achieved rate 0.21) | [S5](#s5-videomae-alone-the-same-head-comparison-and-elapsed-time), [S6](#s6-gap-analysis-sensitivity-analyses-censoring-and-label-audit) |
| Chance | 0.024 | 0.012 | | [Table 1](#table-1) |

AUPRC is the area under the precision-recall curve at 30 min, as a mean over the 9 held-out groups. Warnings
use training-only thresholds at a target of 0.2 alarms per event-free reference cell-hour; the achieved
rates are pooled over the held-out groups, and the shares are means over the groups. Warnings were scored for the methods
of [Figure 2](#figure-2). Recomputed by `deadahead reproduce --part table1` (AUPRC), `--part warnings` and
`--part image-warnings` (warnings), and `--part alarm-rate` (the matched row).

## Check it in 5 minutes

Recompute Table 1 on a CPU, with no GPU and no raw movies (Python 3.10 or newer; on macOS, `shasum -a 256 -c -` replaces
`sha256sum -c -`):

```bash
git clone https://github.com/Mahdi-Koochali/DeadAhead.git && cd DeadAhead
python3 -m venv .venv && . .venv/bin/activate
pip install -e .
export DEADAHEAD_DATA="$PWD/data"
curl -L -o data/deadahead_bundle_v1.zip "https://zenodo.org/records/23883611/files/deadahead_bundle_v1.zip?download=1"
echo "e99a7a7a6afa290fd6e0c7c0bd9421a988793d7b29c84bf558418e3c69f65060  data/deadahead_bundle_v1.zip" | sha256sum -c -
unzip -q data/deadahead_bundle_v1.zip -d data
deadahead reproduce --part table1
```

The last command prints Table 1 in about a minute. Compare these rows (`dinov2_handcrafted|refit` is the paper's DINOv2 +
handcrafted; `(8)`: the eight groups other than PNT1A bp):

```text
method                                       lytic  apoptosis  lytic (8)  apopt. (8)
chance (prevalence)                          0.024      0.012      0.023       0.008
deadahead                                    0.322      0.213      0.324       0.195
dinov2_handcrafted|refit                     0.254      0.180      0.268       0.155
```

and, further down, DeadAhead's lead over that baseline:

```text
  dinov2_handcrafted|refit                LYTIC        +0.068   8/9  DeadAhead ahead (meets the decision rule)
```

The [claim table](#which-command-checks-which-claim) maps each claim of the abstract, the post hoc matched and within-hour results, Table 1, Table 2 and Figure 2 to one command and the lines of output to read.

**Tests only, no download:**

```bash
pip install -e ".[dev]"
pytest tests/test_cli.py tests/test_metrics.py tests/test_paths.py tests/test_packaging.py tests/test_row_weights.py tests/test_horizons_metrics.py tests/test_table1.py
```

This gives `60 passed, 2 skipped` in about 35 s without PyTorch, and includes a reproduce run on a synthetic bundle. The whole suite, `pytest -m "not data"`,
takes about 9 minutes.

**Supplement cited by the paper:** [S1 fixed horizons](#s1-fixed-horizon-forecasts), [S2 per group](#s2-per-group-differences),
[S3 warnings](#s3-warnings-in-detail), [S4 fine-tuning and test-time changes](#s4-fine-tuning-and-test-time-changes),
[S5 VideoMAE alone, same head, elapsed time](#s5-videomae-alone-the-same-head-comparison-and-elapsed-time),
[S6 gap, sensitivity, censoring, labels](#s6-gap-analysis-sensitivity-analyses-censoring-and-label-audit),
[S7 yeast probe](#s7-exploratory-probe-yeast-death), [S8 post hoc analyses](#s8-post-hoc-analyses).

## Contents

- [At a glance](#at-a-glance)
- [Check it in 5 minutes](#check-it-in-5-minutes)
- [Main results](#main-results): [Table 1](#table-1), [Figure 2](#figure-2), [Table 2](#table-2), [within each hour of imaging](#within-each-hour-of-imaging-post-hoc)
- [Reproducing the paper's numbers](#reproducing-the-papers-numbers), with the [claim table](#which-command-checks-which-claim)
- [Installation](#installation)
- [What protects the result](#what-protects-the-result)
- [Closest work](#closest-work)
- [Method in brief](#method-in-brief)
- [Supplementary results](#supplementary-results)
  - [S1. Fixed-horizon forecasts](#s1-fixed-horizon-forecasts)
  - [S2. Per-group differences](#s2-per-group-differences)
  - [S3. Warnings in detail](#s3-warnings-in-detail)
  - [S4. Fine-tuning and test-time changes](#s4-fine-tuning-and-test-time-changes)
  - [S5. VideoMAE alone, the same-head comparison and elapsed time](#s5-videomae-alone-the-same-head-comparison-and-elapsed-time)
  - [S6. Gap analysis, sensitivity analyses, censoring and label audit](#s6-gap-analysis-sensitivity-analyses-censoring-and-label-audit)
  - [S7. Exploratory probe: yeast death](#s7-exploratory-probe-yeast-death)
  - [S8. Post hoc analyses](#s8-post-hoc-analyses)
- [Pretrained models](#pretrained-models), with the [model card](docs/MODEL_CARD.md)
- [Data](#data), with the [data card](docs/DATA.md)
- [Training from scratch](#training-from-scratch)
- [Repository layout](#repository-layout)
- [Licences](#licences)
- [Citation](#citation)

## Main results

### Table 1

*Recomputed by `deadahead reproduce --part table1`.*

**Table 1.** Macro AUPRC at 30 min over the 9 held-out groups, with DeadAhead's lead and, in brackets, the groups
ahead (\*: meets our decision rule). Chance: the positive rate averaged over groups ([Method in brief](#method-in-brief)
gives it pooled). Comparator heads are refit without the untyped-death windows (for Fine-tuned DINOv2 + handcrafted, only the head).

| Model | Lytic | Lead (groups ahead) | Apoptosis | Lead (groups ahead) |
|---|:---:|:---:|:---:|:---:|
| **DeadAhead (ours)** | **0.322** | | **0.213** | |
| VideoMAE (fine-tuned) | 0.318 | +0.004 (5) | 0.190 | +0.023 (6) |
| DeepHit | 0.315 | +0.007 (5) | 0.187 | +0.026 (7)\* |
| DINOv2 + handcrafted | 0.254 | +0.068 (8)\* | 0.180 | +0.033 (5) |
| Fine-tuned DINOv2 + handcrafted | 0.217 | +0.106 (9)\* | 0.130 | +0.083 (8)\* |
| Handcrafted | 0.216 | +0.107 (7)\* | 0.130 | +0.082 (8)\* |
| DINOv2 | 0.159 | +0.163 (9)\* | 0.086 | +0.127 (9)\* |
| Elapsed time | 0.020 | +0.302 (9)\* | 0.068 | +0.145 (7)\* |
| Chance | 0.024 | | 0.012 | |

![Macro AUPRC by method for both routes, with one dot per held-out group and chance as a vertical line](docs/figures/figT1_auprc_by_method.png)

**Figure T1: Table 1 at a glance.** The same values with the spread: each row is a model, each open dot one of the 9
held-out groups, the bar the mean over the groups, printed at the right; the grey line is chance. The values are in
[`docs/figures/data/figT1_auprc_by_method.csv`](docs/figures/data/figT1_auprc_by_method.csv).

For lytic death the lead over DINOv2 + handcrafted ran from −0.059 to +0.172 across groups
(+0.055, 7 of 8, without PNT1A bp; per group in
[S2](#s2-per-group-differences)). Elapsed time stayed near chance for lytic death (0.020 against
0.024; [S5](#s5-videomae-alone-the-same-head-comparison-and-elapsed-time)).

### Figure 2

*Recomputed by `deadahead reproduce --part warnings` (DeadAhead, elapsed time), `--part image-warnings` (DINOv2 +
handcrafted, post hoc) and `--part alarm-rate` (the second row, post hoc).*

![Share of deaths warned X min ahead, with the interquartile range over the groups](docs/figures/fig2_warnings.png)

**Figure 2.** Share of deaths warned X min ahead, means over 9 held-out groups, among deaths with a forecast in
every 30-min bin up to X; bands: interquartile range over the groups. Training-only thresholds at a target reference
alarm rate of 0.2; achieved (pooled): DeadAhead 0.24 and 0.24, DINOv2 + handcrafted (post hoc)
0.31 and 0.22, elapsed time 0.21 and 0.21 (lytic, apoptosis). The values
are in [`docs/figures/data/fig2_warnings_spread.csv`](docs/figures/data/fig2_warnings_spread.csv) (means, ranges and
quartiles; columns `survival`: DeadAhead, `image`: DINOv2 + handcrafted, `clock`: elapsed time) and in
[S3](#s3-warnings-in-detail).

![Deaths warned 60 min ahead against the achieved alarm rate, the second row of Figure 2 (post hoc)](docs/figures/figS8b_alarm_rate.png)

**Figure 2, second row (post hoc).** Deaths warned 60 min ahead (mean over the 9 held-out groups) against the achieved
pooled alarm rate, for DeadAhead, DINOv2 + handcrafted with its own head and elapsed time. Each point is a training-only
threshold on a target grid fixed in advance of this analysis. Rings: the matched lytic comparison
(0.309 against 0.307 pooled; [S8](#s8-post-hoc-analyses)). The values are in
[`docs/figures/data/figS8b_alarm_rate.csv`](docs/figures/data/figS8b_alarm_rate.csv).

### Table 2

*Recomputed by `deadahead reproduce --part horizons`.*

**Table 2.** DeadAhead's lytic forecasts at each horizon h, macro over the 9 held-out groups: prevalence (%),
AUPRC, concordance (C), IPA, and precision (Prec.) and sensitivity (Sens.) at a training-only threshold on each horizon's
own risk (target 0.2, achieved 0.20 to 0.26, means over groups; reference forecasts
event-free through h). Complete-movie follow-up, whereas Table 1 labels up to the cut. Calibration slope and predicted over
observed incidence are in [S1](#s1-fixed-horizon-forecasts).

| h (min) | Prev. | AUPRC | C | IPA | Prec. | Sens. |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 30 | 2.4 | 0.321 | 0.900 | 0.031 | 0.141 | 0.777 |
| 60 | 4.8 | 0.380 | 0.862 | 0.051 | 0.224 | 0.648 |
| 120 | 9.6 | 0.402 | 0.821 | −0.002 | 0.339 | 0.533 |
| 240 | 18.9 | 0.482 | 0.784 | −0.082 | 0.489 | 0.429 |

### Within each hour of imaging (post hoc)

*Recomputed by `deadahead reproduce --part within-time`.*

Scoring on its own each hour of imaging that held a death, so that a forecast competed only with forecasts of that hour
and group, DeadAhead reached a macro within-hour AUPRC of 0.426 for lytic death and 0.325 for apoptosis (a random
order would reach 0.070 and 0.070; DINOv2 + handcrafted 0.353 and 0.299), a lead of +0.073 in
8 of 9 groups for lytic death, which meets our decision rule, and +0.027 in 5 of 9 for
apoptosis, which does not. These values are on their own scale, not Table 1's; the method, every model and the other band
widths are in [S8](#s8-post-hoc-analyses).

## Reproducing the paper's numbers

The results listed below can be recomputed on a CPU, without the raw movies or a GPU, from the released predictions
bundle (`deadahead_bundle_v1.zip`, 211 MB; CC BY 4.0; doi:10.5281/zenodo.23883611). It holds the forecast rows, the outcomes and every
method's held-out predictions (elapsed time is refit from the forecast rows, which takes seconds); its `MANIFEST.json` lists
every file with its sha256 and records the source data, the licence and the changes made:

```bash
export DEADAHEAD_DATA="${DEADAHEAD_DATA:-$PWD/data}"
curl -L -o "$DEADAHEAD_DATA/deadahead_bundle_v1.zip" "https://zenodo.org/records/23883611/files/deadahead_bundle_v1.zip?download=1"
echo "e99a7a7a6afa290fd6e0c7c0bd9421a988793d7b29c84bf558418e3c69f65060  $DEADAHEAD_DATA/deadahead_bundle_v1.zip" | sha256sum -c -
unzip -q "$DEADAHEAD_DATA/deadahead_bundle_v1.zip" -d "$DEADAHEAD_DATA"      # creates bundle/
deadahead reproduce --part all
```

`--part all` takes 15 to 34 min with three threads, depending on the load of the machine, and up to three times as long at full load. It needs about 2.2 GB of RAM.

### Which command checks which claim

Each row is one claim of the paper, one command and the line of its output to read; spaces in the lines are collapsed. Times are
wall times of real runs with three threads on the shared server named below, at moderate load; at full load they can take up to three times as long.

| Claim | Command | Line to read | Time |
|---|---|---|---|
| 30-min lytic macro AUPRC of 0.322 | `deadahead reproduce --part table1` | `deadahead 0.322 0.213`, column `lytic` | 50 to 64 s |
| 0.254 for the best image baseline | `deadahead reproduce --part table1` | `dinov2_handcrafted\|refit 0.254 0.180`, the paper's DINOv2 + handcrafted; the `\|original` row beside it is the fit of [Table S2c](#s2-per-group-differences) | 50 to 64 s |
| Ahead in 8 of 9 groups | `deadahead reproduce --part table1` | `dinov2_handcrafted\|refit LYTIC +0.068 8/9 DeadAhead ahead (meets the decision rule)` | 50 to 64 s |
| Lytic deaths at 13 times chance | `deadahead reproduce --part table1` | `chance (prevalence) 0.024 0.012` and `deadahead 0.322 0.213`, column `lytic` | 50 to 64 s |
| Apoptosis far above chance, 18 times | `deadahead reproduce --part table1` | `chance (prevalence) 0.024 0.012` and `deadahead 0.322 0.213`, column `apoptosis` | 50 to 64 s |
| Alarm on at every forecast of the final hour before 66.2% of lytic deaths | `deadahead reproduce --part warnings` | under `target 0.2, lytic`: `deadahead achieved 0.24 pooled, 0.29 mean`, then `60: 66.2` | 37 to 48 s |
| Far more than an elapsed-time control | `deadahead reproduce --part warnings` | under `target 0.2, lytic`: `elapsed_time achieved 0.21 pooled, 0.18 mean`, then `60: 10.5`; `DeadAhead minus elapsed time`, then `60:+55.7 9/9` | 37 to 48 s |
| Both routes far above chance out to 240 min | `deadahead reproduce --part horizons` | `lytic 240 deadahead 18.9 0.20 0.489 0.429 0.482 -0.082` and `apoptosis 240 deadahead 9.7 0.25 0.311 0.421 0.284 -0.267`: column `AUPRC` against column `Prev.`, the positive rate in %; every horizon is in [S1](#s1-fixed-horizon-forecasts) | 8 to 18 min |
| Post hoc, at a matched alarm rate: 72.8% against 63.5%, ahead in 7 of 9 groups | `deadahead reproduce --part alarm-rate` | `DeadAhead at target 0.26: achieved 0.309 pooled, 0.365 mean over groups, warned 72.8%`; `image baseline at target 0.2: achieved 0.307 pooled, 0.348 mean over groups, warned 63.5%`; `DeadAhead minus the image baseline: +9.3 points, ahead in 7 of 9 groups, DeadAhead ahead (meets the decision rule)` | 4 to 5 min |
| Post hoc, within each hour of imaging: within-hour AUPRC 0.426 lytic and 0.325 apoptosis, a random order 0.070 and 0.070, DINOv2 + handcrafted 0.353 and 0.299; leads +0.073 in 8 of 9 groups, which meets our decision rule, and +0.027 in 5 of 9, which does not | `deadahead reproduce --part within-time` | `Post hoc within-hour AUPRC, lytic: DeadAhead 0.426, DINOv2 + handcrafted 0.353, a random order 0.070; lead +0.073, ahead in 8 of 9 groups, DeadAhead ahead (meets the decision rule)`; `Post hoc within-hour AUPRC, apoptosis: DeadAhead 0.325, DINOv2 + handcrafted 0.299, a random order 0.070; lead +0.027, ahead in 5 of 9 groups, below the decision rule` | 75 to 81 s |
| Tested on held-out line-by-treatment groups | `python -c "import json; [print(f['test_group'], len(f['train_movies']), len(f['test_movies'])) for f in json.load(open('splits/folds.json'))['folds']]"` | nine lines, from `DU145_bp 56 7` to `PNT1A_st 56 7`: each group held out, its movies never in training | under 1 s |
| [Table 1](#table-1), every row | `deadahead reproduce --part table1` | the block under `Table 1: AUPRC at 30 min`. Names: `deadahead`, `videomae_finetuned`, `deephit`, `dinov2_handcrafted\|refit` (DINOv2 + handcrafted), `dinov2_finetuned_handcrafted\|refit` (Fine-tuned DINOv2 + handcrafted), `handcrafted\|refit`, `dinov2\|refit`, `elapsed_time` | 50 to 64 s |
| [Table 2](#table-2), h = 30 min | `deadahead reproduce --part horizons` | `lytic 30 deadahead 2.4 0.26 0.141 0.777 0.321 0.031` | 8 to 18 min |
| [Table 2](#table-2), h = 60 min | `deadahead reproduce --part horizons` | `lytic 60 deadahead 4.8 0.25 0.224 0.648 0.380 0.051` | 8 to 18 min |
| [Table 2](#table-2), h = 120 min | `deadahead reproduce --part horizons` | `lytic 120 deadahead 9.6 0.23 0.339 0.533 0.402 -0.002` | 8 to 18 min |
| [Table 2](#table-2), h = 240 min | `deadahead reproduce --part horizons` | `lytic 240 deadahead 18.9 0.20 0.489 0.429 0.482 -0.082` | 8 to 18 min |
| [Figure 2](#figure-2), the curves | `deadahead reproduce --part image-warnings` | the lines above for DeadAhead and elapsed time; `target 0.2, lytic achieved 0.31 pooled`, then `60: 63.5`, for DINOv2 + handcrafted (post hoc); the bands are in `fig2_warnings_spread.csv` | 69 to 77 s |
| [Figure 2](#figure-2), the image | that command, then `deadahead plot-warnings $DEADAHEAD_DATA/runs/reproduce/fig2_warnings_spread.csv docs/figures/fig2_warnings.png` | `git status docs/figures` shows no change with the versions of `environment/cpu-py312.txt` | 69 to 77 s, then a few seconds |
| [Figure 2](#figure-2), second row (post hoc) | `deadahead reproduce --part alarm-rate`, then `deadahead plot-alarm-rate $DEADAHEAD_DATA/runs/reproduce/figS8b_alarm_rate.csv docs/figures/figS8b_alarm_rate.png` | `git status docs/figures` shows no change with those versions | 4 to 5 min, then a few seconds |

`--part horizons` prints the columns `Prev.`, `rate`, `precision`, `sensitivity`, `AUPRC`, `IPA`, `IPA (train)`, `<0`, `C`, `slope`
and `P/O`; the lines above stop at `IPA`, and [Table 2](#table-2)'s `C` is further along the same line.

### What each part recomputes

| Part | What it recomputes | Time inside `--part all` |
|---|---|:---:|
| `data` | the counts of the [Data and Forecast tables](#method-in-brief): cells, deaths and forecasts over all 63 movies and the 62 scored movies, the cut, the positives, and the movie lengths (439 to 863 frames, 21.95 to 43.15 h) | under 1 s |
| `table1` | Table 1 and its leads, S2, the Table 1 numbers of S5, the fewest and most forecasts per group; also `table1.chance` (`nine_folds`, `eight_folds_without_PNT1A_bp`) and `table1.elapsed_time_fit` (per route, weighted and unweighted: `slope_per_group`, `groups_slope_below_zero`, `groups_slope_above_zero`, `auprc`) | 28 to 53 s |
| `warnings` | Figure 2 (DeadAhead, elapsed time) and S3; prints the mean rate, the row without PNT1A_bp, the eligible deaths and DeadAhead minus elapsed time | 25 to 49 s |
| `image-warnings` | Figure 2 (DINOv2 + handcrafted), that baseline's warning share, S3 (post hoc) | 36 to 100 s |
| `horizons` | Table 2, S1 (Slope is `ipcw_slope`, P/O is `e_over_o`), the same-head comparison and the alarm score of S5 (the comparison is also quoted in S6's gap analysis), the probability accuracy of S6; prints the prevalence, the rate, the groups below zero, Slope, P/O and a block of same-head contrasts | 8 to 17 min |
| `gap` | S6, gap analysis (post hoc); also prints the score by the whole 120-min risk and by the increment given survival to 60 min | 5 to 52 s |
| `sensitivity` | S6, complete follow-up, the follow-up row of S5, and the fixed horizons inside the cut quoted in S8; prints the macro AUPRC per method, the contrasts at the common landmark and the fixed-horizon tables at the cut | 5 to 8 min |
| `decomposition` | S4 | 10 to 11 s |
| `censoring` | S6, keeping censored cells | about 1 s |
| `gap-window` | S8, forecasting beyond the last minutes (post hoc) | 2 to 3 s |
| `alarm-rate` | S8, warnings against the alarm rate and the matched comparison (post hoc); prints the reverse match and the apoptosis match | about 4 min |
| `within-time` | S8, scoring within each hour of imaging (post hoc); prints every method at 1-hour bands, the lead over DINOv2 + handcrafted at each band width and the second design with its ceiling | about 23 s |

The 646 and 1,119 of the `data` part count 11 and 2 excluded deaths dated after their movie ends. Their routes come from the
label spreadsheets (`events.death_type_raw` from `deadahead data-tables`), not from the bundle: in the code they are the
constant `EXCLUDED_ROUTES` in `deadahead/reproduce/data.py`.

These times come from two runs of `--part all` with three threads on a shared 2.2 GHz Xeon E5-2698 v4 server, at different
loads. Most of the time goes to `horizons`, `sensitivity` and `alarm-rate`. A part run alone also loads and checks the bundle
first, and `image-warnings` also computes `warnings`.

The tables behind the figures go next to the JSON output (by default in `$DEADAHEAD_DATA/runs/reproduce/`):
`--part image-warnings` and `--part all` write `fig2_warnings.csv` and `fig2_warnings_spread.csv`, `--part gap-window`
writes `figS8a_gap_window.csv` and `--part alarm-rate` writes `figS8b_alarm_rate.csv`. One command then draws every figure
of this README except Figure 1, which is the paper's own figure:

```bash
deadahead plot-figures --out-dir docs/figures
```

It writes each figure to `docs/figures/` and its values to `docs/figures/data/`. Every figure comes from the JSON, except
`figS8c_alarm_share`, which comes from the bundle, as do `s8_group_match.csv` and `method_row_weights.csv` (`--bundle`, or
`--no-bundle` to skip the three). The command took 51 to 70 s on that server, and about 8 min under heavy load, most of it
reading the bundle. `plot-warnings`, `plot-gap-window` and `plot-alarm-rate` draw single
figures from their tables; they print an error and exit with code 3 on a missing or wrong table, and create the output
folder.


`reproduce` checks the bundle against its `MANIFEST.json`, writes every value as JSON and the figure tables, then prints
each table; the time of each part goes to stderr. The JSON of the post hoc parts carries `"post_hoc": true`, and `--help`
marks them. With numpy 2.x the results equal, bit for bit, the values behind the paper's numbers (S1's training-reference
IPA, recorded from a second implementation of the same estimator, agrees to within 1e-15 absolute); with numpy 1.26 they
agree to within about 1e-12 relative, and every printed value is the same. `configs/evaluation.yaml` holds the alarm and
horizon settings (`horizons_min`, `positivity_min_g`).

## Installation

| Part | Python | Packages |
|---|---|---|
| Base install: tables, features, survival head, evaluation, `deadahead reproduce`, `deadahead yeast-score` | 3.10 or newer | LightGBM 4.7.0 and the CPU packages of [`pyproject.toml`](pyproject.toml) |
| `[video]` extra: VideoMAE, the DINOv2 baselines, DeepHit | 3.10 to 3.13 (no torch 2.5.1 wheel for 3.14); tested fresh on 3.10 and 3.12 | torch 2.5.1, transformers 5.15.1, safetensors 0.8.0, huggingface_hub 1.28.0 |
| `[dev]` extra: tests and lint | 3.10 or newer | pytest and ruff 0.17.3 |

```bash
git clone https://github.com/Mahdi-Koochali/DeadAhead.git
cd DeadAhead
pip install -e .            # CPU: tables, features, survival head, evaluation, reproduction
pip install -e ".[video]"   # adds PyTorch and Transformers 5 (VideoMAE, the DINOv2 baselines, DeepHit)
pip install -e ".[dev]"     # adds pytest and ruff
```

Install in editable mode (`-e`): the settings in `configs/` are read from the checkout. Do not upgrade the `[video]` pins:
a newer Transformers needs torch 2.6 or later and turns PyTorch off. The exact versions of the paper's runs are listed in
`environment/`.

`ruff check` uses the settings in `pyproject.toml`. `pytest` runs the tests on synthetic data; when the yeast bundle is
installed and `DEADAHEAD_DATA` is set it also runs the S7 data test (about 12 minutes at one thread under load), and
`pytest -m "not data"` skips it.

## What protects the result

Leakage is what to look for in a forecast. Each line below is a test in this repository; none needs the data, and
together they take about 11 s (`pip install -e ".[dev]"` first):

```bash
pytest tests/test_history_features.py::test_no_future_frames tests/test_data_windows.py::test_fold_check_rejects_a_leak tests/test_data_windows.py::test_labels_and_landmarks tests/test_survival_head.py::test_no_training_landmark_is_at_risk_past_the_cut tests/test_survival_head.py::test_inner_group_refuses_the_held_out_group tests/test_image_warnings_reproduce.py::test_image_reference_scores_pool_the_training_groups_reference_rows tests/test_horizons.py::test_inner_reference_scores_read_the_table_of_their_horizon
```

| Risk | What the test proves | Test |
|---|---|---|
| Future frames reach the features | Changing or deleting every frame after a cut leaves the handcrafted features at or before that cut exactly as they were. This test covers the handcrafted features | [`tests/test_history_features.py`](tests/test_history_features.py), `test_no_future_frames` |
| A held-out movie enters training | `check_folds` raises an error when a fold's held-out movie is added to its training movies | [`tests/test_data_windows.py`](tests/test_data_windows.py), `test_fold_check_rejects_a_leak` |
| An unresolved outcome counts as a negative | A forecast whose follow-up ends inside the horizon is labelled `CENSORED_IN_HORIZON`, not `NO_EVENT` | [`tests/test_data_windows.py`](tests/test_data_windows.py), `test_labels_and_landmarks` |
| Training reads past the cut | No training forecast is taught "no event" in a bin that ends after its movie's last observed frame, and `assert_inside_cut` raises an error when one is | [`tests/test_survival_head.py`](tests/test_survival_head.py), `test_no_training_landmark_is_at_risk_past_the_cut` |
| A threshold head sees the held-out group | `train-head` exits with code 2 when the inner group is the held-out group, and `inner_split` refuses that group, training rows that contain it, and a missing training group | [`tests/test_survival_head.py`](tests/test_survival_head.py), `test_inner_group_refuses_the_held_out_group` |
| A threshold reads held-out scores | The reference scores that set the image baseline's threshold for a held-out group are exactly the reference rows of its eight training groups | [`tests/test_image_warnings_reproduce.py`](tests/test_image_warnings_reproduce.py), `test_image_reference_scores_pool_the_training_groups_reference_rows` |
| A fixed-horizon threshold reads held-out scores | The reference scores that set DeadAhead's threshold for a held-out group at each horizon are exactly the reference rows of its eight training groups, read from the table of that horizon | [`tests/test_horizons.py`](tests/test_horizons.py), `test_inner_reference_scores_read_the_table_of_their_horizon` |

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
unlabelled adaptation, see [`docs/DATA.md`](docs/DATA.md)).

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
([`docs/figures/data/method_row_weights.csv`](docs/figures/data/method_row_weights.csv)). At an untyped death the survival
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
| Fine-tuned DINOv2 + handcrafted | DINOv2 fine-tuned on the death labels (last two blocks, one fixed recipe) and the 112 features | LightGBM |
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
alarm counts. A death counts as warned X min ahead when the alarm is on at every forecast in its last X min, among deaths
tracked that long; as forecasts are 30 min apart, the first of these alarms came X - 30 to X min before death.

Table 1 labels forecasts up to the cut; the survival metrics, the fixed-horizon results and the warnings use each movie's
complete follow-up, and a sensitivity analysis restricts every method to forecasts with complete follow-up inside the cut
([S6](#s6-gap-analysis-sensitivity-analyses-censoring-and-label-audit)). The main analyses follow a dated protocol written before they ran. A dated amendment, written after
those results were seen, added the threshold, outcome-handling and sensitivity analyses: it replaced alarm thresholds that
had seen the test groups with training-only ones. The gap analyses, the within-hour scoring, the warnings of DINOv2 +
handcrafted and the matched alarm rate, whose target was chosen on the held-out groups, are post hoc.

## Supplementary results

These results come from the same runs as the paper. Each item has a fixed heading (S1 to S8), so that it can be cited,
and names the command that recomputes it. `deadahead plot-figures` draws the figures from the output of
`deadahead reproduce --part all` and the predictions bundle (see
[Reproducing the paper's numbers](#reproducing-the-papers-numbers)).

### S1. Fixed-horizon forecasts

*Recomputed by `deadahead reproduce --part horizons`.*

**Table S1a.** Fixed-horizon forecasts of DeadAhead, macro over all 9 held-out groups: prevalence (Prev., %),
AUPRC, time-dependent concordance (C; same-cell pairs excluded), IPA (index of prediction accuracy: 1 minus the model's
censoring-weighted Brier score over that of a constant forecast, the held-out group's Aalen-Johansen incidence, an
outcome-based benchmark that knows the group's death rate) and the number of groups with IPA below zero, calibration slope
(Slope), predicted over observed incidence (P/O), and at each horizon's own training-only threshold (target 0.2;
reference forecasts followed event-free through it) the achieved rate (alarms per event-free reference cell-hour, mean over
groups), precision (Prec.) and sensitivity (Sens.). Outcomes use complete-movie follow-up, so the 30-min AUPRC differs
slightly from Table 1's, which labels up to the cut. The reference forecasts differ from those of the warnings in S3 (240
min of event-free follow-up), so the 30-min achieved rate does too (0.26 here, 0.29 there, both
means over groups).

| Route | h (min) | Prev. | AUPRC | C | IPA | Groups with IPA < 0 | Slope | P/O | Achieved rate | Prec. | Sens. |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Lytic | 30 | 2.4 | 0.321 | 0.900 | 0.031 | 4 | 0.63 | 1.27 | 0.26 | 0.141 | 0.777 |
| Lytic | 60 | 4.8 | 0.380 | 0.862 | 0.051 | 3 | 0.65 | 1.40 | 0.25 | 0.224 | 0.648 |
| Lytic | 120 | 9.6 | 0.402 | 0.821 | −0.002 | 5 | 0.65 | 1.51 | 0.23 | 0.339 | 0.533 |
| Lytic | 240 | 18.9 | 0.482 | 0.784 | −0.082 | 5 | 0.63 | 1.62 | 0.20 | 0.489 | 0.429 |
| Apoptosis | 30 | 1.2 | 0.207 | 0.879 | −0.069 | 5 | 0.58 | 1.10 | 0.28 | 0.065 | 0.684 |
| Apoptosis | 60 | 2.3 | 0.207 | 0.850 | −0.095 | 5 | 0.57 | 1.33 | 0.27 | 0.111 | 0.598 |
| Apoptosis | 120 | 4.7 | 0.227 | 0.817 | −0.137 | 4 | 0.56 | 1.53 | 0.26 | 0.186 | 0.537 |
| Apoptosis | 240 | 9.7 | 0.284 | 0.775 | −0.267 | 5 | 0.54 | 1.86 | 0.25 | 0.311 | 0.421 |

![AUPRC, concordance and IPA of DeadAhead against the horizon for both routes, with chance and the zero line](docs/figures/figS1_horizons.png)

**Figure S1.** DeadAhead against the horizon h, both routes, macro over the 9 held-out groups: (a) AUPRC, with
its chance level, the prevalence, in grey; (b) concordance, with its axis starting at chance, 0.5; (c) IPA, with the zero
line of a constant forecast of the held-out group's own incidence. The values are in
[`docs/figures/data/figS1_horizons.csv`](docs/figures/data/figS1_horizons.csv).

Ranking is far above chance (AUPRC, concordance), but probability accuracy is a different question: DeadAhead's IPA is
negative except for lytic death up to 60 min, as its probabilities are too extreme and on average too high (slopes below
1, P/O above 1), and they need recalibration before use as risks. Table S1b gives the same two measures for all four
survival methods.

**Table S1b.** Calibration slope and predicted over observed incidence (P/O) of the survival methods, macro over the
9 held-out groups.

| Method | Route | Slope, 30 min | Slope, 60 min | Slope, 120 min | Slope, 240 min | P/O, 30 min | P/O, 60 min | P/O, 120 min | P/O, 240 min |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| DeadAhead | Lytic | 0.63 | 0.65 | 0.65 | 0.63 | 1.27 | 1.40 | 1.51 | 1.62 |
| DeadAhead | Apoptosis | 0.58 | 0.57 | 0.56 | 0.54 | 1.10 | 1.33 | 1.53 | 1.86 |
| DINOv2 + handcrafted, same head | Lytic | 1.01 | 0.96 | 0.88 | 0.81 | 1.11 | 1.26 | 1.43 | 1.64 |
| DINOv2 + handcrafted, same head | Apoptosis | 0.96 | 0.88 | 0.80 | 0.68 | 1.42 | 1.86 | 2.28 | 2.59 |
| Handcrafted, same head | Lytic | 1.02 | 0.99 | 0.88 | 0.71 | 1.64 | 1.80 | 2.01 | 2.14 |
| Handcrafted, same head | Apoptosis | 0.90 | 0.85 | 0.79 | 0.73 | 1.38 | 1.59 | 1.73 | 1.88 |
| DeepHit | Lytic | 0.56 | 0.61 | 0.60 | 0.61 | 1.89 | 2.11 | 1.94 | 1.72 |
| DeepHit | Apoptosis | 0.46 | 0.50 | 0.50 | 0.52 | 2.02 | 2.31 | 2.18 | 2.09 |

**Table S1c.** IPA of DeadAhead with the training groups' incidence as the reference (pre-specified, less stable; the
`IPA (train)` column of `--part horizons`).

| Route | 30 min | 60 min | 120 min | 240 min |
|---|:---:|:---:|:---:|:---:|
| Lytic | 0.040 | 0.067 | 0.030 | −0.012 |
| Apoptosis | −0.060 | −0.075 | −0.098 | −0.179 |

### S2. Per-group differences

*Recomputed by `deadahead reproduce --part table1`.*

**Table S2a.** DeadAhead minus DINOv2 + handcrafted (refit), AUPRC at 30 min per held-out group (mean over the three seeds).
Rows: cell line; columns: route and treatment.

| Line | Lytic, bp | Lytic, do | Lytic, st | Apoptosis, bp | Apoptosis, do | Apoptosis, st |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| PNT1A | +0.172 | +0.068 | +0.130 | −0.024 | +0.117 | +0.121 |
| DU145 | +0.117 | +0.083 | −0.059 | −0.032 | +0.060 | +0.002 |
| LNCaP | +0.049 | +0.017 | +0.037 | +0.083 | −0.005 | −0.026 |

![DeadAhead minus DINOv2 + handcrafted for each of the nine held-out groups, a stem from zero to each value, filled when DeadAhead is ahead](docs/figures/figS2_group_differences.png)

**Figure S2.** The values of Table S2a: a stem from zero to each group's difference, filled when DeadAhead is ahead and open
when it is behind; the grey line is the mean over the groups. Group macros count each group once, however few events it
has: positive forecasts per group run from 53 to 195 for lytic death and from 11 to 131
for apoptosis. The values are in [`docs/figures/data/figS2_group_differences.csv`](docs/figures/data/figS2_group_differences.csv).

Every main comparison is also computed without PNT1A bp, as a fold-influence check (see the training-label audit in
[S6](#s6-gap-analysis-sensitivity-analyses-censoring-and-label-audit)).

**Table S2b.** Macro AUPRC at 30 min, 9 groups and the 8 groups without PNT1A bp.

| Model | Lytic, 9 groups | Lytic, 8 groups | Apoptosis, 9 groups | Apoptosis, 8 groups |
|---|:---:|:---:|:---:|:---:|
| Chance | 0.024 | 0.023 | 0.012 | 0.008 |
| Elapsed time | 0.020 | 0.018 | 0.068 | 0.070 |
| **DeadAhead (ours)** | 0.322 | 0.324 | 0.213 | 0.195 |
| VideoMAE (fine-tuned; ours) | 0.318 | 0.315 | 0.190 | 0.176 |
| DeepHit | 0.315 | 0.316 | 0.187 | 0.173 |
| DINOv2 + handcrafted | 0.254 | 0.268 | 0.180 | 0.155 |
| Fine-tuned DINOv2 + handcrafted | 0.217 | 0.225 | 0.130 | 0.115 |
| Handcrafted | 0.216 | 0.222 | 0.130 | 0.111 |
| DINOv2 | 0.159 | 0.169 | 0.086 | 0.070 |

**Table S2c.** The comparators of Table 1 before the refit: the original fit on the evaluation cohort, against the refit
without the untyped-death windows.

| Model | Lytic, refit | Lytic, original | Apoptosis, refit | Apoptosis, original |
|---|:---:|:---:|:---:|:---:|
| DINOv2 + handcrafted | 0.254 | 0.253 | 0.180 | 0.174 |
| Fine-tuned DINOv2 + handcrafted | 0.217 | 0.218 | 0.130 | 0.128 |
| Handcrafted | 0.216 | 0.215 | 0.130 | 0.124 |
| DINOv2 | 0.159 | 0.157 | 0.086 | 0.084 |

### S3. Warnings in detail

*Recomputed by `deadahead reproduce --part warnings` and `--part image-warnings`.*

**Table S3a.** The warnings at both targets. Achieved: alarms per event-free reference cell-hour, pooled and as the mean
over groups. Warned: share of deaths warned X min ahead (%), mean over groups. Median warning: the median of each death's
largest such X (X in 30, 60, 90, 120, 180 and 240 min; 0 if none) among deaths with a forecast in every 30-min bin of their
last 4 h, then the mean over groups. The DINOv2 + handcrafted rows are post hoc, and [S8](#s8-post-hoc-analyses)
compares the methods at a matched achieved rate (0.309 against 0.307 pooled; post hoc).

| Route | Target | Method | Achieved rate, pooled | Achieved rate, mean over groups | Warned 60 min ahead (%) | Warned 120 min ahead (%) | Warned 240 min ahead (%) | Median warning (min) |
|---|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Lytic | 0.2 | DeadAhead | 0.24 | 0.29 | 66.2 | 38.7 | 13.9 | 89 |
| Lytic | 0.2 | Elapsed time | 0.21 | 0.18 | 10.5 | 8.6 | 2.9 | 3 |
| Lytic | 0.2 | DINOv2 + handcrafted (post hoc) | 0.31 | 0.35 | 63.5 | 38.1 | 16.8 | 89 |
| Lytic | 0.4 | DeadAhead | 0.47 | 0.54 | 83.0 | 62.5 | 34.1 | 151 |
| Lytic | 0.4 | Elapsed time | 0.41 | 0.36 | 21.2 | 17.6 | 9.9 | 20 |
| Lytic | 0.4 | DINOv2 + handcrafted (post hoc) | 0.56 | 0.62 | 77.9 | 55.2 | 31.7 | 143 |
| Apoptosis | 0.2 | DeadAhead | 0.24 | 0.33 | 62.1 | 44.6 | 25.1 | 100 |
| Apoptosis | 0.2 | Elapsed time | 0.21 | 0.18 | 37.8 | 30.1 | 17.0 | 62 |
| Apoptosis | 0.2 | DINOv2 + handcrafted (post hoc) | 0.22 | 0.25 | 49.9 | 28.0 | 11.0 | 54 |
| Apoptosis | 0.4 | DeadAhead | 0.44 | 0.53 | 77.6 | 63.0 | 44.2 | 172 |
| Apoptosis | 0.4 | Elapsed time | 0.41 | 0.36 | 53.2 | 50.1 | 34.7 | 133 |
| Apoptosis | 0.4 | DINOv2 + handcrafted (post hoc) | 0.42 | 0.46 | 68.2 | 48.6 | 27.9 | 115 |

**The example cells of Figure 1.** Examples illustrate the method and are not evidence. The orange cell was drawn from
the 2 deaths warned this early, of the group's 49 tracked long enough; the green cell from 31 such cells of the 60 alive at
the end and tracked long enough.

Without PNT1A bp, DeadAhead warned 60 min ahead for 66.8% of lytic deaths.

**The alarm burden.** At DeadAhead's primary achieved lytic rate (0.24), 12.2% of event-free
reference forecasts raise an alarm (reference time only, not the burden over all monitored cells). Both numbers are pooled
over the groups; each reference forecast counts 0.5 cell-hour, so the share is half the rate. As means over the groups, the
rate is 0.29 and the share 14.5%, the grey line of [Figure S8c](#s8-post-hoc-analyses). Achieved rates
differ from the target and across groups (lytic, at 0.2: 0.02 to 0.58), as the alarm
thresholds come from heads trained on seven groups and scored on the eighth, and the encoder had seen them all. Even at a
higher achieved rate (0.41 against 0.24), elapsed time warned 60 min ahead for fewer lytic
deaths (21.2% against 66.2%).

**Table S3b.** DeadAhead's lead in share of deaths warned X min ahead at the primary target, in percentage points, with the
groups in which it is ahead. Mean over groups; the DINOv2 + handcrafted column is post hoc, and its two shares are at
different achieved rates.

| Route | X (min) | DeadAhead minus elapsed time | DeadAhead minus elapsed time, without PNT1A bp | DeadAhead minus DINOv2 + handcrafted (post hoc) |
|---|---|:---:|:---:|:---:|
| Lytic | 60 | +55.7 (9 of 9) | +55.0 (8 of 8) | +2.7 (5 of 9) |
| Lytic | 120 | +30.1 (8 of 9) | +27.4 (7 of 8) | +0.6 (6 of 9) |
| Lytic | 240 | +11.0 (7 of 9) | +9.9 (6 of 8) | −3.0 (5 of 9) |
| Apoptosis | 60 | +24.3 (6 of 9) | +20.5 (5 of 8) | +12.2 (6 of 9) |
| Apoptosis | 120 | +14.6 (5 of 9) | +12.1 (4 of 8) | +16.6 (8 of 9) |
| Apoptosis | 240 | +8.0 (5 of 9) | +7.6 (4 of 8) | +14.1 (7 of 9) |

**Table S3c.** Eligible deaths: deaths with a forecast in every 30-min bin up to X, summed over the groups.

| Route | X = 30 min | X = 60 min | X = 90 min | X = 120 min | X = 180 min | X = 240 min |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Lytic | 1,020 | 1,011 | 1,004 | 988 | 952 | 905 |
| Apoptosis | 504 | 500 | 498 | 498 | 491 | 478 |

**Table S3d.** Share of deaths warned X min ahead (%), means over groups. Column headers: method, route, target alarm rate.
The DINOv2 + handcrafted columns are post hoc. The values behind Figure 2.

| X (min) | DeadAhead, lytic, 0.2 | DeadAhead, lytic, 0.4 | DeadAhead, apoptosis, 0.2 | DeadAhead, apoptosis, 0.4 | DINOv2 + handcrafted, lytic, 0.2 | DINOv2 + handcrafted, apoptosis, 0.2 | Elapsed time, lytic, 0.2 | Elapsed time, lytic, 0.4 | Elapsed time, apoptosis, 0.2 | Elapsed time, apoptosis, 0.4 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 30 | 85.7 | 93.7 | 77.1 | 89.4 | 84.3 | 71.2 | 13.0 | 22.7 | 41.1 | 54.9 |
| 60 | 66.2 | 83.0 | 62.1 | 77.6 | 63.5 | 49.9 | 10.5 | 21.2 | 37.8 | 53.2 |
| 90 | 49.7 | 71.0 | 52.7 | 70.3 | 46.5 | 35.7 | 9.3 | 19.5 | 32.5 | 51.9 |
| 120 | 38.7 | 62.5 | 44.6 | 63.0 | 38.1 | 28.0 | 8.6 | 17.6 | 30.1 | 50.1 |
| 180 | 25.3 | 48.7 | 33.6 | 51.4 | 25.1 | 17.2 | 5.4 | 14.1 | 23.6 | 43.1 |
| 240 | 13.9 | 34.1 | 25.1 | 44.2 | 16.8 | 11.0 | 2.9 | 9.9 | 17.0 | 34.7 |

### S4. Fine-tuning and test-time changes

*Recomputed by `deadahead reproduce --part decomposition`.*

**Table S4a.** With the same survival head and handcrafted features: VideoMAE features after fine-tuning (DeadAhead) minus
the same encoder's features before fine-tuning.

| Route | Macro AUPRC difference | Groups ahead |
|---|:---:|:---:|
| Lytic | +0.064 | 7 of 9 |
| Apoptosis | +0.084 | 9 of 9 |

The two also differ in read-out layer and input tokens (adaptation saw only the unmasked tokens of a clip, the features
use all of them). The head trains on rows the encoder was fine-tuned on and can over-trust them, an uncorrected bias
against DeadAhead that is not separated from fine-tuning.

**Table S4b.** Macro AUPRC of VideoMAE (fine-tuned) at test time: hiding the tracked cell, shuffling the 16 frames or
repeating the last frame 16 times.

| Route | Intact | Cell hidden | Frames shuffled | Last frame repeated |
|---|:---:|:---:|:---:|:---:|
| Lytic | 0.318 | 0.079 | 0.110 | 0.061 |
| Apoptosis | 0.190 | 0.039 | 0.063 | 0.047 |

![Macro AUPRC of VideoMAE (fine-tuned) intact and under the three test-time changes, with one dot per held-out group](docs/figures/figS4_test_time.png)

**Figure S4.** The values of Table S4b, with one dot per held-out group; the bar is the mean over the groups, printed at
the right, and the grey line is chance. Each change lowered the AUPRC in all 9 groups. Performance was thus
sensitive to removing the cell's content and to disrupting the clip's temporal structure; these changes do not isolate
the source of the gain over the image baselines. The values are in
[`docs/figures/data/figS4_test_time.csv`](docs/figures/data/figS4_test_time.csv).

To rebuild these inputs from the movies, see the rows marked S4 in [Training from scratch](#training-from-scratch).

### S5. VideoMAE alone, the same-head comparison and elapsed time

*Recomputed by `deadahead reproduce --part table1` (the Table 1 numbers), `--part horizons` (the same-head comparison and
the alarm score) and `--part sensitivity` (follow-up inside the cut).*

**Table S5a.** The leads behind the paper's statements on VideoMAE: macro AUPRC difference at 30 min (groups ahead of
9), and whether each lead meets our decision rule.

| Comparison | Lytic | Meets our rule | Apoptosis | Meets our rule |
|---|:---:|:---:|:---:|:---:|
| VideoMAE (fine-tuned) minus DINOv2 + handcrafted | +0.065 (8 of 9) | yes | +0.010 (5 of 9) | no |
| VideoMAE (fine-tuned) minus DINOv2 | +0.159 (8 of 9) | yes | +0.104 (9 of 9) | yes |
| DeadAhead minus VideoMAE (fine-tuned) | +0.004 (5 of 9) | no | +0.023 (6 of 9) | no |
| DeadAhead minus DeepHit | +0.007 (5 of 9) | no | +0.026 (7 of 9) | yes |
| DeadAhead minus Fine-tuned DINOv2 + handcrafted | +0.106 (9 of 9) | yes | +0.083 (8 of 9) | yes |
| Same survival head and handcrafted features (fixed-horizon forecasts): VideoMAE features minus DINOv2 features | +0.079 (9 of 9) | yes | +0.069 (8 of 9) | yes |

The DINOv2 comparison does not isolate temporal modelling: those DINOv2 features saw no labels, the DINOv2 baselines embed
9 single frames against VideoMAE's 16-frame clip, and Fine-tuned DINOv2 tuned only two blocks with one fixed
recipe, started from image rather than video pretraining and took its labels from each movie's complete follow-up. It added
little to the handcrafted features alone (lytic 0.217 against 0.216) and stayed below DINOv2 +
handcrafted (0.254); we did not test why.

**Table S5b.** Elapsed time under each scoring, macro AUPRC at 30 min, and the sign of the fitted slope per fold. Table 1's
elapsed time is a logistic regression fitted on the training groups, and its row weights favour early deaths. Its lytic
slope is negative in all 9 folds, so it ranks early forecasts first and falls below chance; unweighted, or with
later forecasts ranking higher (the alarm score, with complete follow-up), it rises above chance. For apoptosis it ranks
forecasts above chance on every line: timing alone carries signal.

| Score | Lytic AUPRC | Apoptosis AUPRC | Lytic slope below / above 0 | Apoptosis slope below / above 0 |
|---|:---:|:---:|:---:|:---:|
| Chance | 0.024 | 0.012 | | |
| Logistic fit with the row weights (Table 1) | 0.020 | 0.068 | 9 / 0 of 9 | 0 / 9 of 9 |
| Logistic fit without the row weights | 0.041 | 0.068 | 0 / 9 of 9 | 0 / 9 of 9 |
| Later forecasts ranking higher (the alarm score), complete follow-up (`--part horizons`) | 0.039 | 0.041 | | |
| Logistic fit with the row weights, follow-up inside the cut ([S6](#s6-gap-analysis-sensitivity-analyses-censoring-and-label-audit)) | 0.020 | 0.039 | | |

### S6. Gap analysis, sensitivity analyses, censoring and label audit

**Table S6a. Gap analysis, post hoc** (recomputed by `deadahead reproduce --part gap`). Forecasts whose cell died within 60
min are left out, and deaths 60 to 120 min ahead are ranked by the 60-to-120-min increment of the risk, macro AUPRC.
DINOv2 + handcrafted is under the same survival head. [S8](#s8-post-hoc-analyses) repeats this analysis at other gaps and
horizons (post hoc).

| Route | Chance | DeadAhead | DINOv2 + handcrafted | Elapsed time | DeadAhead minus DINOv2 + handcrafted (groups ahead) | The same lead at 30 min (`--part horizons`) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Lytic | 0.052 | 0.175 | 0.162 | 0.071 | +0.013 (5 of 9) | +0.079 (9 of 9) |
| Apoptosis | 0.026 | 0.091 | 0.084 | 0.080 | +0.007 (5 of 9) | +0.069 (8 of 9) |

**Table S6b. Probability accuracy** (recomputed by `deadahead reproduce --part horizons`). Lytic IPA at 30 min.

| DeadAhead | DINOv2 + handcrafted, same head | Groups with DeadAhead ahead |
|:---:|:---:|:---:|
| 0.031 | 0.089 | 3 of 9 |

**Table S6c. Complete follow-up** (recomputed by `deadahead reproduce --part sensitivity`). Every method restricted to
forecasts with complete follow-up inside the cut; warnings restricted to forecasts with four hours of follow-up inside
it, with thresholds reselected on the restricted training forecasts. Warned: means over groups; rates: achieved alarms per
event-free reference cell-hour, pooled. The rows of groups ahead count DeadAhead against elapsed time.

| Quantity | Main analysis | Follow-up inside the cut |
|---|:---:|:---:|
| DeadAhead minus DINOv2 + handcrafted, lytic, 30 min (groups ahead) | +0.068 (8 of 9) | +0.069 (8 of 9) |
| DeadAhead minus DINOv2 + handcrafted, apoptosis, 30 min (groups ahead) | +0.033 (5 of 9) | +0.033 (4 of 9) |
| Elapsed time (logistic fit), lytic AUPRC | 0.020 | 0.020 |
| Elapsed time (logistic fit), apoptosis AUPRC | 0.068 | 0.039 |
| DeadAhead's achieved lytic rate | 0.24 | 0.25 |
| DeadAhead's achieved apoptosis rate | 0.24 | 0.25 |
| Reference forecasts | 38,014 | 35,767 |
| Eligible lytic deaths, X = 60 min | 1,011 | 937 |
| DeadAhead warned 60 min ahead, lytic (%) | 66.2 | 66.9 |
| Elapsed time warned 60 min ahead, lytic (%) | 10.5 | 10.8 |
| Eligible apoptotic deaths, X = 60 min | 500 | 371 |
| DeadAhead warned 60 min ahead, apoptosis (%) | 62.1 | 63.7 |
| Elapsed time warned 60 min ahead, apoptosis (%) | 37.8 | 32.2 |
| Eligible apoptotic deaths, X = 120 min | 498 | 369 |
| DeadAhead warned 120 min ahead, apoptosis (%) | 44.6 | 46.3 |
| Elapsed time warned 120 min ahead, apoptosis (%) | 30.1 | 21.2 |
| Groups with DeadAhead ahead, apoptosis, X = 120 min | 5 of 9 | 7 of 9 |
| Eligible apoptotic deaths, X = 240 min | 478 | 349 |
| DeadAhead warned 240 min ahead, apoptosis (%) | 25.1 | 22.5 |
| Elapsed time warned 240 min ahead, apoptosis (%) | 17.0 | 0.0 |
| Groups with DeadAhead ahead, apoptosis, X = 240 min | 5 of 9 | 9 of 9 |

The restriction leaves out the 16 forecasts at frame 430 that count only because their cell died by the cut
(13 apoptotic, 3 lytic). A forecast whose follow-up ends inside the horizon with no death is left
out, so forecasts near the end of a movie count mostly when the cell dies, which favours a score that rises with time.

**Table S6d. Keeping censored cells** (recomputed by `deadahead reproduce --part censoring`). As censoring here comes only
from movies ending, a pre-specified experiment on the evaluation cohort adds 40% administrative censoring in
training (Handcrafted; scored against the uncensored labels). Lead of censoring-aware training (groups ahead of
9). The first row is the primary comparison.

| Censoring-aware training minus | Horizon (min) | Lytic | Meets our rule | Apoptosis | Meets our rule |
|---|:---:|:---:|:---:|:---:|:---:|
| Counting censored cells as negatives | 30 | +0.002 (4 of 9) | no | +0.001 (5 of 9) | no |
| Counting censored cells as negatives | 240 | −0.049 (lower in 7 of 9) | yes, behind | −0.026 (lower in 5 of 9) | no |
| Deleting censored cells | 30 | +0.042 (8 of 9) | yes | +0.019 (8 of 9) | no |

**Table S6e. Training-label audit** (reported from the paper's runs). Training windows that contain an untyped death were
labelled event-free when VideoMAE (fine-tuned) and Fine-tuned DINOv2 were trained, a limitation DeadAhead and DeepHit
inherit through VideoMAE features. Such windows exceeded a limit only with PNT1A bp held out, so every main comparison is
also computed without that fold, as a fold-influence check.

| Quantity, PNT1A bp held out | Value | Limit |
|---|:---:|:---:|
| Windows that contain an untyped death | 5 | |
| Their share of the class-weighted training weight | 0.003% | 0.1% |
| Their share of the apoptotic weight, counted as deaths | 1.3 to 1.4% | 1% |

With DU145 bp held out, one such window entered the inner selection group; re-scoring without it left the
choice unchanged.

Routes are expert morphotypes, and a QPI classifier matched 75.4% of them (Vicar et al., 2020). Near death,
outlines were usable in 59 of 79 sampled tracks (14 of 30 under staurosporine).

### S7. Exploratory probe: yeast death

*Recomputed by `deadahead yeast-score` from the yeast bundle (`deadahead_yeast_bundle_v1.zip`, 27 MB, CC BY
4.0; unzip it into `$DEADAHEAD_DATA`, which creates `yeast_bundle/`; see [`docs/WEIGHTS.md`](docs/WEIGHTS.md)).*

The same recipe on budding-yeast death (DetecDiv, Aspert et al., 2022; Zenodo 6078462, CC BY 4.0), over 7
held-out positions. The fine-tuned start is the released DeadAhead VideoMAE that held out PNT1A bp (same seed).

**Table S7.** Macro AUPRC at 30 min over the 7 held-out positions. With 7 positions, our decision
rule needs a lead of at least 0.02 with the same sign on at least 6 of them. The best simple rule is picked per
test position. Chance: the positive rate averaged over the positions.

| Model | Macro AUPRC | Lead of the fine-tuned start (positions ahead) | Meets our rule |
|---|:---:|:---:|:---:|
| Chance | 0.0072 | | |
| VideoMAE fine-tuned on yeast, from the released DeadAhead VideoMAE | 0.0488 | | |
| VideoMAE fine-tuned on yeast, from the public VideoMAE weights | 0.0459 | +0.003 (5 of 7) | no |
| Best simple rule | 0.0394 | +0.009 (4 of 7) | no |

`deadahead yeast-score --no-sensitivity --no-lead-time` gives the numbers above in about 5 minutes with one thread on a
shared 2.2 GHz Xeon server; `deadahead yeast-score` adds the label sensitivity and the lead times and took well over 20
minutes there. Set `OMP_NUM_THREADS=1` on a shared machine; the LightGBM label rules always use 2 threads. `yeast-score` checks the bundle
against its `MANIFEST.json` and returns 3 on a mismatch. The code is in `deadahead/yeast/`; the trained yeast models were
not kept, so the predictions in the bundle are the record of those runs. To score your own runs, make a folder with
`labels/` (the classes, brightness and size parquet files plus `folds.json` from `$DEADAHEAD_DATA/yeast`), `predictions/`
(`runs/yeast/pred`) and `survival/` (`runs/yeast/surv`), then run `deadahead yeast-score --bundle <folder> --no-verify`.

### S8. Post hoc analyses

*Post hoc: these analyses were defined after every horizon, gap and warning result above had been seen. They read the
stored predictions only, and the within-hour scoring refits elapsed time as in Table 1; nothing else was
trained or predicted again. Recomputed by `deadahead reproduce --part gap-window`,
`--part alarm-rate` and `--part within-time`; the alarm share by time before death and the match group by group come from
`deadahead plot-figures`, which reads the bundle. Under `--part all` each reproduce part is also checked against the parts
it extends: g = 0 against `horizons`, g = 60 against `gap`, the targets 0.2 and 0.4 against `warnings`
and `image-warnings`, and the within-hour scoring with one band for every forecast against `sensitivity` (the `checks` block
of its JSON output).*

**Forecasting beyond the last minutes.** A forecast made a few minutes before a cell dies may read a death
already under way. So, for a gap g, the forecasts whose cell died (by either route) within g
min are left out, and we ask how well each method ranks the cells that die by the route later in the window, between g
and H min ahead. A method whose ranking holds as g grows is forecasting deaths beyond the last minutes. Nothing is
claimed here about when the death process began.

**Scoring.** The score is each method's stored probability of death by the route within H min (elapsed time: later
forecasts rank higher). The AUPRC runs over the forecasts that remain: deaths by the route between g and H min ahead are the
positives; deaths by the other route in that window and cells followed event-free through H are the negatives; follow-up
that ends before H without a death is left out and never counts as a negative. Values are averaged over the seeds within
each held-out group, then over the 9 groups (chance: the mean share of positives). DINOv2 + handcrafted is
here under the same survival head as DeadAhead. H = 120 min gives the primary windows, H = 240 min the secondary ones. A
difference that meets our decision rule is marked; there are no p-values.

**Table S8a (post hoc).** Lytic death, macro AUPRC per window (g, H] in minutes before death (`--part gap-window`).

| Window (min) | Chance | DeadAhead | DINOv2 + handcrafted | Elapsed time | DeadAhead minus DINOv2 + handcrafted (groups ahead) | DeadAhead minus elapsed time (groups ahead) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| (0, 120] | 0.096 | 0.402 | 0.347 | 0.128 | +0.055 (6 of 9) | +0.274 (9 of 9, meets the rule) |
| (30, 120] | 0.075 | 0.292 | 0.251 | 0.101 | +0.041 (6 of 9) | +0.191 (8 of 9, meets the rule) |
| (60, 120] | 0.052 | 0.186 | 0.166 | 0.071 | +0.020 (6 of 9) | +0.115 (9 of 9, meets the rule) |
| (90, 120] | 0.027 | 0.098 | 0.089 | 0.037 | +0.009 (6 of 9) | +0.061 (8 of 9, meets the rule) |
| (0, 240] | 0.189 | 0.482 | 0.461 | 0.217 | +0.021 (5 of 9) | +0.264 (8 of 9, meets the rule) |
| (30, 240] | 0.172 | 0.431 | 0.415 | 0.198 | +0.016 (5 of 9) | +0.233 (8 of 9, meets the rule) |
| (60, 240] | 0.154 | 0.381 | 0.370 | 0.177 | +0.012 (4 of 9) | +0.205 (8 of 9, meets the rule) |
| (90, 240] | 0.133 | 0.336 | 0.324 | 0.154 | +0.011 (4 of 9) | +0.182 (8 of 9, meets the rule) |
| (120, 240] | 0.112 | 0.285 | 0.276 | 0.129 | +0.009 (4 of 9) | +0.156 (8 of 9, meets the rule) |

**Table S8b (post hoc).** Apoptosis, the same.

| Window (min) | Chance | DeadAhead | DINOv2 + handcrafted | Elapsed time | DeadAhead minus DINOv2 + handcrafted (groups ahead) | DeadAhead minus elapsed time (groups ahead) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| (0, 120] | 0.047 | 0.227 | 0.179 | 0.153 | +0.048 (8 of 9, meets the rule) | +0.074 (6 of 9) |
| (30, 120] | 0.037 | 0.152 | 0.128 | 0.116 | +0.023 (7 of 9, meets the rule) | +0.036 (6 of 9) |
| (60, 120] | 0.026 | 0.100 | 0.091 | 0.080 | +0.009 (5 of 9) | +0.021 (6 of 9) |
| (90, 120] | 0.014 | 0.054 | 0.053 | 0.043 | +0.001 (5 of 9) | +0.011 (7 of 9) |
| (0, 240] | 0.097 | 0.284 | 0.255 | 0.286 | +0.029 (8 of 9, meets the rule) | −0.002 (5 of 9) |
| (30, 240] | 0.089 | 0.242 | 0.226 | 0.264 | +0.016 (6 of 9) | −0.022 (5 of 9) |
| (60, 240] | 0.081 | 0.208 | 0.200 | 0.238 | +0.008 (5 of 9) | −0.030 (5 of 9) |
| (90, 240] | 0.071 | 0.176 | 0.172 | 0.207 | +0.004 (5 of 9) | −0.031 (5 of 9) |
| (120, 240] | 0.060 | 0.142 | 0.142 | 0.168 | +0.001 (5 of 9) | −0.026 (5 of 9) |

DeadAhead stayed above chance in every window. For lytic death it met our decision rule against elapsed time in every
window and against DINOv2 + handcrafted under the same survival head in none. For apoptosis within 120 min it met the
rule against that baseline at g = 0 and 30 min only, and against elapsed time at no gap; within 240 min elapsed time
scored above DeadAhead at every gap, though not by the rule in either direction. As in S6, forecasts near the end of a
movie count mostly when the cell dies, which favours a score that rises with time. On forecasts whose 240 min end inside
the cut (the fixed horizons of `--part sensitivity`),
DeadAhead reached 0.249 and elapsed time 0.176 for apoptosis, a lead of +0.074, ahead in
7 of 9 groups (at H = 240 min on all forecasts, g = 0: 0.284 against 0.286,
ahead in 5 of 9).

[S6](#s6-gap-analysis-sensitivity-analyses-censoring-and-label-audit)'s gap analysis (g = 60 min, H = 120 min) scores the
60-to-120-min increment F(120) minus F(60) rather than the whole 120-min risk F(120), so its values differ from the (60,
120] rows above; `--part gap-window` writes that increment beside the primary score in its JSON output (`secondary_macro`
under `summary`), and at g = 60 min it gives S6's values for DeadAhead (0.175 lytic, 0.091 apoptosis). The
forecast at 90 min is not stored, so the increment is not computed at g = 90 min.

![Macro AUPRC against the gap before death, per route and horizon (post hoc)](docs/figures/figS8a_gap_window.png)

**Figure S8a (post hoc).** Macro AUPRC over the 9 held-out groups against the gap g, per route and horizon,
with chance as the grey line; DINOv2 + handcrafted under the same survival head as DeadAhead. The values are in
[`docs/figures/data/figS8a_gap_window.csv`](docs/figures/data/figS8a_gap_window.csv).

**Warnings against the alarm rate.** How do the share of deaths warned and the alarm burden trade off across target rates,
and at a matched achieved lytic alarm rate (0.309 against 0.307, pooled), which method warns more
lytic deaths 60 min ahead?

**Scoring.** Each method raises an alarm when its predicted 30-min risk (elapsed time: its score) exceeds a threshold chosen
on the training groups only, as in [S3](#s3-warnings-in-detail) with only the target changed: from 0.05 to 0.80 alarms per
event-free reference cell-hour in steps of 0.05 (for matching, DeadAhead and DINOv2 + handcrafted also from 0.005 to 0.800
in steps of 0.005). Here DINOv2 + handcrafted is the baseline with its own head, as in S3. Achieved rates on the
held-out groups differ from the target, so each row prints the achieved rate (pooled over the held-out groups) beside
the share of deaths warned 60 min ahead (the alarm on at every forecast in the last 60 min, among deaths tracked that
long; mean over the 9 groups). A difference at the same target compares different achieved rates; only the
matched points below (0.309 against 0.307 pooled, lytic) compare matched alarm burdens.

**Table S8c (post hoc).** Lytic death: target, achieved alarms per event-free reference cell-hour (pooled) and share of
deaths warned 60 min ahead (%, mean over the groups), per method (`--part alarm-rate`).

| Target | DeadAhead: achieved | DeadAhead: warned (%) | DINOv2 + handcrafted: achieved | DINOv2 + handcrafted: warned (%) | Elapsed time: achieved | Elapsed time: warned (%) | DeadAhead minus DINOv2 + handcrafted, points (groups ahead) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.05 | 0.08 | 38.5 | 0.08 | 32.8 | 0.06 | 4.7 | +5.7 (6 of 9) |
| 0.10 | 0.13 | 51.3 | 0.16 | 48.0 | 0.11 | 6.4 | +3.3 (6 of 9) |
| 0.15 | 0.19 | 59.4 | 0.24 | 57.2 | 0.16 | 9.4 | +2.2 (6 of 9) |
| 0.20 | 0.24 | 66.2 | 0.31 | 63.5 | 0.21 | 10.5 | +2.7 (5 of 9) |
| 0.25 | 0.30 | 71.7 | 0.38 | 67.9 | 0.26 | 13.9 | +3.8 (5 of 9) |
| 0.30 | 0.36 | 76.1 | 0.44 | 71.8 | 0.31 | 16.7 | +4.3 (6 of 9) |
| 0.35 | 0.41 | 79.8 | 0.50 | 75.0 | 0.36 | 19.0 | +4.8 (4 of 9) |
| 0.40 | 0.47 | 83.0 | 0.56 | 77.9 | 0.41 | 21.2 | +5.1 (4 of 9) |
| 0.45 | 0.53 | 85.9 | 0.61 | 80.0 | 0.46 | 23.0 | +5.9 (6 of 9) |
| 0.50 | 0.58 | 88.6 | 0.66 | 82.3 | 0.51 | 26.1 | +6.3 (6 of 9) |
| 0.55 | 0.64 | 90.6 | 0.71 | 84.0 | 0.56 | 29.9 | +6.5 (6 of 9) |
| 0.60 | 0.69 | 92.2 | 0.75 | 85.6 | 0.61 | 32.5 | +6.6 (6 of 9) |
| 0.65 | 0.74 | 93.5 | 0.80 | 87.0 | 0.66 | 36.4 | +6.4 (6 of 9) |
| 0.70 | 0.80 | 94.4 | 0.84 | 88.0 | 0.71 | 39.0 | +6.4 (6 of 9) |
| 0.75 | 0.85 | 95.4 | 0.88 | 88.9 | 0.76 | 42.1 | +6.5 (6 of 9) |
| 0.80 | 0.90 | 96.2 | 0.92 | 89.9 | 0.81 | 45.4 | +6.3 (6 of 9) |

**Table S8d (post hoc).** Apoptosis, the same.

| Target | DeadAhead: achieved | DeadAhead: warned (%) | DINOv2 + handcrafted: achieved | DINOv2 + handcrafted: warned (%) | Elapsed time: achieved | Elapsed time: warned (%) | DeadAhead minus DINOv2 + handcrafted, points (groups ahead) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.05 | 0.09 | 31.8 | 0.06 | 20.3 | 0.06 | 19.2 | +11.6 (6 of 9) |
| 0.10 | 0.15 | 45.3 | 0.11 | 33.0 | 0.11 | 25.9 | +12.3 (6 of 9) |
| 0.15 | 0.20 | 55.7 | 0.17 | 42.8 | 0.16 | 31.5 | +12.9 (7 of 9, meets the rule) |
| 0.20 | 0.24 | 62.1 | 0.22 | 49.9 | 0.21 | 37.8 | +12.2 (6 of 9) |
| 0.25 | 0.29 | 67.9 | 0.27 | 56.5 | 0.26 | 42.8 | +11.4 (7 of 9, meets the rule) |
| 0.30 | 0.34 | 72.6 | 0.32 | 61.0 | 0.31 | 49.1 | +11.6 (7 of 9, meets the rule) |
| 0.35 | 0.39 | 75.1 | 0.37 | 65.2 | 0.36 | 51.2 | +9.9 (7 of 9, meets the rule) |
| 0.40 | 0.44 | 77.6 | 0.42 | 68.2 | 0.41 | 53.2 | +9.4 (7 of 9, meets the rule) |
| 0.45 | 0.49 | 79.7 | 0.47 | 71.0 | 0.46 | 55.6 | +8.7 (6 of 9) |
| 0.50 | 0.54 | 82.9 | 0.51 | 74.2 | 0.51 | 58.4 | +8.7 (7 of 9, meets the rule) |
| 0.55 | 0.59 | 84.9 | 0.56 | 77.0 | 0.56 | 61.6 | +7.9 (7 of 9, meets the rule) |
| 0.60 | 0.64 | 87.5 | 0.60 | 78.4 | 0.61 | 63.9 | +9.1 (7 of 9, meets the rule) |
| 0.65 | 0.69 | 89.1 | 0.64 | 80.2 | 0.66 | 67.0 | +8.8 (7 of 9, meets the rule) |
| 0.70 | 0.73 | 90.2 | 0.69 | 81.9 | 0.71 | 69.7 | +8.3 (7 of 9, meets the rule) |
| 0.75 | 0.78 | 91.1 | 0.73 | 83.1 | 0.76 | 72.2 | +8.0 (7 of 9, meets the rule) |
| 0.80 | 0.83 | 92.3 | 0.78 | 84.4 | 0.81 | 73.9 | +8.0 (7 of 9, meets the rule) |

![Deaths warned 60 min ahead against the achieved alarm rate (post hoc)](docs/figures/figS8b_alarm_rate.png)

**Figure S8b (post hoc).** Deaths warned 60 min ahead (mean over the 9 groups) against the achieved
pooled alarm rate, for DeadAhead, DINOv2 + handcrafted with its own head and elapsed time, each with its own training-only
thresholds. Markers: the targets of Tables S8c and S8d; lines through the DeadAhead and DINOv2 + handcrafted markers: the
fine grid of targets; rings: the matched lytic comparison, with both achieved rates printed. The values are in
[`docs/figures/data/figS8b_alarm_rate.csv`](docs/figures/data/figS8b_alarm_rate.csv).

**Matched comparison (post hoc; lytic death, the single comparison of this analysis).** At its target of 0.2,
DINOv2 + handcrafted achieved 0.307 lytic alarms per event-free reference cell-hour (pooled). On the fine
grid of targets, DeadAhead's achieved pooled rate came closest to that value at a target of 0.26. This
target was chosen post hoc, by matching achieved rates on the held-out groups, while each threshold was still set on
the training groups only. At the matched rates, 0.309 for DeadAhead and 0.307 for
DINOv2 + handcrafted, DeadAhead warned 72.8% of lytic deaths 60 min ahead and DINOv2 +
handcrafted 63.5%: a mean difference of +9.3 points, DeadAhead ahead in 7 of
9 groups and behind in 2, which meets our decision rule (+11.3 points,
7 of 8, without PNT1A bp). Matched the other way round, DeadAhead is ahead in
6 of 9 groups, which does not meet it. A matched alarm burden is claimed only at these points.

**Table S8e (post hoc; `--part alarm-rate`).** The first row is the matched comparison. The other two rows are for context
and decide nothing: the same match for apoptosis, and the reverse match (DINOv2 + handcrafted's target closest to DeadAhead's
achieved lytic rate at its target of 0.2). Achieved rates: alarms per event-free reference cell-hour,
pooled and as the mean over groups.

| Match | DeadAhead: target | DeadAhead: achieved (pooled) | DeadAhead: achieved (mean over groups) | DeadAhead: warned (%) | DINOv2 + handcrafted: target | DINOv2 + handcrafted: achieved (pooled) | DINOv2 + handcrafted: achieved (mean over groups) | DINOv2 + handcrafted: warned (%) | Difference, points (groups ahead) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Lytic | 0.26 | 0.309 | 0.365 | 72.8 | 0.2 | 0.307 | 0.348 | 63.5 | +9.3 (7 of 9) |
| Apoptosis | 0.18 | 0.223 | 0.306 | 60.3 | 0.2 | 0.222 | 0.254 | 49.9 | +10.4 (6 of 9) |
| Lytic, reverse | 0.2 | 0.244 | 0.290 | 66.2 | 0.155 | 0.243 | 0.279 | 58.1 | +8.1 (6 of 9) |

**Table S8f (post hoc, for context only).** The match group by group, from `deadahead plot-figures`. The rates
above are matched pooled over the groups, while group rates still differ. Here, within each held-out group and seed,
DeadAhead's threshold is read off that group's reference forecasts, so that it raises as many reference alarms as DINOv2 +
handcrafted at its target of 0.2. This threshold has seen the held-out group.

| Match | DeadAhead: warned (%) | DINOv2 + handcrafted: warned (%) | Difference, points (groups ahead) |
|---|:---:|:---:|:---:|
| Lytic, group by group | 71.2 | 63.5 | +7.7 (8 of 9) |

The per-group values are in [`docs/figures/data/s8_group_match.csv`](docs/figures/data/s8_group_match.csv).

**Alarm share by time before death.** How often is DeadAhead's alarm on at each time before death, against
the alarm share on event-free reference forecasts?

![Share of forecasts with the alarm on per 30-min bin before death, by route, against the share of reference forecasts (post hoc)](docs/figures/figS8c_alarm_share.png)

**Figure S8c (post hoc).** Forecasts of dying cells with DeadAhead's alarm on, per 30-min bin before death, at the
training-only threshold of target 0.2 (seed means, then means over the 9 held-out groups). Lytic
death runs from 85.7% in the last 30 min to 23.8% at 210 to 240 min, apoptosis from 77.1% to
35.9%; the grey line is the share of held-out reference forecasts that raise an alarm, also a mean over the groups
(14.5% and 16.3%). The paper's and [S3](#s3-warnings-in-detail)'s 12.2% is the same lytic share
pooled over the groups. The values are in
[`docs/figures/data/figS8c_alarm_share.csv`](docs/figures/data/figS8c_alarm_share.csv).

**Scoring within each hour of imaging (post hoc).** Scores can rise with elapsed imaging time. Forecasts just
before an apoptotic death tend to come later in a movie than event-free ones; for lytic death the pattern is mixed.
If a forecast competes only with forecasts of the
same group made in the same hour of imaging, how well does each method rank the forecasts followed by a death by the route
within 30 min? This analysis was defined after every Table 1, horizon, gap and warning result had been seen, and an earlier
form of the second design below had been seen too; the within-hour AUPRC was chosen before any method's value under it
existed. The reading is descriptive: DeadAhead is called ahead only where our decision rule says so, and there are no
p-values.

**Scoring.** The rows are Table 1's at the common cut of [Table S6c](#s6-gap-analysis-sensitivity-analyses-censoring-and-label-audit),
t + 10 <= 438, with Table 1's labels and each method's stored 30-min score (elapsed time: its Table 1 fit). An hour is a band of 20 frames of
forecast time, from frame 40. Within each held-out group, for each method, route and seed, every hour that held a death by the route is
scored on its own by Table 1's AUPRC, over that hour's forecasts of that group. The hours are averaged with their deaths as
weights, so each of the 1,017 lytic and 491 apoptotic deaths with a forecast in its last 30 min counts
once, and an hour without such a death adds nothing. Seeds and groups are then averaged as in Table 1. Only hours with a
death count, so 36,642 of the 51,441 forecasts count for lytic death and 29,941 for apoptosis; they
differ by route, never by method.

**Its own scale.** Within-hour AUPRC is not on Table 1's scale and is never compared with it: a perfect score reaches 1,
but a random order reaches 0.070 on each route (its exact expected AUPRC in each hour, weighted the same way),
because an hour holds few forecasts and only hours with a death count. The hours also take away the comparison between
hours, which is real information for any score, so a difference from a Table 1 value, in either direction, is not read as
the share of the ranking that is timing. Elapsed time has no order among forecasts made at the same time, so it is set beside
the positive rate of the hours counted, not beside a random order.

**Table S8g (post hoc; `--part within-time`).** Within-hour AUPRC at 30 min, macro over the 9 held-out groups, with
DeadAhead's lead, the groups ahead and whether the lead meets our decision rule. The row of DINOv2 + handcrafted is the
primary comparison; the other rows decide nothing. Leads are computed before rounding, here and in Tables S8h and S8i.

| Method | Lytic | DeadAhead minus the method (groups ahead) | Apoptosis | DeadAhead minus the method (groups ahead) |
|---|:---:|:---:|:---:|:---:|
| **DeadAhead (ours)** | **0.426** | | **0.325** | |
| VideoMAE (fine-tuned) | 0.417 | +0.010 (6 of 9) | 0.307 | +0.019 (7 of 9) |
| DeepHit | 0.418 | +0.008 (8 of 9) | 0.304 | +0.021 (7 of 9, meets the rule) |
| DINOv2 + handcrafted | 0.353 | +0.073 (8 of 9, meets the rule) | 0.299 | +0.027 (5 of 9) |
| Fine-tuned DINOv2 + handcrafted | 0.309 | +0.118 (9 of 9, meets the rule) | 0.254 | +0.071 (8 of 9, meets the rule) |
| Handcrafted | 0.328 | +0.098 (8 of 9, meets the rule) | 0.279 | +0.047 (6 of 9) |
| DINOv2 | 0.240 | +0.186 (9 of 9, meets the rule) | 0.175 | +0.150 (9 of 9, meets the rule) |
| A random order (beside the methods above) | 0.070 | | 0.070 | |
| Elapsed time | 0.054 | +0.373 (9 of 9, meets the rule) | 0.045 | +0.280 (9 of 9, meets the rule) |
| The positive rate of the hours counted (beside elapsed time) | 0.052 | | 0.042 | |

Post hoc, DeadAhead met our decision rule against DINOv2 + handcrafted for lytic death (+0.073, 8 of 9 groups;
+0.059, 7 of 8, without PNT1A bp) and not for apoptosis (+0.027, 5 of
9; +0.031, 5 of 8, without PNT1A bp).

**Table S8h (post hoc; `--part within-time`).** DeadAhead against DINOv2 + handcrafted with bands of forecast time 1 hour,
2 hours or 30 min wide. The 1-hour bands are the primary analysis; the 2-hour and 30-min bands are secondary and decide
nothing. A random order reaches a different value at each width, so values are read within a row.

| Bands | Route | A random order | DeadAhead | DINOv2 + handcrafted | DeadAhead minus DINOv2 + handcrafted (groups ahead) |
|---|---|:---:|:---:|:---:|:---:|
| 1 hour (primary) | Lytic | 0.070 | 0.426 | 0.353 | +0.073 (8 of 9, meets the rule) |
| 1 hour (primary) | Apoptosis | 0.070 | 0.325 | 0.299 | +0.027 (5 of 9) |
| 2 hours | Lytic | 0.059 | 0.400 | 0.324 | +0.076 (7 of 9, meets the rule) |
| 2 hours | Apoptosis | 0.052 | 0.282 | 0.253 | +0.029 (6 of 9) |
| 30 min | Lytic | 0.086 | 0.466 | 0.397 | +0.069 (8 of 9, meets the rule) |
| 30 min | Apoptosis | 0.093 | 0.374 | 0.358 | +0.016 (6 of 9) |

**A second design, which decides nothing (post hoc).** Here each score becomes its rank position within its hour and group,
(r - 1) / (n - 1) with r the average rank and n the forecasts of that hour (0.5 for an hour of one forecast), and one AUPRC is
taken over the group's forecasts. It stays here so that the design seen first stays visible. It is not on Table 1's scale
either: every hour, with or without a death, puts forecasts at the top of the pooled list, so a score that puts every death
first in its hour, with no two forecasts of an hour tied, stops at the ceiling below. The label itself, which ties forecasts
inside an hour, reaches 1.

**Table S8i (post hoc; `--part within-time`; a second design, deciding nothing).** AUPRC on the hour ranks, macro over the
9 held-out groups, 1-hour bands.

| Route | DeadAhead | DINOv2 + handcrafted | DeadAhead minus DINOv2 + handcrafted (groups ahead) | Ceiling |
|---|:---:|:---:|:---:|:---:|
| Lytic | 0.192 | 0.149 | +0.043 (7 of 9, meets the rule) | 0.565 |
| Apoptosis | 0.081 | 0.073 | +0.008 (5 of 9) | 0.440 |

## Pretrained models

| Files | Number | Licence |
|---|:---:|---|
| Fine-tuned VideoMAE-L, one per held-out group and seed | 27 | CC BY-NC 4.0 |
| DeadAhead survival heads | 27 | MIT |

The [model card](docs/MODEL_CARD.md) says what they are and how to use them; sizes and checksums are in
[`docs/WEIGHTS.md`](docs/WEIGHTS.md). Download: https://zenodo.org/records/23883153.

## Data

CellDeathDetect is public on Zenodo (record 2601562, CC BY 4.0). Download the three archives, check them against
[`data/celldeathdetect_md5.txt`](data/celldeathdetect_md5.txt), and extract them into
`$DEADAHEAD_DATA/raw/celldeathdetect/` (by default `data/` inside the repository). [`docs/DATA.md`](docs/DATA.md), the data card, gives
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

The yeast probe (S7) reads the DetecDiv archive (see [`docs/DATA.md`](docs/DATA.md)):

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

## Repository layout

```
deadahead/                    the Python package (below)
configs/                      settings of the windows, the survival head, VideoMAE and the evaluation (alarms, horizons)
splits/folds.json             the nine leave-one-group-out folds
data/celldeathdetect_md5.txt  checksums of the three CellDeathDetect archives
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
| `deadahead/report/bundle_tables.py` | Tables of the README read from the predictions bundle, not from the reproduce output: the weight of each outcome in the training rows, and two post hoc views of the warnings (the alarm share per 30-min bin before death, the match group by group) |
| `deadahead/report/figures.py` | Figures for the README: the warning curves of Figure 2, the overview of Table 1, the figures of S1, S2 and S4 and the post hoc figures of S8 |
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
| Code | MIT ([`LICENSE`](LICENSE)) |
| Tables and predictions derived from CellDeathDetect, including the predictions bundle | CC BY 4.0, with attribution to Vicar et al. (2020) |
| Fine-tuned VideoMAE weights | CC BY-NC 4.0, inherited from the VideoMAE checkpoint (MCG-NJU/videomae-large) |
| DeadAhead survival heads (`deadahead_head__*.txt`) | MIT; their inputs include the features of the fine-tuned VideoMAE weights (CC BY-NC 4.0) |
| Yeast predictions bundle, derived from DetecDiv (Aspert et al., 2022) | CC BY 4.0 |
| Figures and figure data in `docs/figures/` | CC BY 4.0; Figure 1 shows CellDeathDetect crops (Vicar et al., 2020) |

Third-party components and datasets are listed in [`NOTICE.md`](NOTICE.md).

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
