# Model card: DeadAhead

DeadAhead forecasts when and how a cell will die from quantitative phase video. This card covers the models behind the
paper's results, one per held-out group and seed. File sizes, checksums and loading code are in
[`WEIGHTS.md`](WEIGHTS.md); the data card is [`DATA.md`](DATA.md).

## What the models are

| Files | Number | What they are | Licence |
|---|:---:|---|---|
| `videomae_large_finetuned__<group>__seed<seed>.safetensors` | 27 | VideoMAE-L fine-tuned end to end on CellDeathDetect. Input: 16 frames over the past 120 min, as 128-pixel crops around the cell, resized to 224 pixels. Output: no event, apoptosis or lytic death within 30 min | CC BY-NC 4.0 |
| `deadahead_head__<group>__seed<seed>.txt` | 27 | Competing-risks survival head: a LightGBM 4.7.0 multiclass booster on the pooled VideoMAE features, the 112 handcrafted features and the bin index. Output: the hazard of each route per 30-min bin, and so the probability of each route within 30, 60, 120 and 240 min | MIT |

There are 9 line-by-treatment groups and three seeds (2601562, 2601563, 2601564). A model whose name carries a group
was trained without that group.

## Intended use

- Recompute or extend the paper's results on CellDeathDetect: forecast a group with the models that held it out.
- Start further research on forecasting cell death from quantitative phase video.

The models are for research. They are not for diagnosis or any clinical decision, and the VideoMAE weights are not for
commercial use.

## Training data

[CellDeathDetect](DATA.md#celldeathdetect) (Vicar et al., 2020; doi:10.5281/zenodo.2601562, CC BY 4.0): 63 quantitative phase
movies of PNT1A, DU145 and LNCaP cells under three treatments. Each model trained on eight groups and held out the ninth;
one setting was chosen on an inner training group. VideoMAE started from `MCG-NJU/videomae-large` (Tong et al., 2022),
pretrained without labels on Kinetics-400. The [method](../README.md#method-in-brief) gives every step.

## Metrics

Held-out groups, mean over the 9 groups. DeadAhead's alarm thresholds were set on training groups only.

| Measure | Lytic death | Apoptosis | Details |
|---|:---:|:---:|---|
| Macro AUPRC at 30 min | 0.322 | 0.213 | [Table 1](../README.md#table-1) |
| Chance | 0.024 | 0.012 | [Table 1](../README.md#table-1) |
| Deaths warned 60 min ahead, target 0.2 | 66.2% (achieved rate 0.24) | 62.1% (achieved rate 0.24) | [Figure 2](../README.md#figure-2), [S3](../README.md#s3-warnings-in-detail) |

Every horizon is in [S1](../README.md#s1-fixed-horizon-forecasts), every group in [S2](../README.md#s2-per-group-differences).
One command recomputes each number from the released predictions: see the
[claim table](../README.md#which-command-checks-which-claim).

## Limits

- Use a model on the group it held out. The other groups were in its training data.
- One dataset: three prostate cell lines on one microscope. Each held-out line and treatment also appears in training under
  another treatment or line.
- Crops are centred on the dataset's curated outlines, seeded from Hoechst-stained nuclei. Use with other segmentations is
  untested.
- The probabilities rank well but need recalibration before use as risks ([S1](../README.md#s1-fixed-horizon-forecasts)).
- The 216 cross-fitted inner boosters behind the alarm thresholds are not released; the predictions bundle holds their
  predictions.

## Licence and citation

The VideoMAE weights are released under CC BY-NC 4.0, the licence of the checkpoint they start from; the survival heads are
MIT, and their inputs include the features of those weights. Please cite the paper ([`CITATION.cff`](../CITATION.cff)) and
VideoMAE (Tong et al., NeurIPS 2022, arXiv:2203.12602).
