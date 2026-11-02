# Weights

Two sets of files are released, one per held-out group (9 groups: 3 cell lines x 3 treatments) and seed
(2601562, 2601563, 2601564): 27 fine-tuned VideoMAE-large classifiers and 27 DeadAhead survival heads.
Download them from https://zenodo.org/records/23883153 into `$DEADAHEAD_DATA/weights/`.

A model whose name carries a group was trained without that group (leave-one-group-out): use it to
predict that group only. The three seeds of a group are three independent trainings.

## Licences

- `videomae_large_finetuned__*.safetensors`: derived from VideoMAE-large
  (`MCG-NJU/videomae-large`, revision `12da269a02d3e1fbbb7011e610e591fea8061dca`), so they are released under
  **CC BY-NC 4.0**, the licence of that checkpoint. Cite VideoMAE (Tong et al., NeurIPS 2022, arXiv:2203.12602).
- `deadahead_head__*.txt`: **MIT**, as the code; their inputs include the features of the fine-tuned VideoMAE
  weights above (CC BY-NC 4.0).

## Fine-tuned VideoMAE-large (27 files, fp32 safetensors)

`VideoMAEForVideoClassification` with mean pooling and three outputs (no event, apoptosis, lytic death within
30 min), 390 tensors. The provenance column is the sha256 of the training checkpoint (`.pt`) each file was converted
from; the converter checks it, then checks that every tensor of the `.safetensors` file equals the source
(same keys, shapes, dtypes and values). Each run used an effective batch of 128, split as shown; the split
changes the order of random draws, so use the same `--micro-batch` to repeat a run's sampling.

| File | Held-out group | Seed | Micro-batch x accumulation | Size | Training checkpoint sha256 (provenance) | sha256 of the file |
|---|---|---|---|---|---|---|
| `videomae_large_finetuned__PNT1A_bp__seed2601562.safetensors` | PNT1A_bp | 2601562 | 16 x 8 | 1,215,602,132 B (1.22 GB) | `029b0974a07dbcec162917907a85c56b7bf095ef4c4d3871d20f306af764b8db` | `a6c807cf864aeb97017cdd11046969104cd79a4371c30121e8f72ab125cfc822` |
| `videomae_large_finetuned__PNT1A_bp__seed2601563.safetensors` | PNT1A_bp | 2601563 | 16 x 8 | 1,215,602,132 B (1.22 GB) | `4298ecab416530ff8d7496886d2c0bdb79c95966121da92f26890b9da821b3c2` | `73753b803243439bd6a2d53109ee493bdc59c78a7e2be9e119d355cf3715731d` |
| `videomae_large_finetuned__PNT1A_bp__seed2601564.safetensors` | PNT1A_bp | 2601564 | 16 x 8 | 1,215,602,132 B (1.22 GB) | `33ba8ebbb89d32a90024786063dcd8c2c592b50b274960e510a03f5ebba5e9aa` | `7ca1a2f7acb809c1c1807471bdf085377ae074ecad8a6c00a4eed02b844198a2` |
| `videomae_large_finetuned__PNT1A_do__seed2601562.safetensors` | PNT1A_do | 2601562 | 32 x 4 | 1,215,602,132 B (1.22 GB) | `fe2a0d7adefa09905162e3462f2ce9276bc55384c4cc3020c7a47c7984ada33d` | `e8522892f441c22f22d35017db9297852637443d8a15939ceee2ec737ea88499` |
| `videomae_large_finetuned__PNT1A_do__seed2601563.safetensors` | PNT1A_do | 2601563 | 32 x 4 | 1,215,602,132 B (1.22 GB) | `07ea3ce76114af418974c89b6c1e77a6151859e370dc3a8b9f163ebe06c95bd3` | `8f22266a88f78a387be861fda5cfb74bb414771e8ad61356f44479648ff6ddeb` |
| `videomae_large_finetuned__PNT1A_do__seed2601564.safetensors` | PNT1A_do | 2601564 | 32 x 4 | 1,215,602,132 B (1.22 GB) | `fde92755aee9b390d27c7764f1cb2b3ca42fd75ad56b9e3bf79fc41bdcef3698` | `d366f6698f89a4174d32cc677af4a40d2f5512cbad462176cc0d4b9c4d4849de` |
| `videomae_large_finetuned__PNT1A_st__seed2601562.safetensors` | PNT1A_st | 2601562 | 16 x 8 | 1,215,602,132 B (1.22 GB) | `550eb75cad29d1efd2b0ed7ec811f9c72b4447d34b01948821dbb4fe5ac5055e` | `9ca2140ac1587c2d921ab7fa69da3a5ace805a7582901658d7ac36fd5cf92531` |
| `videomae_large_finetuned__PNT1A_st__seed2601563.safetensors` | PNT1A_st | 2601563 | 32 x 4 | 1,215,602,132 B (1.22 GB) | `11305206429b56c0ffe192c6943c5d60662aef0b330d823988c68c8b3c12d108` | `a5eb2c897f032dcb084e64f0ed22e47fed839cdd9951cab03c1260dbcd6a111e` |
| `videomae_large_finetuned__PNT1A_st__seed2601564.safetensors` | PNT1A_st | 2601564 | 16 x 8 | 1,215,602,132 B (1.22 GB) | `495d0d7efc17d5df750926fbd089772af6598e5b2e57c5e52343b628843314da` | `0390d5d3ed41d91b5278c195a746c7d4a9f3c13baeda9ee0dd51024608321988` |
| `videomae_large_finetuned__DU145_bp__seed2601562.safetensors` | DU145_bp | 2601562 | 16 x 8 | 1,215,602,132 B (1.22 GB) | `681eb7e416362543a6c8e78793c91b7ea85100dc1390023a0ed06af96f1e6703` | `f170ccb1fc68d9f03f21a002e043ed5b2a26fcd8149d93a9e4dec8f39635b5ee` |
| `videomae_large_finetuned__DU145_bp__seed2601563.safetensors` | DU145_bp | 2601563 | 32 x 4 | 1,215,602,132 B (1.22 GB) | `9f735569ebf22066cf900c8fe3e9babd422f2c4193ab748fdef5d35aa6549944` | `7ac3fa07e4daac96bbc40483ca6448e08249a975f0ec6605bdec91e4f8bee1f0` |
| `videomae_large_finetuned__DU145_bp__seed2601564.safetensors` | DU145_bp | 2601564 | 16 x 8 | 1,215,602,132 B (1.22 GB) | `6562f2813a50bd669bf8d4d3f58f8e8e581a5e43c3170afcf5723382e3bddaef` | `a864d43754330da77015c0f1588dd7f0b9b1851480295eb16bd102b353b36664` |
| `videomae_large_finetuned__DU145_do__seed2601562.safetensors` | DU145_do | 2601562 | 32 x 4 | 1,215,602,132 B (1.22 GB) | `13b6c0af3d1192f9c74a476930d55e368a791cb3b674eb2adbff20453b2ec5c9` | `75b2e7a5194b78ec3dea92821ac6682892ed6d4bca15a1b2f5e7640b63ff6bcc` |
| `videomae_large_finetuned__DU145_do__seed2601563.safetensors` | DU145_do | 2601563 | 32 x 4 | 1,215,602,132 B (1.22 GB) | `965d52204f2319786ea6031215554acc6f8011c41513f5ed88e59dbd365034ab` | `70a32da8b0d4189a0b357e593fb4d3f9c29731f54820cdb66c0d9daa27369fa2` |
| `videomae_large_finetuned__DU145_do__seed2601564.safetensors` | DU145_do | 2601564 | 32 x 4 | 1,215,602,132 B (1.22 GB) | `c276ff784506169b7f1b7bf835fd2d21afd0a486e2761725986004467808aff7` | `3a5a6033fe6e0baace449bc39b3e51700fda9dff8f17c7c3e535054a8865c98b` |
| `videomae_large_finetuned__DU145_st__seed2601562.safetensors` | DU145_st | 2601562 | 32 x 4 | 1,215,602,132 B (1.22 GB) | `9a9d31022666a4f77880e26161a2b77ecedcc4751e26f711b42b716695dd82e5` | `4e6ad8819edce27b01dff6b664e1df9235c8a3c27a0310d2ff69e0af9bcaf9f9` |
| `videomae_large_finetuned__DU145_st__seed2601563.safetensors` | DU145_st | 2601563 | 16 x 8 | 1,215,602,132 B (1.22 GB) | `7d2cef2c49731f519c9c5efbc2306422e6205743826d37e6a46355ff7a420669` | `ff78e98cbcb5d713d8d106536c5e3a3af614356244126e51f9ccecde8e44d9c2` |
| `videomae_large_finetuned__DU145_st__seed2601564.safetensors` | DU145_st | 2601564 | 32 x 4 | 1,215,602,132 B (1.22 GB) | `9f86ac3c7aefb352cf8c382ee520ec282ff5169eeb76f34aa574f8748321af0b` | `93d7fb0fb6fdd46b7ad9f2008287d1a0701ca4bd312992e6a81d985047d4da60` |
| `videomae_large_finetuned__LNCaP_bp__seed2601562.safetensors` | LNCaP_bp | 2601562 | 32 x 4 | 1,215,602,132 B (1.22 GB) | `8a14ae087bbf4744966fdc562020dfda3fc99ac99986ce2451cc9c3dc55dd7ea` | `466eafcada1f86dc5117b6c05b4eb63f99ab35bb6062648bb2de8b9986ce55c6` |
| `videomae_large_finetuned__LNCaP_bp__seed2601563.safetensors` | LNCaP_bp | 2601563 | 32 x 4 | 1,215,602,132 B (1.22 GB) | `feeb760467fd0a9e26d8229cc91a1b094651fb7dab6025114f1bd53bd1e90b50` | `3190ae7cf65935e3db57345cfc318624aa2df7418a999781afdf2a051901c109` |
| `videomae_large_finetuned__LNCaP_bp__seed2601564.safetensors` | LNCaP_bp | 2601564 | 32 x 4 | 1,215,602,132 B (1.22 GB) | `afb4c69a2cd1d5db2adc92ef58aed203d7c47b854b1c71f196f3b310bf60f475` | `0f60a406331097aafcd03b0d9a57c92835136e2d2ad504f0d23cec2ebcf52326` |
| `videomae_large_finetuned__LNCaP_do__seed2601562.safetensors` | LNCaP_do | 2601562 | 16 x 8 | 1,215,602,132 B (1.22 GB) | `5f3d269d05fb175c02d58701fc0d26083f57ebe63a214c28bffb7d400aa43d05` | `a640a0f34f8f4c14ca20e4ce195f64d7ead559da6ed199cd3517b378aadd10d0` |
| `videomae_large_finetuned__LNCaP_do__seed2601563.safetensors` | LNCaP_do | 2601563 | 32 x 4 | 1,215,602,132 B (1.22 GB) | `93edc6f7c673fe2ff5466fa4b4d077ba96663240195308992963cd67a9d957bd` | `2eda8a27699a7044dae6c3d0f3515e4675f8432e961bb926a8d0535dba4f87e1` |
| `videomae_large_finetuned__LNCaP_do__seed2601564.safetensors` | LNCaP_do | 2601564 | 16 x 8 | 1,215,602,132 B (1.22 GB) | `20ea2758569bebc0549bfb8cb2a8732ae7be50d0bf8f2e7aad5cce61a2004052` | `ae0a72f3b73f52e487e0dfba6a4722f128ab10343cb197c4e6fe479a81e2c49f` |
| `videomae_large_finetuned__LNCaP_st__seed2601562.safetensors` | LNCaP_st | 2601562 | 16 x 8 | 1,215,602,132 B (1.22 GB) | `b8b91592fecd8e755ff112fc4e2da7b4dd8e7cb6547248d5022a9e384fc36fe0` | `942b66cd9b7eb3cbe5b9697c5392ecb62f3a67d0c64af0f396d0ad886fd5edea` |
| `videomae_large_finetuned__LNCaP_st__seed2601563.safetensors` | LNCaP_st | 2601563 | 16 x 8 | 1,215,602,132 B (1.22 GB) | `8623a9eb2b2dee044673c348cf11b3f677d77d938437db6776832b74c785b9dc` | `38f340d7b0e1e710e4ea3106656691afba8af248d14b183e11804a748c979be2` |
| `videomae_large_finetuned__LNCaP_st__seed2601564.safetensors` | LNCaP_st | 2601564 | 16 x 8 | 1,215,602,132 B (1.22 GB) | `fdd8805b6e52d8f493da5f0a06bdbdf2809d688ce564d8637da6ccd1f06f6ddd` | `8abcdd20e160cba93eb907ef20b4080e5da1f87a7f0e1328aa7b6a7c22471c64` |

## DeadAhead survival heads (27 files, LightGBM text models)

LightGBM 4.7.0 multiclass boosters (3 classes, 1,137 inputs), one per held-out group and seed. The input
columns are the features in the order built by `deadahead.survival`.

| File | Held-out group | Seed | Size | sha256 |
|---|---|---|---|---|
| `deadahead_head__PNT1A_bp__seed2601562.txt` | PNT1A_bp | 2601562 | 2,305,192 B | `75066b1a6bb62c1aa63ba76fea9303bcef0bc8004cb9e2bbc387f0587f011bd9` |
| `deadahead_head__PNT1A_bp__seed2601563.txt` | PNT1A_bp | 2601563 | 2,304,713 B | `4bf3e4c9e132d6c0e40f4889b39e9ad4a31a2d803bd874e5c3bfedb9ab908c26` |
| `deadahead_head__PNT1A_bp__seed2601564.txt` | PNT1A_bp | 2601564 | 2,304,459 B | `386d7084e22296848c392751184442a4c82d68b49422658438d323a12a88f3b9` |
| `deadahead_head__PNT1A_do__seed2601562.txt` | PNT1A_do | 2601562 | 2,302,183 B | `657e67ac1e029d411e8846364f3a827acfba49bd8aea9f7ba0d3a705698c6985` |
| `deadahead_head__PNT1A_do__seed2601563.txt` | PNT1A_do | 2601563 | 2,302,146 B | `b7b0c87f2823bb851f1c2c89e0908e47ff558bdf6edec8a16a0e8442c245e70f` |
| `deadahead_head__PNT1A_do__seed2601564.txt` | PNT1A_do | 2601564 | 2,302,637 B | `2cbda9f32f523fdd16e692786e0d9a3a9c26d914d6cd6115900e7e0fb2dcb827` |
| `deadahead_head__PNT1A_st__seed2601562.txt` | PNT1A_st | 2601562 | 2,303,894 B | `ef5711ec88fba7e52cb37aeb40097505b6dc5571bf6c7a3a2aaa5d270167e30e` |
| `deadahead_head__PNT1A_st__seed2601563.txt` | PNT1A_st | 2601563 | 2,303,717 B | `d88f02b669734e86cfe345e2d9580434616dd49c2b80a324954b0360d35916e6` |
| `deadahead_head__PNT1A_st__seed2601564.txt` | PNT1A_st | 2601564 | 2,303,495 B | `fd08ab92729043679072feff42969639919948857c3182e3ec6133cc3fa25fa9` |
| `deadahead_head__DU145_bp__seed2601562.txt` | DU145_bp | 2601562 | 2,303,821 B | `ef36305bfbf121e34354b7fde5ccf85eea333121ac36ad34c6fcd44930232870` |
| `deadahead_head__DU145_bp__seed2601563.txt` | DU145_bp | 2601563 | 2,304,410 B | `47c5498c094b96dd0a9c15e16b205304405e92776558eebccd98156debe42a77` |
| `deadahead_head__DU145_bp__seed2601564.txt` | DU145_bp | 2601564 | 2,305,763 B | `f0baf58a605c1e9e85692985f2c3ac64b1eca2ae2821e5cc5398f916acdbf6d1` |
| `deadahead_head__DU145_do__seed2601562.txt` | DU145_do | 2601562 | 2,301,197 B | `46df68096a1c55e73f3bc9b316a41bd88e993680c983568e6a059b5f3210e651` |
| `deadahead_head__DU145_do__seed2601563.txt` | DU145_do | 2601563 | 2,299,920 B | `34f8eb5f6b7bb3fafba4076ae95559ac94e0c416ca3450ed655dbca59320ebf8` |
| `deadahead_head__DU145_do__seed2601564.txt` | DU145_do | 2601564 | 2,300,737 B | `56478a49be52af229417e8ddead591d69928850c0fb54d41b56d52d301964ab0` |
| `deadahead_head__DU145_st__seed2601562.txt` | DU145_st | 2601562 | 2,302,978 B | `db2e63054f4286be1950a8992e43544e02f0a79a438be1df9d726869ff92935c` |
| `deadahead_head__DU145_st__seed2601563.txt` | DU145_st | 2601563 | 2,303,148 B | `7d012b73e6f0f36d418f59c37ca0431adf501157b8cfe1f3192dd5c1cad0a28b` |
| `deadahead_head__DU145_st__seed2601564.txt` | DU145_st | 2601564 | 2,303,790 B | `39c5d4ced4cd0cb5bdb5526ecf5530bc0b84a6cbc8b4b94b5489453099bb4ebe` |
| `deadahead_head__LNCaP_bp__seed2601562.txt` | LNCaP_bp | 2601562 | 2,303,815 B | `37bba8c592936f645c574c993e9d6112dd50ba1f636d518847a1edf4fbd11e37` |
| `deadahead_head__LNCaP_bp__seed2601563.txt` | LNCaP_bp | 2601563 | 2,302,952 B | `5f8b73eb9a922b7eabba9302d9786c62ecf5eeb88e116ab449223b187905ae3a` |
| `deadahead_head__LNCaP_bp__seed2601564.txt` | LNCaP_bp | 2601564 | 2,303,473 B | `87f695380f64c432fbe4734b030f9acdcbb8f8ecb944b51676d441812d2138e3` |
| `deadahead_head__LNCaP_do__seed2601562.txt` | LNCaP_do | 2601562 | 2,301,107 B | `17d9ef5417fe6e4ba5b4e46f7df7592f35e8decde71af319f7ecc3701d656985` |
| `deadahead_head__LNCaP_do__seed2601563.txt` | LNCaP_do | 2601563 | 2,300,612 B | `75ac72e12073fa8e3f2600816d105a106d9e48ff0215859e6b587662446cdabf` |
| `deadahead_head__LNCaP_do__seed2601564.txt` | LNCaP_do | 2601564 | 2,301,268 B | `5c13ae0ea5ad60802d885ae5ad81e3643c32119f54a4c12d5aeae5f247e890bb` |
| `deadahead_head__LNCaP_st__seed2601562.txt` | LNCaP_st | 2601562 | 2,301,785 B | `c648bca0c19b6403d36a1c209a52123784a541e3d218c0ff0636d755f6e2ef5e` |
| `deadahead_head__LNCaP_st__seed2601563.txt` | LNCaP_st | 2601563 | 2,300,395 B | `9b106c263f31ba9a4f2070ce50e25238e8af64afd1295b2e186575f804b78d65` |
| `deadahead_head__LNCaP_st__seed2601564.txt` | LNCaP_st | 2601564 | 2,301,964 B | `e650ffd9a7e6622a380ced1f6a4de90f0709947ac9d0a8ad8f09db7414b1455e` |

## Loading

Video model, with the package:

```bash
deadahead video-predict --fold DU145_bp --seed 2601562 \
    --weights $DEADAHEAD_DATA/weights/videomae_large_finetuned__DU145_bp__seed2601562.safetensors
deadahead video-features --fold DU145_bp --seed 2601562 \
    --weights $DEADAHEAD_DATA/weights/videomae_large_finetuned__DU145_bp__seed2601562.safetensors
```

Video model, directly:

```python
from safetensors.torch import load_file
from transformers import VideoMAEForVideoClassification

model = VideoMAEForVideoClassification.from_pretrained(
    "MCG-NJU/videomae-large", revision="12da269a02d3e1fbbb7011e610e591fea8061dca",
    num_labels=3, use_mean_pooling=True, attn_implementation="sdpa")
model.load_state_dict(load_file("videomae_large_finetuned__DU145_bp__seed2601562.safetensors"), strict=True)
model.eval()
```

The input is a `(batch, 16, 3, 224, 224)` tensor built by `deadahead.video.training.to_model_input` from
16 frames of 128-px raw phase crops after `phase_to_unit` (see `configs/videomae.yaml`). The released
predictions were computed with bf16 autocast on a GPU; a CPU fp32 pass differs by bf16 rounding
(up to 0.013 in probability on the 8 rows checked).

Survival head, with the package (forecast probabilities of the held-out group, without refitting):

```bash
deadahead train-head --group DU145_bp --seed 2601562 \
    --booster $DEADAHEAD_DATA/weights/deadahead_head__DU145_bp__seed2601562.txt
```

Survival head, directly: the booster gives the hazards of one 30-min bin, so the forecast probabilities (the
cumulative incidence of each route) come from `deadahead.survival.boosters.predict_incidence`:

```python
from deadahead.survival.boosters import load_booster, predict_incidence
from deadahead.survival.config import head_config, windows_config

booster = load_booster("deadahead_head__DU145_bp__seed2601562.txt")
F = predict_incidence(booster, X, head_config(), windows_config())   # X: (n_landmarks, 1136) float32
# F[h]["LYTIC"], F[h]["APOPTOSIS"]: probability of that death within h = 30, 60, 120, 240 min
```

`X` holds the head's inputs without the bin index, in the order built by `deadahead.survival`.

Check a download with `sha256sum <file>` against the tables above.

## Predictions bundles (inputs of `deadahead reproduce` and `deadahead yeast-score`)

| File | Size | sha256 | Licence |
|---|---|---|---|
| `deadahead_bundle_v1.zip` | 211,129,181 bytes | `e99a7a7a6afa290fd6e0c7c0bd9421a988793d7b29c84bf558418e3c69f65060` | CC BY 4.0 |
| `deadahead_yeast_bundle_v1.zip` | 26,647,536 bytes | `1b0de0d4664c92ad9067021d6f5d7416e3840fceb12cd737cb23a742eb06edf2` | CC BY 4.0 |

`deadahead_bundle_v1.zip` unzips to `bundle/` (26 files; `bundle/MANIFEST.json`, sha256 `5d5a955b677f91d4bbae7d4b764832dc0516aeb55ecbf3d836b0f146a0a0d30b`,
lists every file with its size and sha256). It holds the forecast rows, the outcomes and every method's held-out
predictions on CellDeathDetect. `deadahead_yeast_bundle_v1.zip` unzips to `yeast_bundle/` (201 files; its
`MANIFEST.json` has sha256 `0604ce33d10b8ee2857ef0e456f0f21f7d66b46e1a0918c12b69ee94affc5eb2`) and holds the yeast probe's labels, predictions and survival
frames (DetecDiv, Aspert et al., 2022, Zenodo 6078462). Each `MANIFEST.json` also records the source dataset, its
licence and the changes made. Download both from https://zenodo.org/records/23883611 and unzip them into `$DEADAHEAD_DATA`.
`deadahead reproduce` and `deadahead yeast-score` check their bundle against its `MANIFEST.json` and return 3 on a
mismatch; `--no-verify` skips the check, for your own runs arranged in the bundle layout.

The 216 cross-fitted inner boosters behind the training-only alarm thresholds are not released; the bundle holds their
predictions. `train-head --booster` with `--inner-group` therefore needs boosters you trained yourself.

Loading the VideoMAE weights directly needs Transformers 5 (tested with 5.15.1), whose VideoMAE attention layers carry
query, key and value biases.
