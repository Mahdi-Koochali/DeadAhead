# Third-party components and data

| Component | Role in the paper | Licence |
|---|---|---|
| CellDeathDetect (Vicar et al., 2020; Zenodo 2601562) | main dataset; Fig. 1 crops; derived tables and predictions | CC BY 4.0 |
| DetecDiv yeast lifespan data (Aspert et al., 2022; Zenodo 6078462) | exploratory probe (S7) | CC BY 4.0 |
| VideoMAE-L checkpoint `MCG-NJU/videomae-large` (Tong et al., 2022) | video encoder; our fine-tuned weights derive from it | CC BY-NC 4.0 |
| DINOv2 ViT-B/14 (Oquab et al., 2024), loaded with `torch.hub` from facebookresearch/dinov2 at commit 7764ea0f | image baselines | Apache-2.0 |
| DeepHit (Lee et al., 2018); its losses follow pycox, rewritten in PyTorch (not copied) | survival baseline | pycox: BSD-2-Clause |
| LightGBM (Ke et al., 2017) | survival head and image-baseline heads | MIT |
| PyTorch, Hugging Face Transformers | model training and inference | BSD-3-Clause, Apache-2.0 |

Derived tables, figures and predictions built from CellDeathDetect are shared under CC BY 4.0 and were modified from
the original data (tracks, outlines and annotations re-tabulated; model predictions added). Tables and predictions
built from DetecDiv are shared under CC BY 4.0 and were modified from the original data (human per-frame classes
re-tabulated; per-landmark brightness and size measurements computed from the trap videos; model predictions added; no
pixel is redistributed). The fine-tuned VideoMAE weights are shared under CC BY-NC 4.0 because the checkpoint they start
from carries that licence.
