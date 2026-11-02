"""Budding-yeast death probe on DetecDiv (Aspert et al. 2022, eLife, doi:10.7554/eLife.79519; data: Zenodo record
6078462, CC BY 4.0).

data (landmark rows, row files, pixel store, per-fold inputs, picture tables) -> finetune (VideoMAE fine-tuned on six
positions from the released fine-tuned VideoMAE or from the public weights) -> predict -> rules (the simple rules on
each held-out position) -> score (the readings, from the stored predictions in the yeast bundle).
Recipe and per-fold inputs: ``configs/yeast.yaml``.
"""
