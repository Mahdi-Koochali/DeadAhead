"""The DeadAhead survival head.

frames (landmark rows, training and evaluation labels, 1,024 VideoMAE + 112 handcrafted inputs) -> head
(discrete-time competing-risks LightGBM, 8 bins of 30 min) -> boosters (cumulative incidence from a saved booster).
``train`` fits one held-out group and seed, or one cross-fitted inner head; ``censoring`` runs the injected-censoring
experiment. Settings: ``configs/survival_head.yaml``, ``configs/windows.yaml``.
"""
