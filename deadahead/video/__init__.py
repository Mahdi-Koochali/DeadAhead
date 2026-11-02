"""The fine-tuned VideoMAE-large pipeline.

crops (128-px all-frames crop store) -> clips (16-frame clip indexes) -> adapt (masked-autoencoder
adaptation per held-out group) -> finetune (four settings, then the chosen one for three seeds) ->
selection (selection-group scoring and choice) -> predict (held-out probabilities) -> features
(1,024-d per-row features) -> test_time (test-time changes). Recipe: ``configs/videomae.yaml``.
"""
