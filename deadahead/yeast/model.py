"""VideoMAE-large on yeast trap videos: the intensity map, the two-class model of each start, and the forward pass.

* ``mapped_batch``: stored uint8 frames of the fold's channel mapped affinely into the [0, 1] input range of the
  quantitative-phase model; training then applies ``deadahead.video.training.augment_clip`` (before the resize) and
  both training and prediction apply ``to_model_input`` (bilinear resize to 224, three channels, ImageNet
  normalisation).
* ``logits_and_embedding``: the classifier's own forward, split so that the 1,024-d pooled embedding after
  ``fc_norm`` comes out of the same pass as the logits.
* ``build_model``: the two starts. ``videomae_finetuned``: a released fine-tuned VideoMAE-large, its 3-class head
  left out (``fc_norm`` kept). ``videomae_public``: the pinned public weights, attention biases restored, ``fc_norm``
  new. In both, the torch seed is set before the model and its new 1,024 -> 2 head are built.
Functions that need torch import it locally, so this module also imports without torch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import constants as C
from . import data as D

NEW_HEAD = ("fc_norm.", "classifier.")
#: The keys of the 3-class head that the yeast model replaces.
START_HEAD_KEYS = ("classifier.weight", "classifier.bias")


class ModelLoadError(RuntimeError):
    pass


def mapped_batch(store: D.VidStore, traps, frame_rows, fold: dict, src_m: dict, dev):
    """(B, 16, 60, 59) uint8 frames to [0, 1]: x' = clip((x / 255 - m_D) / s_D * s_F + m_F, 0, 1), on ``dev``."""
    import torch
    raw = store.batch(traps, frame_rows, int(fold["channel"]))                  # (B, 16, 60, 59) uint8
    x = torch.from_numpy(np.ascontiguousarray(raw)).to(dev, non_blocking=True).float() / 255.0
    x = (x - float(fold["m_D"])) / float(fold["s_D"]) * float(src_m["s_F"]) + float(src_m["m_F"])
    return x.clamp(0.0, 1.0)


def logits_and_embedding(model, pixel_values):
    """The classifier's forward with mean pooling: the encoder, the mean over the 1,568 tokens, fc_norm, the linear
    head; the pooled vector is returned too."""
    seq = model.videomae(pixel_values=pixel_values)[0]
    emb = model.fc_norm(seq.mean(1)) if model.fc_norm is not None else seq[:, 0]
    return model.classifier(emb), emb


def predict(model, store: D.VidStore, keys, fold: dict, src_m: dict, mean, std, dev, *, batch: int = C.PRED_BATCH,
            want_embedding: bool = False):
    """Eval mode, bf16 autocast on CUDA, batches of 64, no augmentation: (softmax probabilities, embeddings or None)
    in the row order of ``keys`` (unit, t)."""
    import torch

    from deadahead.video import training as T
    model.eval()
    traps = keys["unit"].to_numpy()
    frames = [D.clip_frames(t) for t in keys["t"].to_numpy()]
    n = len(keys)
    probs = None
    emb = np.empty((n, model.config.hidden_size), dtype=np.float32) if want_embedding else None
    with torch.no_grad():
        for s in range(0, n, batch):
            pick = np.arange(s, min(s + batch, n))
            x = T.to_model_input(mapped_batch(store, traps[pick], [frames[i] for i in pick], fold, src_m, dev),
                                 mean, std)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                logits, e = logits_and_embedding(model, x)
            p = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            if probs is None:
                probs = np.empty((n, p.shape[1]), dtype=np.float64)
            probs[pick] = p
            if want_embedding:
                emb[pick] = e.float().cpu().numpy()
    return probs, emb


def released_weights(seed: int, weights_dir: Path | None = None, group: str = C.START_GROUP) -> Path:
    """The released fine-tuned VideoMAE-large of ``group`` for ``seed`` (``docs/WEIGHTS.md``)."""
    from deadahead import paths
    d = Path(weights_dir) if weights_dir is not None else paths.data_root() / "weights"
    return d / f"videomae_large_finetuned__{group}__seed{seed}.safetensors"


def read_state_dict(path: Path) -> dict:
    """Tensors of a ``.safetensors`` file, or the ``state_dict`` of a ``.pt`` checkpoint, on the CPU."""
    import torch
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(path))
    return torch.load(path, map_location="cpu", weights_only=False)["state_dict"]


def build_model(start: str, seed: int, snap: Path, dev, *, start_weights: Path | None = None):
    """The 2-class model of a start, seeded before it is built. Returns (model, notes)."""
    import torch
    from transformers import VideoMAEForVideoClassification

    from deadahead.video.model import (ADAPTED_ONLY_KEYS, DISCARDED_BY_CLASSIFIER, checkpoint_tensors,
                                       expected_bias_keys, load_key_problems, restore_attention_biases)
    torch.manual_seed(seed)                                                  # before the model and its head
    model, info = VideoMAEForVideoClassification.from_pretrained(
        str(snap), num_labels=len(C.CLASSES), use_mean_pooling=True, attn_implementation="sdpa",
        output_loading_info=True)
    n_enc = model.config.num_hidden_layers
    notes = {"start": start}
    if start == "videomae_public":
        unexpected = info.get("unexpected_keys", [])
        probs = load_key_problems(info.get("missing_keys", []), unexpected, n_encoder=n_enc, n_decoder=0,
                                  new_head_prefixes=NEW_HEAD, discards_pretraining_head=True)
        if probs:
            raise ModelLoadError("the public checkpoint did not load as expected: " + "; ".join(probs))
        ck = checkpoint_tensors(snap)
        restored = restore_attention_biases(model, ck)
        if restored != expected_bias_keys(n_enc)[1]:
            raise ModelLoadError(f"{restored} attention biases restored, expected {expected_bias_keys(n_enc)[1]}")
        notes.update(restored_bias_keys=restored, missing_keys=len(info.get("missing_keys", [])),
                     discarded=sum(1 for k in unexpected if k.startswith(DISCARDED_BY_CLASSIFIER)
                                   or k in {f"videomae.{x}" for x in ADAPTED_ONLY_KEYS}),
                     init="the pinned public VideoMAE-large weights; fc_norm and the head new")
    elif start == "videomae_finetuned":
        if start_weights is None:
            raise ModelLoadError("the videomae_finetuned start needs the released fine-tuned weights")
        sd = read_state_dict(start_weights)
        body = {k: v for k, v in sd.items() if k not in START_HEAD_KEYS}
        missing, unexpected = model.load_state_dict(body, strict=False)
        if sorted(missing) != sorted(START_HEAD_KEYS) or unexpected:
            raise ModelLoadError(f"start weights: missing {sorted(missing)[:4]}, unexpected {sorted(unexpected)[:4]}"
                                 f"; expected exactly the 3-class head missing")
        tgt = model.state_dict()
        bad = [k for k, v in body.items() if not torch.equal(tgt[k].detach().cpu(), v.detach().cpu())]
        if bad:
            raise ModelLoadError(f"{len(bad)} tensors differ after loading the start weights, e.g. {bad[:3]}")
        notes.update(loaded_tensors=len(body), head_left_out=list(START_HEAD_KEYS),
                     init=f"{Path(start_weights).name}; its 3-class head left out; fc_norm kept")
    else:
        raise ModelLoadError(f"unknown start {start!r}")
    return model.to(dev), notes


def load_yeast_classifier(weights: Path, snap: Path, dev):
    """A trained 2-class yeast model (``state_dict`` of ``deadahead.yeast.finetune``), loaded strictly."""
    from transformers import VideoMAEForVideoClassification
    model = VideoMAEForVideoClassification.from_pretrained(
        str(snap), num_labels=len(C.CLASSES), use_mean_pooling=True, attn_implementation="sdpa")
    model.load_state_dict(read_state_dict(weights), strict=True)
    return model.to(dev).eval()
