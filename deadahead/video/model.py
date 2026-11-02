"""VideoMAE-large for quantitative phase clips: the pinned checkpoint, model construction, attention
biases, layer-wise learning-rate decay and the phase intensity map.

Token layout: 224 px, patch 16, 16 frames, tubelet 2 give (16 / 2) * (224 / 16)^2 = 1,568 tokens,
ordered time-major (token ``t * 196 + s``). The checkpoint is CC BY-NC 4.0 (VideoMAE, arXiv:2203.12602).
Functions that need torch import it locally, so this module also imports without torch.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


REPO_ID = "MCG-NJU/videomae-large"
#: Full commit hash of the checkpoint revision used for every result.
REVISION = "12da269a02d3e1fbbb7011e610e591fea8061dca"
#: The files the models read.
LOADED_FILES = ("config.json", "preprocessor_config.json", "model.safetensors")

IMAGE_SIZE = 224
PATCH = 16
FRAMES = 16
TUBELET = 2
SPATIAL_TOKENS = (IMAGE_SIZE // PATCH) ** 2          # 196
TEMPORAL_TOKENS = FRAMES // TUBELET                  # 8

#: Tube-masking ratio of the masked-autoencoder adaptation (the checkpoint's own setting).
MASK_RATIO = 0.90

#: Fixed global phase map: ``clip((x - LO) / (HI - LO), 0, 1)`` then ``log1p(9 * lin) / ln 10``.
#: A-priori bounds in radians, fitted to nothing; applied by ``training.phase_to_unit``.
PHASE_LO, PHASE_HI = -0.3, 4.0

#: Output classes of the fine-tuned classifier, in logit order.
CLASSES = ("NONE", "APOPTOSIS", "LYTIC")
#: Parameters of the classifier that the pretrained checkpoint does not supply.
NEW_HEAD = ("fc_norm.", "classifier.")
#: File name of a released model: the group it held out and its seed.
WEIGHTS_NAME = re.compile(r"videomae_large_finetuned__(.+)__seed(\d+)\.safetensors")


class InputError(RuntimeError):
    """Raised instead of reshaping an input into whatever the model would accept."""


def layer_of_parameter(name: str, n_layers: int) -> int:
    """Depth index of a parameter: 0 is the patch embedding, ``n_layers + 1`` the head.

    Layer-wise decay multiplies the base rate by ``decay ** (n_layers + 1 - d)``.
    Names follow ``VideoMAEForVideoClassification``.
    """
    if name.startswith("videomae.embeddings"):
        return 0
    if name.startswith("videomae.encoder.layer."):
        return int(name.split("videomae.encoder.layer.")[1].split(".")[0]) + 1
    return n_layers + 1                       # fc_norm, classifier, and anything new


def layer_decay_groups(named_parameters, *, base_lr: float, head_lr: float, decay: float,
                       weight_decay: float, n_layers: int = 24) -> list[dict]:
    """Optimiser parameter groups for layer-wise learning-rate decay.

    Block l (0..23) gets ``base_lr * decay ** (24 - l)``, the patch embedding ``decay ** 25`` and
    the new head ``head_lr``. Weight decay is the same on every parameter, biases and norms included.
    """
    if not 0 < decay <= 1:
        raise InputError(f"layer decay must be in (0, 1], got {decay}")
    groups: dict[tuple, dict] = {}
    for name, param in named_parameters:
        if not getattr(param, "requires_grad", True):
            continue
        d = layer_of_parameter(name, n_layers)
        is_head = d == n_layers + 1
        if d not in groups:
            lr = head_lr if is_head else base_lr * (decay ** (n_layers + 1 - d))
            groups[d] = {"params": [], "lr": lr, "weight_decay": weight_decay,
                         "layer": d, "name": f"layer{d}"}
        groups[d]["params"].append(param)
    return [groups[k] for k in sorted(groups)]


#: The encoder's final LayerNorm, present in the pretraining model and replaced by ``fc_norm`` in the
#: mean-pooling classifier. The only tensors the adapted encoder may leave behind when it is loaded
#: into the classifier.
ADAPTED_ONLY_KEYS = frozenset({"layernorm.weight", "layernorm.bias"})


def handover_problems(adapted_keys, target_keys, missing, unexpected) -> list[str]:
    """Problems with loading the adapted encoder into the classifier; empty if none.

    ``adapted_keys`` are the keys of the adapted encoder (before fine-tuning); ``missing`` and
    ``unexpected`` come from ``load_state_dict(strict=False)``. Only
    ``ADAPTED_ONLY_KEYS`` may be unexpected, and nothing may be missing.
    """
    problems = []
    if missing:
        problems.append(f"{len(missing)} encoder tensors were not supplied by the adapted encoder, e.g. "
                        f"{sorted(missing)[:3]}")
    extra = set(unexpected) - ADAPTED_ONLY_KEYS
    if extra:
        problems.append(f"{len(extra)} adapted-encoder tensors have no home in the classifier, e.g. "
                        f"{sorted(extra)[:3]}")
    absent = set(adapted_keys) - set(target_keys) - ADAPTED_ONLY_KEYS
    if absent and not extra:
        problems.append(f"{len(absent)} adapted-encoder keys absent from the classifier, e.g. {sorted(absent)[:3]}")
    return problems


def expected_bias_keys(n_encoder: int, n_decoder: int = 0) -> tuple[int, int]:
    """(missing, restored) attention-bias counts when the checkpoint is loaded.

    The checkpoint stores ``q_bias`` and ``v_bias``; the transformers port has ``query.bias``,
    ``key.bias`` and ``value.bias``. Each block reports three bias keys missing and two are restored
    by ``restore_attention_biases``; the key bias stays zero.
    """
    n = n_encoder + n_decoder
    return 3 * n, 2 * n


#: Prefixes of the pretraining head (decoder) that the mean-pooling classifier discards on load.
DISCARDED_BY_CLASSIFIER = ("decoder.", "encoder_to_decoder.", "mask_token")


def load_key_problems(missing, unexpected, *, n_encoder: int, n_decoder: int = 0,
                      new_head_prefixes: tuple[str, ...] = (),
                      discards_pretraining_head: bool = False) -> list[str]:
    """Problems with a ``from_pretrained`` load; empty if it loaded as expected.

    Tolerated: exactly the attention-bias keys missing, keys under ``new_head_prefixes`` missing,
    ``q_bias``/``v_bias`` unexpected, and (for the classifier) the discarded pretraining head.
    """
    want_missing, _ = expected_bias_keys(n_encoder, n_decoder)
    bias = [k for k in missing if k.endswith((".query.bias", ".key.bias", ".value.bias"))]
    head = [k for k in missing if k not in bias and k.startswith(new_head_prefixes)] if new_head_prefixes else []
    other = [k for k in missing if k not in bias and k not in head]
    def tolerated(k: str) -> bool:
        if k.endswith((".q_bias", ".v_bias")):
            return True                                  # the checkpoint's own bias names
        if not discards_pretraining_head:
            return False
        return (k.startswith(DISCARDED_BY_CLASSIFIER)
                or k in {f"videomae.{x}" for x in ADAPTED_ONLY_KEYS})
    bad_unexp = [k for k in unexpected if not tolerated(k)]
    problems = []
    if len(bias) != want_missing:
        problems.append(f"{len(bias)} attention-bias keys missing, expected exactly {want_missing}")
    if other:
        problems.append(f"{len(other)} other keys missing, e.g. {sorted(other)[:3]}")
    if bad_unexp:
        problems.append(f"{len(bad_unexp)} unexpected keys that are neither q/v biases nor part of the "
                        f"discarded pretraining head, e.g. {sorted(bad_unexp)[:3]}")
    return problems


def restore_attention_biases(model, checkpoint_tensors: dict) -> int:
    """Copy each checkpoint ``<attention>.q_bias`` / ``v_bias`` onto ``query.bias`` / ``value.bias``.

    ``from_pretrained`` leaves these biases at zero. ``key.bias`` stays zero, as in the original
    VideoMAE. Returns the number of tensors restored; raises if a copy is not exact.
    """
    import torch
    mods = dict(model.named_modules())
    n = 0
    for key, t in checkpoint_tensors.items():
        if not key.endswith((".q_bias", ".v_bias")):
            continue
        prefix, which = key.rsplit(".", 1)
        att = mods.get(prefix)
        if att is None:                      # e.g. decoder tensors when loading the classifier
            continue
        lin = att.query if which == "q_bias" else att.value
        with torch.no_grad():
            lin.bias.copy_(t)
        if not torch.equal(lin.bias.detach(), t):
            raise AssertionError(f"restored bias differs from the checkpoint: {key}")
        if float(att.key.bias.detach().abs().max()) != 0.0:
            raise AssertionError(f"key.bias is not zero: {prefix}")
        n += 1
    return n


# ------------------------------------------------------------------ checkpoint and construction
def resolve_device(device: str | None = None):
    """The torch device: ``device`` if given, else the GPU; with no GPU the CPU, and a warning."""
    import torch
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("warning: no GPU visible, running in fp32 on the CPU; probabilities differ slightly from the released "
          "bf16 ones; pass --device cpu to silence this", file=sys.stderr)
    return torch.device("cpu")


def check_weights_name(weights: str | Path, fold: str, seed: int) -> None:
    """Raise if a released file's name says it held out another group or used another seed.

    A model that trained on the group it scores would be written under the held-out name. Other names pass.
    """
    name = Path(weights).name
    m = WEIGHTS_NAME.fullmatch(name)
    if m and (m[1] != fold or int(m[2]) != seed):
        raise InputError(f"{name} is the model of {m[1]} with seed {m[2]}, not {fold} with seed {seed}")


def pretrained_snapshot(cache_dir: str | Path | None = None) -> Path:
    """Local directory of the pinned checkpoint revision, downloaded once from the Hugging Face Hub.

    Only ``LOADED_FILES`` are fetched. Offline use works once they are cached (``HF_HUB_OFFLINE=1``).
    """
    from huggingface_hub import snapshot_download
    path = snapshot_download(repo_id=REPO_ID, revision=REVISION, allow_patterns=list(LOADED_FILES),
                             cache_dir=None if cache_dir is None else str(cache_dir))
    snap = Path(path)
    for name in LOADED_FILES:
        if not (snap / name).is_file():
            raise InputError(f"{name} is missing from the checkpoint snapshot {snap}")
    return snap


def image_stats(snap: str | Path) -> tuple[list, list]:
    """ImageNet mean and std from the snapshot's ``preprocessor_config.json``."""
    pre = json.loads((Path(snap) / "preprocessor_config.json").read_text())
    return pre["image_mean"], pre["image_std"]


def checkpoint_tensors(snap: str | Path) -> dict:
    """Every tensor of the snapshot's ``model.safetensors``, on the CPU."""
    from safetensors import safe_open
    with safe_open(str(Path(snap) / "model.safetensors"), "pt") as f:
        return {k: f.get_tensor(k) for k in f.keys()}


def build_pretraining_model(snap: str | Path, ck: dict | None = None):
    """``VideoMAEForPreTraining`` (encoder and decoder) from the snapshot, attention biases restored.

    Set the torch seed before calling. Raises if the load reports anything but the expected keys.
    """
    from transformers import VideoMAEForPreTraining
    ck = checkpoint_tensors(snap) if ck is None else ck
    model, info = VideoMAEForPreTraining.from_pretrained(str(snap), attn_implementation="sdpa",
                                                         output_loading_info=True)
    n_enc, n_dec = model.config.num_hidden_layers, model.config.decoder_num_hidden_layers
    problems = load_key_problems(info.get("missing_keys", []), info.get("unexpected_keys", []),
                                 n_encoder=n_enc, n_decoder=n_dec)
    if problems:
        raise InputError("the checkpoint did not load as expected: " + "; ".join(problems))
    restored = restore_attention_biases(model, ck)
    if restored != expected_bias_keys(n_enc, n_dec)[1]:
        raise InputError(f"{restored} attention biases restored, expected {expected_bias_keys(n_enc, n_dec)[1]}")
    return model


def build_classifier(snap: str | Path, ck: dict | None = None):
    """Mean-pooling ``VideoMAEForVideoClassification`` with three classes, attention biases restored.

    The new ``fc_norm`` and ``classifier`` are initialised from the current torch seed.
    """
    from transformers import VideoMAEForVideoClassification
    ck = checkpoint_tensors(snap) if ck is None else ck
    model, info = VideoMAEForVideoClassification.from_pretrained(
        str(snap), num_labels=3, use_mean_pooling=True, attn_implementation="sdpa",
        output_loading_info=True)
    n_enc = model.config.num_hidden_layers
    problems = load_key_problems(info.get("missing_keys", []), info.get("unexpected_keys", []),
                                 n_encoder=n_enc, n_decoder=0, new_head_prefixes=NEW_HEAD,
                                 discards_pretraining_head=True)
    if problems:
        raise InputError("the checkpoint did not load as expected: " + "; ".join(problems))
    restored = restore_attention_biases(model, ck)
    if restored != expected_bias_keys(n_enc)[1]:
        raise InputError(f"{restored} attention biases restored, expected {expected_bias_keys(n_enc)[1]}")
    return model


def load_adapted_encoder(model, adapted: str | Path) -> int:
    """Load an adapted encoder (``adapt__<group>.pt``) into a classifier's ``videomae``; every kept
    tensor must match exactly. Returns the number of tensors loaded."""
    import torch
    sd = torch.load(adapted, map_location="cpu", weights_only=False)["state_dict"]
    missing, unexpected = model.videomae.load_state_dict(sd, strict=False)
    target = model.videomae.state_dict()
    problems = handover_problems(sd.keys(), target.keys(), missing, unexpected)
    bad = [k for k, v in sd.items()
           if k in target and not torch.equal(target[k].detach().cpu(), v.detach().cpu())]
    if bad:
        problems.append(f"{len(bad)} tensors differ after loading, e.g. {bad[:3]}")
    if problems:
        raise InputError("the adapted encoder did not load exactly: " + "; ".join(problems))
    return sum(1 for k in sd if k in target)


def load_classifier(ckpt: Path, snap: Path, dev):
    """The classifier built as in training, with every tensor of a ``.pt`` checkpoint loaded strictly."""
    import torch
    from transformers import VideoMAEForVideoClassification
    model = VideoMAEForVideoClassification.from_pretrained(
        str(snap), num_labels=3, use_mean_pooling=True, attn_implementation="sdpa")
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(sd, strict=True)
    return model.to(dev).eval()


def load_released_classifier(weights: str | Path, snap: str | Path, dev):
    """The classifier with a released ``.safetensors`` file loaded strictly (``docs/WEIGHTS.md``)."""
    from safetensors.torch import load_file
    from transformers import VideoMAEForVideoClassification
    model = VideoMAEForVideoClassification.from_pretrained(
        str(snap), num_labels=3, use_mean_pooling=True, attn_implementation="sdpa")
    model.load_state_dict(load_file(str(weights)), strict=True)
    return model.to(dev).eval()
