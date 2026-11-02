"""End-to-end run of the VideoMAE pipeline on synthetic data with a 2-layer random VideoMAE (CPU).

Builds a crop store, clip indexes, then runs adaptation, fine-tuning, selection scoring, held-out
prediction and feature export through their ``main`` entry points; a run stopped halfway and resumed; and
the commands' refusal of files left by a different run.
"""

from __future__ import annotations

import json
import signal
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
safetensors_torch = pytest.importorskip("safetensors.torch")

pytestmark = pytest.mark.torch

GROUPS = ("AAA_bp", "AAA_do", "AAA_st")
HELD_OUT = "AAA_bp"
N_FRAMES = 60
CROP = 32
TRACKS = ("1", "2", "3")
LANDMARKS = (40, 50)


def tiny_snapshot(root):
    """A random 2-layer VideoMAE saved in the original checkpoint layout (q_bias / v_bias, no key bias)."""
    from safetensors.torch import load_file, save_file
    from transformers import VideoMAEConfig, VideoMAEForPreTraining
    cfg = VideoMAEConfig(image_size=224, patch_size=16, num_channels=3, num_frames=16, tubelet_size=2,
                         hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                         qkv_bias=True, use_mean_pooling=False, decoder_num_attention_heads=2,
                         decoder_hidden_size=16, decoder_num_hidden_layers=1, decoder_intermediate_size=32,
                         norm_pix_loss=True)
    torch.manual_seed(0)
    model = VideoMAEForPreTraining(cfg)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith((".query.bias", ".value.bias")):
                p.normal_()
            elif name.endswith(".key.bias"):
                p.zero_()
    model.save_pretrained(root)
    sd = load_file(str(root / "model.safetensors"))
    out = {}
    for k, v in sd.items():
        if k.endswith(".key.bias"):
            continue
        k = k.replace(".query.bias", ".q_bias").replace(".value.bias", ".v_bias")
        out[k] = v.contiguous()
    save_file(out, str(root / "model.safetensors"), metadata={"format": "pt"})
    (root / "preprocessor_config.json").write_text(json.dumps(
        {"image_mean": [0.485, 0.456, 0.406], "image_std": [0.229, 0.224, 0.225]}))
    return root, out


def synthetic_data(data):
    """Windows table and all-frames crop store for three groups of one movie each."""
    rng = np.random.default_rng(1)
    store = data / "crops" / "frames128_all"
    store.mkdir(parents=True)
    rows = []
    labels = {"AAA_bp": ["LYTIC", "NO_EVENT", "APOPTOSIS"], "AAA_do": ["LYTIC", "LYTIC", "APOPTOSIS"],
              "AAA_st": ["LYTIC", "APOPTOSIS", "CENSORED_IN_HORIZON"]}
    for g in GROUPS:
        movie = f"{g}_1"
        idx = [(movie, i + 1, tr, f, r) for r, (i, tr, f) in enumerate(
            (i, tr, f) for f in range(N_FRAMES) for i, tr in enumerate(TRACKS))]
        px = rng.uniform(-0.3, 4.0, size=(len(idx), CROP, CROP)).astype(np.float16)
        np.save(store / f"{movie}.npy", px)
        pd.DataFrame(idx, columns=["movie_id", "cell_num", "track_id", "frame_index", "row"]).assign(
            border_clipped=False).to_parquet(store / f"{movie}.parquet", index=False)
        for j, tr in enumerate(TRACKS):
            for t in LANDMARKS:
                lab = labels[g][j] if t == LANDMARKS[1] else "NO_EVENT"
                for h in (30, 60):
                    rows.append({"movie_id": movie, "track_id": tr, "t_obs_frame": t, "panel": "primary_439",
                                 "group": g, "history_start_frame": t - 40, "history_end_frame": t,
                                 "horizon_min": h, "label": lab, "window_weight": 0.5 + 0.1 * j})
    (data / "tables").mkdir()
    pd.DataFrame(rows).to_parquet(data / "tables" / "windows.parquet", index=False)


def tiny_project(tmp_path, monkeypatch, split=(2, 1)):
    """Synthetic data, clip indexes and a tiny checkpoint, with training batches of ``split``."""
    from deadahead.video import training as T
    from deadahead.video.clips import main as clips_main
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("DEADAHEAD_DATA", str(data))
    snap, ck = tiny_snapshot(tmp_path / "snap")
    synthetic_data(data)
    monkeypatch.setattr(T, "split_batch", lambda micro_batch, effective=8: T.Split(*split))
    assert clips_main([]) == 0
    return data, snap, ck


class StopAfterTwoSteps:
    """Stands in for ``training.Interrupt``: the flag, read once per step, turns true on its second read."""

    def __init__(self):
        self.reads = 0

    @property
    def hit(self):
        self.reads += 1
        return self.reads == 2


def test_offsets_and_tube_mask():
    from deadahead.video import training as T
    from deadahead.video.clips import clip_offsets
    offs = clip_offsets()
    assert offs[0] == 0 and offs[-1] == 40 and len(set(offs)) == 16
    m = T.tube_mask(3, np.random.default_rng(0))
    assert m.shape == (3, 1568) and T.is_tube(m) and int(m[0].sum()) == 176 * 8


def test_layer_decay_groups():
    from deadahead.video.model import layer_decay_groups
    names = ["videomae.embeddings.patch_embeddings.projection.weight", "videomae.encoder.layer.0.output.dense.weight",
             "videomae.encoder.layer.23.output.dense.weight", "fc_norm.weight", "classifier.weight"]
    params = [(n, torch.nn.Parameter(torch.zeros(1))) for n in names]
    g = layer_decay_groups(params, base_lr=1.0, head_lr=5.0, decay=0.5, weight_decay=0.05)
    lr = {d["layer"]: d["lr"] for d in g}
    assert lr[0] == 0.5 ** 25 and lr[1] == 0.5 ** 24 and lr[24] == 0.5 and lr[25] == 5.0


def test_pipeline_end_to_end(tmp_path, monkeypatch):
    from deadahead.survival.train import load_features
    from deadahead.video import adapt, features, finetune, predict
    from deadahead.video import selection as select
    from deadahead.video.clips import ClipGatherer
    from deadahead.video.model import build_classifier

    data, snap, ck = tiny_project(tmp_path, monkeypatch, split=(4, 2))
    monkeypatch.setattr(select, "FOLD_ORDER", GROUPS)
    torch.set_num_threads(4)

    clips_dir = data / "runs" / "videomae" / "clips"
    idx = pd.read_parquet(clips_dir / f"clips__{HELD_OUT}.parquet")
    assert HELD_OUT not in set(idx["group"]) and len(idx) == 12

    # adaptation: 2 steps; the finished file removed, the last saved state gives the same encoder again
    assert adapt.main(["--fold", HELD_OUT, "--steps", "2", "--ckpt-every", "1", "--snapshot", str(snap)]) == 0
    enc = data / "runs" / "videomae" / "adapt" / f"adapt__{HELD_OUT}.pt"
    first = torch.load(enc, weights_only=False)["state_dict"]
    assert not any(k.startswith("decoder") for k in first)
    enc.unlink()
    assert adapt.main(["--fold", HELD_OUT, "--steps", "2", "--snapshot", str(snap)]) == 0
    again = torch.load(enc, weights_only=False)["state_dict"]
    assert all(torch.equal(first[k], again[k]) for k in first)

    # the adapted encoder's features, its read-out through the bare encoder
    assert features.main(["--fold", HELD_OUT, "--adapted", "--snapshot", str(snap)]) == 0
    fdir = data / "runs" / "videomae" / "features"
    held = load_features(fdir / "adapted" / f"{HELD_OUT}.npy", len(pd.read_parquet(fdir / "rows.parquet")))
    assert held.shape == (18, 32)
    enc_model = features.load_adapted(enc, snap, torch.device("cpu"))
    assert enc_model.layernorm is not None
    inp = features.model_inputs(ClipGatherer(data / "crops" / "frames128_all"),
                                features.rows_with_clips(data / "tables" / "windows.parquet",
                                                         data / "crops" / "frames128_all"),
                                np.arange(2), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], torch.device("cpu"))
    with torch.no_grad():
        direct = enc_model(pixel_values=inp).last_hidden_state.mean(dim=1).numpy()
    # 18 rows in the command's pass, 2 here: the batch and the thread count move the last float32 places
    np.testing.assert_allclose(direct, held[:2], rtol=0, atol=1e-6)

    # the restored attention biases equal the checkpoint's
    torch.manual_seed(0)
    clf = build_classifier(snap, ck)
    q = clf.videomae.encoder.layer[0].attention.attention
    assert torch.equal(q.query.bias, ck["videomae.encoder.layer.0.attention.attention.q_bias"])
    assert float(q.key.bias.abs().max()) == 0.0

    common = ["--fold", HELD_OUT, "--backbone-lr", "2e-4", "--layer-decay", "0.9", "--steps", "2",
              "--snapshot", str(snap)]
    assert finetune.main(common) == 0
    assert finetune.main(common + ["--final"]) == 0
    ft_dir = data / "runs" / "videomae" / "finetune"
    final = ft_dir / f"ft__{HELD_OUT}__lr0.0002_ld0.9__final__s2601562.pt"
    assert final.is_file()
    meta = json.loads((ft_dir / f"ft__{HELD_OUT}__lr0.0002_ld0.9__s2601562.json").read_text())
    assert meta["selection_group"] == "AAA_do" and "AAA_do" not in meta["training_groups"]

    assert select.main(["score", "--fold", HELD_OUT, "--backbone-lr", "2e-4", "--layer-decay", "0.9",
                        "--snapshot", str(snap)]) == 0
    sel_dir = data / "runs" / "videomae" / "select"
    for lr, ld in select.GRID[:3]:
        select.score_path(sel_dir, HELD_OUT, lr, ld).write_text(json.dumps({"inner_lytic_auprc": None}))
    assert select.main(["choose", "--fold", HELD_OUT]) == 0
    chosen = json.loads((sel_dir / f"selected__{HELD_OUT}.json").read_text())
    assert chosen["winner"]["tag"] == "lr0.0002_ld0.9"

    assert predict.main(["--fold", HELD_OUT, "--seed", "2601562", "--backbone-lr", "2e-4",
                         "--layer-decay", "0.9", "--snapshot", str(snap)]) == 0
    pred = pd.read_parquet(data / "runs" / "videomae" / "pred" / f"{HELD_OUT}__s2601562.parquet")
    assert list(pred.columns) == list(predict.PRED_COLUMNS) and len(pred) == 6
    assert np.allclose(pred[["p_none", "p_apoptosis", "p_lytic"]].sum(axis=1), 1.0)

    # the same checkpoint as safetensors gives the same predictions
    sd = torch.load(final, weights_only=False)["state_dict"]
    st = tmp_path / "w.safetensors"
    safetensors_torch.save_file({k: v.contiguous() for k, v in sd.items()}, str(st))
    out2 = tmp_path / "pred2"
    assert predict.main(["--fold", HELD_OUT, "--seed", "2601562", "--weights", str(st), "--snapshot", str(snap),
                         "--out-dir", str(out2)]) == 0
    pred2 = pd.read_parquet(out2 / f"{HELD_OUT}__s2601562.parquet")
    np.testing.assert_allclose(pred2["p_lytic"].to_numpy(), pred["p_lytic"].to_numpy(), rtol=0, atol=1e-6)

    assert features.main(["--fold", HELD_OUT, "--seed", "2601562", "--weights", str(st),
                          "--snapshot", str(snap)]) == 0
    f = np.load(fdir / f"{HELD_OUT}__s2601562.npy")
    lg = np.load(fdir / f"{HELD_OUT}__s2601562.logits.npy")
    rows = pd.read_parquet(fdir / "rows.parquet")
    assert f.shape == (18, 32) and f.dtype == np.float32 and lg.shape == (18, 3) and len(rows) == 18
    model = transformers.VideoMAEForVideoClassification.from_pretrained(
        str(snap), num_labels=3, use_mean_pooling=True, attn_implementation="sdpa")
    model.load_state_dict(sd)
    with torch.no_grad():
        assert np.allclose(model.classifier(torch.from_numpy(f)).numpy(), lg, atol=1e-5)


def test_stopped_and_resumed_run_equals_a_straight_run(tmp_path, monkeypatch):
    """Exit code 10, then the weights of an uninterrupted run, bit for bit, in both training stages. One thread,
    because the same comparison at two threads on a loaded CPU differed in the last bits."""
    from deadahead.video import adapt, finetune
    from deadahead.video import training as T
    _, snap, _ = tiny_project(tmp_path, monkeypatch)
    torch.set_num_threads(1)
    base = ["--fold", HELD_OUT, "--steps", "4", "--ckpt-every", "0", "--snapshot", str(snap)]
    ft = base + ["--backbone-lr", "2e-4", "--layer-decay", "0.9", "--adapt-dir", str(tmp_path / "adapt_straight")]
    ft_name = finetune.checkpoint_path(tmp_path, HELD_OUT, 2e-4, 0.9, finetune.SEED, final=False).name
    for stage, main, args, name in (("adapt", adapt.main, base, f"adapt__{HELD_OUT}.pt"),
                                    ("finetune", finetune.main, ft, ft_name)):
        straight, stopped = tmp_path / f"{stage}_straight", tmp_path / f"{stage}_stopped"
        assert main(args + ["--out-dir", str(straight)]) == 0
        with monkeypatch.context() as m:
            m.setattr(T, "Interrupt", StopAfterTwoSteps)
            assert main(args + ["--out-dir", str(stopped)]) == 10
        assert not (stopped / name).exists()
        state = torch.load((stopped / name).with_suffix(".state.pt"), weights_only=False)
        assert state["step"] == 2 and len(state["losses"]) == 2
        assert main(args + ["--out-dir", str(stopped)]) == 0
        a = torch.load(straight / name, weights_only=False)["state_dict"]
        b = torch.load(stopped / name, weights_only=False)["state_dict"]
        assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a), stage


def test_a_different_command_refuses_the_files_of_an_earlier_run(tmp_path, monkeypatch, capsys):
    from deadahead.video import adapt, finetune
    from deadahead.video.model import InputError
    data, snap, _ = tiny_project(tmp_path, monkeypatch)
    cmd = ["--fold", HELD_OUT, "--snapshot", str(snap)]
    ad = data / "runs" / "videomae" / "adapt"
    enc = ad / f"adapt__{HELD_OUT}.pt"

    assert adapt.main(cmd + ["--steps", "2"]) == 0
    capsys.readouterr()
    assert adapt.main(cmd + ["--steps", "2"]) == 0                      # the same command
    assert "nothing to do" in capsys.readouterr().out
    assert adapt.main(cmd + ["--steps", "4"]) == 3
    assert adapt.main(cmd + ["--steps", "2", "--seed", "5"]) == 3
    assert "trained differently" in capsys.readouterr().err
    assert json.loads(enc.with_suffix(".json").read_text())["steps"] == 2

    enc.rename(tmp_path / "kept.pt")                                    # only the state file is left
    with pytest.raises(InputError, match="another run"):
        adapt.main(cmd + ["--steps", "4"])
    (tmp_path / "kept.pt").rename(enc)

    ft = cmd + ["--backbone-lr", "2e-4", "--layer-decay", "0.9"]
    ckpt = finetune.checkpoint_path(data / "runs" / "videomae" / "finetune", HELD_OUT, 2e-4, 0.9, finetune.SEED,
                                    final=False)
    assert finetune.main(ft + ["--steps", "2"]) == 0
    assert finetune.main(ft + ["--steps", "2"]) == 0
    assert finetune.main(ft + ["--steps", "4"]) == 3
    changed = torch.load(enc, weights_only=False)                       # another adapted encoder, same name
    changed["state_dict"]["layernorm.weight"] += 1.0
    torch.save(changed, enc)
    assert finetune.main(ft + ["--steps", "2"]) == 3
    assert "adapted_sha256" in capsys.readouterr().err
    ckpt.unlink()
    with pytest.raises(InputError, match="another run"):
        finetune.main(ft + ["--steps", "2"])


class ClipBlindLogits(torch.nn.Module):
    """Stands in for the classifier: logits that ignore the clip, so a step's loss can be worked out by hand."""

    config = SimpleNamespace(num_hidden_layers=2)

    def __init__(self, logits):
        super().__init__()
        self.classifier = torch.nn.Parameter(torch.tensor(logits))

    def forward(self, pixel_values):
        return SimpleNamespace(logits=self.classifier.expand(len(pixel_values), -1))


@pytest.mark.parametrize("micro_batch,grad_accum", [(8, 1), (4, 2), (2, 4)])
def test_the_loss_weights_each_row_by_its_window_weight(tmp_path, monkeypatch, micro_batch, grad_accum):
    """The first step's loss is the window-weighted mean of the class-weighted cross-entropy over the rows it drew,
    for every split of the batch; a uniform weight gives the plain mean, at any scale."""
    from deadahead.video import finetune
    from deadahead.video import training as T

    logits, cls_w, seed = [0.7, -0.4, 0.2], np.array([0.5, 1.0, 2.0]), 3
    y = np.array([0, 0, 1, 1, 2, 2])
    monkeypatch.setattr(finetune, "ClipGatherer", lambda store: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(T, "clip_batch", lambda gather, movies, rows, pick, *_: torch.zeros(len(pick), 16, 2, 2))

    def first_loss(weights):
        data = pd.DataFrame({"movie_id": "m", "frame_rows": 0, "y": y, "window_weight": weights})
        losses = finetune.train(ClipBlindLogits(logits), data, tmp_path, [0.5] * 3, [0.5] * 3, torch.device("cpu"),
                                steps=1, split=T.Split(micro_batch, grad_accum), seed=seed, cls_w=cls_w,
                                state=tmp_path / "state.pt", run={}, backbone_lr=1e-3, layer_decay=0.9,
                                ckpt_every=0, log=lambda s: None)
        return losses[0]

    pick = np.random.default_rng(seed).integers(0, len(y), size=8)
    per_class = cls_w * -torch.log_softmax(torch.tensor(logits, dtype=torch.float64), 0).numpy()
    weights = np.array([1.0, 0.1, 0.1, 0.3, 0.9, 0.2], dtype=np.float32)
    by_hand = (weights[pick] * per_class[y[pick]]).sum() / weights[pick].sum()
    assert first_loss(weights) == pytest.approx(by_hand, rel=1e-5)
    plain = per_class[y[pick]].mean()
    assert first_loss(np.full(6, 0.25, dtype=np.float32)) == pytest.approx(plain, rel=1e-5)
    assert abs(by_hand - plain) > 0.05


def test_interrupt_without_sigusr1(monkeypatch):
    from deadahead.video import training as T
    monkeypatch.delattr(signal, "SIGUSR1")
    stop = T.Interrupt()
    assert not stop.hit
    signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
    assert stop.hit


def test_a_checkpoint_without_its_record_is_refused(tmp_path, capsys):
    from deadahead.video import training as T
    ckpt = tmp_path / "adapt__G.pt"
    ckpt.write_bytes(b"")
    assert T.finished_run(ckpt, {"steps": 2}) == 3
    assert "trained differently" in capsys.readouterr().err
    ckpt.with_suffix(".json").write_text(json.dumps({"steps": 2}))
    assert T.finished_run(ckpt, {"steps": 2}) == 0


def test_missing_inputs_name_the_command_that_builds_them(tmp_path, monkeypatch, capsys):
    from deadahead.video import adapt, features, finetune, predict
    monkeypatch.setenv("DEADAHEAD_DATA", str(tmp_path))
    assert adapt.main(["--fold", HELD_OUT]) == 3
    assert "deadahead video-clips" in capsys.readouterr().err
    assert finetune.main(["--fold", HELD_OUT, "--backbone-lr", "2e-4", "--layer-decay", "0.9"]) == 3
    assert "deadahead video-adapt" in capsys.readouterr().err
    assert predict.main(["--fold", HELD_OUT, "--seed", "1", "--weights", "w.safetensors"]) == 3
    assert "deadahead data-windows" in capsys.readouterr().err
    assert features.main(["--fold", HELD_OUT, "--seed", "1", "--weights", "w.safetensors"]) == 3
    assert "deadahead data-windows" in capsys.readouterr().err


def test_cpu_fallback_warns_unless_asked_for(monkeypatch, capsys):
    from deadahead.video.model import resolve_device
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device(None).type == "cpu"
    assert "no GPU visible" in capsys.readouterr().err
    assert resolve_device("cpu").type == "cpu"
    assert capsys.readouterr().err == ""


def test_crop_extraction(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    from deadahead.video.crops import crop_one, extract_movie
    img = np.arange(100, dtype=np.float32).reshape(10, 10)
    c, clipped = crop_one(img, 1.0, 1.0, 4, 0.0)
    assert clipped and c[0, 0] == 0.0 and c[1, 1] == img[0, 0]
    stack = np.random.default_rng(0).normal(size=(3, 40, 40)).astype(np.float32)
    tifffile.imwrite(tmp_path / "m.tif", stack, photometric="minisblack")
    cells = pd.DataFrame({"movie_id": "m", "cell_num": [1, 2, 1], "track_id": ["a", "b", "a"],
                          "frame_index": [0, 0, 2], "centroid_y_px": [20.0, 5.0, 21.0],
                          "centroid_x_px": [20.0, 5.0, 19.0]})
    arr, df = extract_movie(str(tmp_path / "m.tif"), "m", cells, 16, 1, 0.0)
    assert arr.shape == (3, 16, 16) and arr.dtype == np.float16
    assert list(df["row"]) == [0, 1, 2] and list(df["border_clipped"]) == [False, True, False]
    assert np.array_equal(arr[0], stack[0, 12:28, 12:28].astype(np.float16))


def test_config_matches_constants():
    import yaml

    from deadahead import paths
    from deadahead.video import adapt, clips, crops, finetune, model, predict
    from deadahead.video import selection as select
    from deadahead.video import training as T
    cfg = yaml.safe_load((paths.REPO_ROOT / "configs" / "videomae.yaml").read_text())
    assert cfg["checkpoint"]["repo_id"] == model.REPO_ID and cfg["checkpoint"]["revision"] == model.REVISION
    assert tuple(cfg["checkpoint"]["files"]) == model.LOADED_FILES
    assert cfg["batch"]["effective"] == T.EFFECTIVE_BATCH and cfg["batch"]["micro_batch"] == adapt.MICRO_BATCH
    assert finetune.MICRO_BATCH == adapt.MICRO_BATCH and finetune.WARMUP_FRACTION == adapt.WARMUP_FRACTION == 0.05
    i = cfg["input"]
    assert (i["crop_px"], i["frames"], i["history_frames"], i["image_size"]) == (
        crops.SIZE, model.FRAMES, clips.HISTORY_FRAMES, model.IMAGE_SIZE)
    assert model.FRAMES == clips.CLIP_FRAMES and T.AUG_STREAM_OFFSET == 1 and T.n_masked() == 176
    assert cfg["predict"]["batch"] == predict.PRED_BATCH
    a, f = cfg["adapt"], cfg["finetune"]
    assert (a["steps"], a["lr"], a["weight_decay"], a["mask_ratio"], a["seed"]) == (
        adapt.STEPS, adapt.LR, adapt.WEIGHT_DECAY, model.MASK_RATIO, adapt.SEED)
    assert (f["steps"], f["head_lr"], f["weight_decay"], f["grad_clip_global_norm"], f["horizon_min"]) == (
        finetune.STEPS, finetune.HEAD_LR, finetune.WEIGHT_DECAY, finetune.GRAD_CLIP, finetune.HORIZON)
    assert (f["panel"], tuple(f["classes"]), tuple(f["excluded_movies"])) == (
        finetune.PANEL, model.CLASSES, finetune.EXCLUDED_MOVIES)
    assert finetune.PANEL == clips.PANEL
    assert sorted({g[0] for g in finetune.GRID}) == sorted(f["backbone_lr"])
    assert sorted({g[1] for g in finetune.GRID}) == sorted(f["layer_decay"])
    assert [tuple(c) for c in cfg["select"]["order"]] == finetune.GRID
    assert tuple(cfg["select"]["final_seeds"]) == finetune.FINAL_SEEDS and cfg["select"]["grid_seed"] == select.GRID_SEED
    assert cfg["input"]["phase_map"]["lo"] == model.PHASE_LO and cfg["input"]["phase_map"]["hi"] == model.PHASE_HI


def test_pretrained_snapshot_pins_revision(tmp_path, monkeypatch):
    import huggingface_hub

    from deadahead.video import model
    for name in model.LOADED_FILES:
        (tmp_path / name).write_text("{}")
    seen = {}

    def fake(**kw):
        seen.update(kw)
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)
    assert model.pretrained_snapshot() == tmp_path
    assert seen["repo_id"] == "MCG-NJU/videomae-large" and seen["revision"] == model.REVISION
    assert set(seen["allow_patterns"]) == set(model.LOADED_FILES)
