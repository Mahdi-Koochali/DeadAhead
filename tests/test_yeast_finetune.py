"""The yeast fine-tuning loop and predictions on a synthetic trap-video store with a 2-layer random VideoMAE (CPU).

Both starts train 2 steps through ``deadahead.yeast.finetune.main``, then predict through
``deadahead.yeast.predict.main``; the zero-shot pass of a 3-class model in the released format is checked too.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
safetensors_torch = pytest.importorskip("safetensors.torch")

from test_video_tiny import tiny_snapshot  # noqa: E402

pytestmark = pytest.mark.torch

N_FRAMES = 30
LANDMARKS = (6, 12, 18, 24, 29)
HELD_OUT = "Pos16_7"
SEED = 2601562


def synthetic_yeast(data):
    """Two traps per position: a (T, 60, 59, 3) uint8 video each, the landmark keys, the final row files and the
    per-fold inputs."""
    from deadahead.yeast import constants as C
    rng = np.random.default_rng(3)
    ydir = data / "yeast"
    (ydir / "vid_store").mkdir(parents=True)
    (ydir / "rows").mkdir()
    keys = []
    for p in C.POSITIONS:
        for k in (1, 2):
            trap = f"{p}_{k}"
            np.save(ydir / "vid_store" / f"{trap}.npy",
                    rng.integers(0, 256, size=(N_FRAMES, *C.CROP_SHAPE), dtype=np.uint8))
            keys += [{"unit": trap, "position": p, "t": t} for t in LANDMARKS]
    keys = pd.DataFrame(keys)
    keys.to_parquet(ydir / "rows" / "all_keys.parquet", index=False)
    for p in C.POSITIONS:
        tr = keys[keys["position"].isin(C.training_positions(p))].reset_index(drop=True)
        tr.assign(y=(np.arange(len(tr)) % 3 == 0).astype(int)).to_parquet(ydir / "rows" / f"final__{p}.parquet",
                                                                           index=False)
    cfg = {"folds": {p: {"channel": i % 3, "m_D": 0.5, "s_D": 0.29} for i, p in enumerate(C.POSITIONS)},
           "source_moments": {"m_F": 0.27, "s_F": 0.15}}
    (data / "yeast.json").write_text(json.dumps(cfg))
    return keys, data / "yeast.json"


def released_start(data, snap, ck, seed):
    """A random 3-class classifier saved as a released fine-tuned VideoMAE file (safetensors, full state dict)."""
    from deadahead.video.model import build_classifier
    torch.manual_seed(7)
    clf = build_classifier(snap, ck)
    path = data / "weights" / f"videomae_large_finetuned__PNT1A_bp__seed{seed}.safetensors"
    path.parent.mkdir(parents=True)
    safetensors_torch.save_file({k: v.contiguous() for k, v in clf.state_dict().items()}, str(path))
    return path


def test_plan_of_positions():
    from deadahead.yeast import constants as C
    from deadahead.yeast.finetune import plan_of, run_name
    f = plan_of("final", "videomae_public", "Pos3_3", 2601563)
    assert f["predict_positions"] == ("Pos3_3",) and f["embed_positions"] == C.POSITIONS
    assert "Pos3_3" not in f["train_positions"] and len(f["train_positions"]) == 6
    t = plan_of("thr", "videomae_finetuned", "Pos3_3")
    assert t["predict_positions"] == ("Pos16_7",) and len(t["train_positions"]) == 5
    assert set(t["embed_positions"]) == set(C.POSITIONS) - {"Pos3_3"}
    le = plan_of("le", "videomae_finetuned", "Pos3_3")
    assert len(le["train_positions"]) == 3 and le["embed_positions"] == ()
    assert run_name("final", "videomae_public", "Pos3_3", 2601563) == "videomae_public__Pos3_3__s2601563"
    assert run_name("le", "videomae_public", "Pos3_3", SEED) == "le__videomae_public__Pos3_3"
    with pytest.raises(ValueError):
        plan_of("thr", "videomae_public", "Pos3_3", 2601563)


def test_finetune_and_predict(tmp_path, monkeypatch):
    from deadahead.video import training as T
    from deadahead.yeast import finetune, predict

    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("DEADAHEAD_DATA", str(data))
    monkeypatch.setattr(T, "split_batch", lambda micro_batch, effective=8: T.Split(4, 2))
    torch.set_num_threads(3)
    snap, ck = tiny_snapshot(tmp_path / "snap")
    keys, cfg = synthetic_yeast(data)
    released = released_start(data, snap, ck, SEED)
    runs = data / "runs" / "yeast"
    held = keys[keys["position"] == HELD_OUT].reset_index(drop=True)

    for start in ("videomae_finetuned", "videomae_public"):
        name = f"{start}__{HELD_OUT}__s{SEED}"
        args = ["--start", start, "--heldout", HELD_OUT, "--seed", str(SEED), "--steps", "2", "--ckpt-every", "1",
                "--snapshot", str(snap), "--config", str(cfg), "--device", "cpu"]
        assert finetune.main(args) == 0
        pred = pd.read_parquet(runs / "pred" / f"{name}.parquet")
        assert list(pred.columns) == ["unit", "position", "t", "p_none", "p_death"]
        assert pred[["unit", "position", "t"]].equals(held)
        assert np.isfinite(pred[["p_none", "p_death"]].to_numpy()).all()
        assert np.allclose(pred["p_none"] + pred["p_death"], 1.0)
        weights = runs / "weights" / f"{name}.pt"
        summary = json.loads(weights.with_suffix(".json").read_text())
        assert summary["all_steps_finite"] and len(summary["losses"]) == 2
        emb = np.load(runs / "embed" / f"{name}.npy")
        assert emb.shape == (len(keys), 32) and np.isfinite(emb).all()
        # the resumable state goes once the weights are saved
        assert weights.is_file() and not (runs / "weights" / f"{name}.state.pt").exists()

        # a re-run with the prediction present does nothing; without it, the pass reruns from the saved weights
        assert finetune.main(args) == 0
        (runs / "pred" / f"{name}.parquet").unlink()
        assert finetune.main(args) == 0
        again = pd.read_parquet(runs / "pred" / f"{name}.parquet")
        # multithreaded CPU reductions may change the last bits between runs, so compare to float32 precision
        np.testing.assert_allclose(again["p_death"].to_numpy(), pred["p_death"].to_numpy(), rtol=0, atol=1e-6)

        out = tmp_path / f"predict__{name}.parquet"
        assert predict.main(["--weights", str(weights), "--heldout", HELD_OUT, "--snapshot", str(snap),
                             "--config", str(cfg), "--device", "cpu", "--out", str(out)]) == 0
        p2 = pd.read_parquet(out)
        assert p2[["unit", "position", "t"]].equals(held)
        np.testing.assert_allclose(p2["p_death"].to_numpy(), pred["p_death"].to_numpy(), rtol=0, atol=1e-6)

    # the start weights were loaded: the encoder of the trained model moved away from them, but its head is 2-class
    sd = torch.load(runs / "weights" / f"videomae_finetuned__{HELD_OUT}__s{SEED}.pt", weights_only=False)["state_dict"]
    assert tuple(sd["classifier.weight"].shape) == (2, 32)

    # zero-shot pass of the released 3-class file on every position
    out = tmp_path / "zero_shot.parquet"
    assert predict.main(["--zero-shot", "--weights", str(released), "--snapshot", str(snap), "--config", str(cfg),
                         "--device", "cpu", "--out", str(out)]) == 0
    z = pd.read_parquet(out)
    assert len(z) == len(keys) and set(z["read_with_fold"]) == set(keys["position"])
    assert np.allclose(z["p_death"], z["p_apoptosis"] + z["p_lytic"])
    assert np.allclose(z[["p_none", "p_apoptosis", "p_lytic"]].sum(axis=1), 1.0)


def tiny_run(tmp_path, monkeypatch):
    """Arguments of a 2-step run on the synthetic store, and its output files."""
    from deadahead.video import training as T
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("DEADAHEAD_DATA", str(data))
    monkeypatch.setattr(T, "split_batch", lambda micro_batch, effective=8: T.Split(4, 2))
    torch.set_num_threads(3)
    snap, _ = tiny_snapshot(tmp_path / "snap")
    _, cfg = synthetic_yeast(data)
    name = f"videomae_public__{HELD_OUT}__s{SEED}"
    runs = data / "runs" / "yeast"
    args = ["--start", "videomae_public", "--heldout", HELD_OUT, "--steps", "2", "--snapshot", str(snap),
            "--config", str(cfg), "--device", "cpu"]
    return args, {"pred": runs / "pred" / f"{name}.parquet", "embed": runs / "embed" / f"{name}.npy",
                  "weights": runs / "weights" / f"{name}.pt"}


def test_a_crash_before_the_prediction_reruns_the_pass(tmp_path, monkeypatch):
    from deadahead.yeast import finetune
    args, f = tiny_run(tmp_path, monkeypatch)

    def killed(*_):
        raise OSError("killed")
    with monkeypatch.context() as m:
        m.setattr(finetune.np, "save", killed)                 # dies while the embeddings are written
        with pytest.raises(OSError):
            finetune.main(args)
    assert f["weights"].is_file() and not f["pred"].exists() and not f["embed"].exists()
    assert finetune.main(args) == 0
    assert f["pred"].is_file() and f["embed"].is_file()


def test_a_lost_embedding_comes_back_with_the_prediction(tmp_path, monkeypatch):
    from deadahead.yeast import finetune
    args, f = tiny_run(tmp_path, monkeypatch)
    assert finetune.main(args) == 0
    f["embed"].unlink()
    f["pred"].unlink()
    assert finetune.main(args) == 0
    assert f["embed"].is_file() and f["pred"].is_file()


def test_a_non_finite_pass_exits_5_and_leaves_no_prediction(tmp_path, monkeypatch):
    from deadahead.yeast import finetune
    from deadahead.yeast import model as M
    args, f = tiny_run(tmp_path, monkeypatch)
    real = M.predict

    def nan_predict(*a, **k):
        probs, emb = real(*a, **k)
        probs[0, 0] = np.nan
        return probs, emb
    with monkeypatch.context() as m:
        m.setattr(M, "predict", nan_predict)
        assert finetune.main(args) == 5
    assert f["weights"].is_file() and not f["pred"].exists()
    assert finetune.main(args) == 0 and f["pred"].is_file()


def test_a_different_command_refuses_the_files_of_an_earlier_run(tmp_path, monkeypatch, capsys):
    from deadahead.video import training as T
    from deadahead.video.model import InputError
    from deadahead.yeast import finetune
    args, f = tiny_run(tmp_path, monkeypatch)
    four = [("4" if v == "2" else v) for v in args]
    other = ["--out-dir", str(tmp_path / "other")]

    assert finetune.main(args) == 0
    capsys.readouterr()
    assert finetune.main(args) == 0                             # the same command
    assert "nothing to do" in capsys.readouterr().out
    assert finetune.main(four) == 3                             # finished predictions of 2 steps
    err = capsys.readouterr().err
    assert "trained differently" in err and "steps 2, not 4" in err and str(f["pred"]) in err
    f["pred"].unlink()
    assert finetune.main(four) == 3                             # the weights alone
    assert "trained differently" in capsys.readouterr().err
    assert json.loads(f["weights"].with_suffix(".json").read_text())["steps"] == 2
    assert finetune.main(args) == 0 and f["pred"].is_file()     # the pass of the same run is still open

    class StopNow:
        hit = True
    with monkeypatch.context() as m:
        m.setattr(T, "Interrupt", StopNow)
        assert finetune.main(args + other) == 10
    with pytest.raises(InputError, match="another run"):        # a state file of 2 steps, schedule included
        finetune.main(four + other)
    assert finetune.main(args + other) == 0


def test_missing_inputs_name_the_command_that_builds_them(tmp_path, capsys):
    from deadahead.yeast import finetune, predict
    empty = tmp_path / "rows"
    empty.mkdir()
    assert finetune.main(["--start", "videomae_public", "--heldout", HELD_OUT, "--rows-dir", str(empty),
                          "--out-dir", str(tmp_path / "out")]) == 3
    assert "yeast-data" in capsys.readouterr().err
    weights = tmp_path / "w.pt"
    weights.touch()
    assert predict.main(["--weights", str(weights), "--heldout", HELD_OUT, "--rows-dir", str(empty)]) == 3
    assert "yeast-data" in capsys.readouterr().err


def test_videomae_finetuned_start_loads_the_released_body(tmp_path):
    from deadahead.yeast import model as M
    snap, ck = tiny_snapshot(tmp_path / "snap")
    released = released_start(tmp_path, snap, ck, SEED)
    model, notes = M.build_model("videomae_finetuned", SEED, snap, torch.device("cpu"), start_weights=released)
    src = safetensors_torch.load_file(str(released))
    tgt = model.state_dict()
    assert notes["loaded_tensors"] == len(src) - 2
    assert all(torch.equal(tgt[k], v) for k, v in src.items() if not k.startswith("classifier."))
    assert tuple(tgt["classifier.weight"].shape) == (2, 32)
    # the new head is drawn after the seed: the same seed gives the same head
    again, _ = M.build_model("videomae_finetuned", SEED, snap, torch.device("cpu"), start_weights=released)
    assert torch.equal(again.classifier.weight, model.classifier.weight)
    with pytest.raises(M.ModelLoadError):
        M.build_model("videomae_finetuned", SEED, snap, torch.device("cpu"))


def test_mapped_batch_equals_the_numpy_map(tmp_path):
    from deadahead.yeast import constants as C
    from deadahead.yeast import data as D
    from deadahead.yeast import model as M
    rng = np.random.default_rng(5)
    (tmp_path / "s").mkdir()
    np.save(tmp_path / "s" / "Pos3_3_1.npy", rng.integers(0, 256, size=(30, *C.CROP_SHAPE), dtype=np.uint8))
    store = D.VidStore(tmp_path / "s")
    fold, src_m = {"channel": 2, "m_D": 0.41, "s_D": 0.13}, {"m_F": 0.27, "s_F": 0.15}
    frames = [D.clip_frames(t) for t in (6, 29)]
    x = M.mapped_batch(store, np.array(["Pos3_3_1", "Pos3_3_1"]), frames, fold, src_m, torch.device("cpu"))
    ref = D.to_unit(store.batch(["Pos3_3_1", "Pos3_3_1"], frames, 2), 0.41, 0.13, 0.27, 0.15)
    assert x.shape == (2, 16, 60, 59) and float(x.min()) >= 0.0 and float(x.max()) <= 1.0
    assert np.allclose(x.numpy(), ref, atol=1e-6)


def test_interrupted_run_resumes_to_the_same_weights(tmp_path, monkeypatch):
    """Bit for bit, so one thread: at more, the last bits of a run depend on the load."""
    from deadahead.video import training as T
    from deadahead.yeast import finetune

    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("DEADAHEAD_DATA", str(data))
    monkeypatch.setattr(T, "split_batch", lambda micro_batch, effective=8: T.Split(4, 2))
    torch.set_num_threads(1)
    snap, _ = tiny_snapshot(tmp_path / "snap")
    _, cfg = synthetic_yeast(data)
    name = f"videomae_public__{HELD_OUT}__s{SEED}"
    args = ["--start", "videomae_public", "--heldout", HELD_OUT, "--steps", "2", "--snapshot", str(snap),
            "--config", str(cfg), "--device", "cpu"]
    assert finetune.main(args + ["--out-dir", str(tmp_path / "straight")]) == 0

    class StopNow:
        hit = True
    monkeypatch.setattr(T, "Interrupt", StopNow)
    assert finetune.main(args + ["--out-dir", str(tmp_path / "resumed")]) == 10
    assert (tmp_path / "resumed" / "weights" / f"{name}.state.pt").is_file()
    monkeypatch.undo()
    monkeypatch.setenv("DEADAHEAD_DATA", str(data))
    monkeypatch.setattr(T, "split_batch", lambda micro_batch, effective=8: T.Split(4, 2))
    assert finetune.main(args + ["--out-dir", str(tmp_path / "resumed")]) == 0
    a = torch.load(tmp_path / "straight" / "weights" / f"{name}.pt", weights_only=False)["state_dict"]
    b = torch.load(tmp_path / "resumed" / "weights" / f"{name}.pt", weights_only=False)["state_dict"]
    assert all(torch.equal(a[k], b[k]) for k in a)
    pa = pd.read_parquet(tmp_path / "straight" / "pred" / f"{name}.parquet")
    pb = pd.read_parquet(tmp_path / "resumed" / "pred" / f"{name}.parquet")
    assert np.array_equal(pa["p_death"].to_numpy(), pb["p_death"].to_numpy())
