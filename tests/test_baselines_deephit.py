"""DeepHit on synthetic data: targets, losses, selection, resume from a checkpoint and the command end to end."""

import json

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.torch

from deadahead.baselines import deephit as D
from deadahead.paths import REPO_ROOT


def test_survival_targets_by_hand():
    tte = np.array([3.0, 30.0, 33.0, 240.0, 243.0, 600.0, 45.0, 30.0])
    term = np.array(["APOPTOSIS", "LYTIC", "CENSORED", "APOPTOSIS", "LYTIC", "CENSORED", "UNTYPED_DEATH", "CENSORED"])
    L, k = D.survival_targets(tte, term, 30.0, 8)
    assert list(L) == [1, 1, 1, 8, 8, 8, 1, 0]
    assert list(k) == [1, 2, 0, 1, 0, 0, 0, 0]
    assert list(D.l0_mask(L)) == [True] * 7 + [False]


def test_cif_monotone_and_comparable_pairs():
    logits = torch.randn(5, D.N_OUT)
    F = D.cif_from_logits(logits)
    assert F.shape == (5, 2, 8) and (F[:, :, 1:] >= F[:, :, :-1] - 1e-7).all()
    assert (F[:, 0, -1] + F[:, 1, -1] <= 1 + 1e-6).all()
    L = torch.tensor([2, 3, 2, 5])
    k = torch.tensor([1, 0, 2, 1])
    A = D.comparable(L, k, 1)
    # i = 0 (apoptosis in bin 2): j = 1 (none, L 3 >= 2), j = 3 (typed, L 5 > 2); not j = 2 (a same-bin death)
    assert A[0].tolist() == [0.0, 1.0, 0.0, 1.0] and A[1].sum() == 0 and A[2].sum() == 0


def test_nll_matches_a_direct_computation():
    torch.manual_seed(0)
    logits = torch.randn(4, D.N_OUT, dtype=torch.float64)
    L = torch.tensor([1, 3, 8, 8])
    k = torch.tensor([1, 0, 2, 0])
    w = torch.tensor([1.0, 2.0, 0.5, 1.5], dtype=torch.float64)
    p = torch.softmax(logits, dim=1)
    terms = []
    for i in range(4):
        if k[i] > 0:
            terms.append(-torch.log(p[i, (k[i] - 1) * 8 + L[i] - 1]))
        else:
            surv = 1 - sum(p[i, c * 8: c * 8 + L[i]].sum() for c in range(2))
            terms.append(-torch.log(surv))
    want = (w * torch.stack(terms)).sum() / w.sum()
    assert torch.allclose(D.nll(logits, L, k, w), want)
    assert torch.isfinite(D.rank_loss(logits.float(), L, k, 0.1))


def synthetic_arrays(n=400, d=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    risk = 1 / (1 + np.exp(-(X[:, 0] * 2 - 1)))
    k = np.where(rng.random(n) < risk * 0.5, np.where(X[:, 1] > 0, 1, 2), 0)
    L = np.where(k > 0, rng.integers(1, 9, n), rng.integers(0, 9, n))
    w = rng.uniform(0.5, 1.5, n)
    return X, L.astype(np.int64), k.astype(np.int64), w


def test_selection_then_final_and_choice():
    torch.set_num_threads(1)
    X, L, k, w = synthetic_arrays()
    tr, inn = np.arange(300), np.arange(300, 400)
    s = D.train(X[tr], L[tr], k[tr], w[tr], D.GRID[1], 1, X_in=X[inn], L_in=L[inn], k_in=k[inn], w_in=w[inn])
    assert s["e_star"] is not None and 1 <= s["e_star"] <= s["epochs_run"] <= D.MAX_EPOCHS
    assert s["epochs_run"] - s["e_star"] <= D.PATIENCE and s["n_l0_removed"] == int((L[tr] == 0).sum())
    pred = D.predict(s["model"], X[inn], s["standardiser"], state=s["best_state"])
    assert pred["F_APOPTOSIS"].shape == (100, 8) and np.isfinite(D.inner_auprc(pred, L[inn], k[inn]))
    f = D.train(X[tr], L[tr], k[tr], w[tr], D.GRID[1], 1, epochs=s["e_star"])
    assert f["epochs_run"] == s["e_star"] and "e_star" not in f
    assert D.choose_setting([{"no_improvement": True, "inner_auprc": 0.9}, {"no_improvement": False, "inner_auprc": 0.3},
                             {"no_improvement": False, "inner_auprc": 0.3}, {"no_improvement": False,
                                                                           "inner_auprc": float("nan")}]) == 1
    assert D.choose_setting([{"no_improvement": True, "inner_auprc": 0.5}]) is None


def test_resume_from_checkpoint_equals_one_run(tmp_path):
    torch.set_num_threads(1)
    X, L, k, w = synthetic_arrays(n=200)
    ck = tmp_path / "fit.pt"
    first = D.train(X, L, k, w, D.GRID[3], 5, epochs=4, ckpt=ck, stop_after=2)
    assert first["interrupted"] and first["epoch"] == 2
    resumed = D.train(X, L, k, w, D.GRID[3], 5, epochs=4, ckpt=ck)
    whole = D.train(X, L, k, w, D.GRID[3], 5, epochs=4)
    assert resumed["hist"]["train"] == whole["hist"]["train"]
    a, b = D.predict(resumed["model"], X, resumed["standardiser"]), D.predict(whole["model"], X, whole["standardiser"])
    assert all(np.array_equal(a[c], b[c]) for c in a)
    with pytest.raises(D.CheckpointMismatch):
        D.train(X, L, k, w, D.GRID[2], 5, epochs=4, ckpt=ck)


def test_parameter_count_and_config():
    assert D.n_params(D.DeepHit(1136)) == D.PARAMS[1136] and D.n_params(D.DeepHit(2416)) == D.PARAMS[2416]
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "baselines.yaml").read_text())["deephit"]
    assert [tuple(x) for x in cfg["grid"]] == list(D.GRID)
    assert (cfg["selection"]["max_epochs"], cfg["selection"]["patience"], cfg["batch"]) == (D.MAX_EPOCHS, D.PATIENCE,
                                                                                          D.BATCH)
    assert cfg["network"]["outputs"] == D.N_OUT and cfg["network"]["dropout"] == D.DROPOUT


def test_command_end_to_end(tmp_path, monkeypatch):
    from tests.test_survival_head import synthetic_tables, write_inputs
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):    # main pins them; restored afterwards
        monkeypatch.setenv(v, "1")
    tables, hist = synthetic_tables(tracks_per_group=6)
    d = write_inputs(tmp_path, tables, hist)
    tables["windows"].assign(label=np.where(tables["windows"]["track_terminal_event"] == "LYTIC", "LYTIC", "NONE")) \
        .to_parquet(d / "windows.parquet")
    args = ["--group", "LNCaP_do", "--seed", "3", "--tables-dir", str(d), "--history", str(tmp_path / "history.parquet"),
            "--features", str(tmp_path / "feats.npy"), "--rows", str(tmp_path / "rows.parquet"), "--threads", "1",
            "--out-dir", str(tmp_path / "dh")]
    assert D.main(args + ["--item", "sel2"]) == 0
    assert D.main(args) == 0
    unit = D.unit_name("LNCaP_do", 3)
    rec = json.loads((tmp_path / "dh" / f"{unit}.json").read_text())
    if rec["status"] == "done":
        p = np.load(tmp_path / "dh" / "pred" / f"{unit}.npz")
        held = tables["windows"][tables["windows"]["group"] == "LNCaP_do"].drop_duplicates(list(D.KEY))
        n = len(p["movie_id"])                                   # the held-out landmarks with a known outcome
        assert 0 < n <= len(held) and p["F_LYTIC"].shape == (n, 8) and p["extra_mass"].shape == (n,)
        assert set(p["track_id"]) <= set(held["track_id"])
        assert (np.diff(p["F_LYTIC"], axis=1) >= -1e-7).all()
    assert len(list((tmp_path / "dh" / "sel").glob(f"{unit}__sel*.json"))) == 8
    assert D.main(args) == 0                                     # finished: nothing is refitted
