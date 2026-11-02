"""DINOv2 baselines on synthetic data: the tile pool, then adaptation, embedding and fine-tuning end to end with a
small randomly initialised ViT in place of DINOv2 ViT-B/14 (CPU, no download)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

GROUPS = ("AAA_bp", "AAA_do", "AAA_st")
HELD_OUT = "AAA_bp"
N_FRAMES = 60
STRIDE = 5
TRACKS = (1, 2, 3)


def _torch():
    return pytest.importorskip("torch")


def write_movies(data, n_pages=4, side=120):
    """Phase stacks and the movie table; the held-out group's stack is never written, so reading it fails."""
    tifffile = pytest.importorskip("tifffile")
    raw = data / "raw" / "celldeathdetect"
    raw.mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows = []
    for g in GROUPS:
        line, treat = g.split("_")
        movie = f"{g}_1"
        if g != HELD_OUT:
            stack = rng.uniform(-0.5, 4.5, size=(n_pages, side, side)).astype(np.float32)
            tifffile.imwrite(raw / f"QPI_{movie}.tif", stack, photometric="minisblack")
        rows.append({"movie_id": movie, "cell_line": line, "treatment": treat, "fov": 1,
                     "image_path": f"QPI_{movie}.tif"})
    (data / "tables").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(data / "tables" / "movies.csv", index=False)


def write_tables_and_crops(data, crop=32):
    """A stride-5 crop store and the windows, events and movies tables of three tracks per group."""
    store = data / "crops" / "frames128"
    store.mkdir(parents=True)
    rng = np.random.default_rng(1)
    win, ev, mv = [], [], []
    death = {1: ("LYTIC", 48), 2: ("APOPTOSIS", 47), 3: ("NO_EVENT", None)}
    for g in GROUPS:
        line, treat = g.split("_")
        movie = f"{g}_1"
        frames = list(range(0, N_FRAMES, STRIDE))
        idx = [(movie, c, f"{movie}:{c}", f, r) for r, (f, c) in enumerate((f, c) for f in frames for c in TRACKS)]
        np.save(store / f"{movie}.npy", rng.uniform(-0.3, 4.0, size=(len(idx), crop, crop)).astype(np.float16))
        pd.DataFrame(idx, columns=["movie_id", "cell_num", "track_id", "frame_index", "row"]).to_parquet(
            store / f"{movie}.parquet", index=False)
        mv.append({"movie_id": movie, "num_frames": N_FRAMES})
        for c in TRACKS:
            tid = f"{movie}:{c}"
            kind, t = death[c]
            ev.append({"track_id": tid, "event_type": kind, "t_event": np.nan if t is None else float(t)})
            for t_obs in (40,):
                for h in (30, 60):
                    win.append({"movie_id": movie, "track_id": tid, "t_obs_frame": t_obs, "panel": "primary_439",
                                "group": g, "cell_line": line, "treatment": treat, "horizon_min": h,
                                "window_weight": 0.5 + 0.1 * c})
    (data / "tables").mkdir(exist_ok=True)
    pd.DataFrame(win).to_parquet(data / "tables" / "windows.parquet", index=False)
    pd.DataFrame(ev).to_parquet(data / "tables" / "events.parquet", index=False)
    pd.DataFrame(mv).to_parquet(data / "tables" / "movies.parquet", index=False)
    return store


def tiny_vit_class(torch):
    class TinyViT(torch.nn.Module):
        """A 3-block ViT with the attributes the baselines use: blocks, norm, embed_dim; returns the class token."""

        def __init__(self, dim=768, depth=3, patch=14):
            super().__init__()
            self.embed_dim = dim
            self.patch_embed = torch.nn.Conv2d(3, dim, patch, patch)
            self.cls_token = torch.nn.Parameter(torch.randn(1, 1, dim) * 0.02)
            self.blocks = torch.nn.ModuleList(
                [torch.nn.TransformerEncoderLayer(dim, 2, 32, dropout=0.0, batch_first=True) for _ in range(depth)])
            self.norm = torch.nn.LayerNorm(dim)

        def forward(self, x):
            t = self.patch_embed(x).flatten(2).transpose(1, 2)
            t = torch.cat([self.cls_token.expand(len(t), -1, -1), t], 1)
            for b in self.blocks:
                t = b(t)
            return self.norm(t)[:, 0]

    return TinyViT


# ------------------------------------------------------------------ without torch
def test_tiles_and_resize():
    pytest.importorskip("PIL")
    pytest.importorskip("tifffile")
    from deadahead.baselines.dinov2_adapt import SIZE, _resize01, _tile
    img = np.zeros((300, 280), dtype=np.float32)
    tiles = _tile(img, np.random.default_rng(0), 20)
    assert len(tiles) == 20 and all(t.shape[0] == t.shape[1] and 96 <= t.shape[0] < 260 for t in tiles)
    r = _resize01(np.linspace(-1, 2, 100 * 100).reshape(100, 100))
    assert r.shape == (SIZE, SIZE) and r.dtype == np.float16 and r.min() >= 0 and r.max() <= 1


def test_fold_pool_leaves_out_the_held_out_group(tmp_path, monkeypatch):
    from deadahead.baselines import dinov2_adapt as AD
    data = tmp_path / "data"
    monkeypatch.setenv("DEADAHEAD_DATA", str(data))
    write_movies(data)
    monkeypatch.setattr(AD, "TARGET", 30)          # 15 tiles per training movie: one per frame of 4 frames
    inv = AD.movie_table()
    assert list(inv["group"]) == list(GROUPS)
    pool = AD.fold_pool(inv, HELD_OUT)             # the held-out stack does not exist; opening it would fail
    assert pool.shape == (2 * 4, AD.SIZE, AD.SIZE) and pool.dtype == np.float16
    assert float(pool.min()) >= 0.0 and float(pool.max()) <= 1.0
    assert np.array_equal(pool, AD.fold_pool(inv, HELD_OUT))
    with pytest.raises(Exception):
        AD.fold_pool(inv, "AAA_do")


# ------------------------------------------------------------------ with torch
@pytest.mark.torch
def test_hub_entry_is_pinned(monkeypatch):
    torch = _torch()
    from deadahead.baselines import dinov2_common as C
    seen = {}

    def fake_load(repo, name, **kw):
        seen.update(repo=repo, name=name, **kw)
        return torch.nn.Identity()

    monkeypatch.setattr(torch.hub, "load", fake_load)
    C.load_encoder(C.MODEL_NAME, torch.device("cpu"))
    assert seen["repo"] == f"facebookresearch/dinov2:{C.HUB_COMMIT}" and seen["name"] == "dinov2_vitb14"
    assert seen["skip_validation"] is True and len(C.HUB_COMMIT) == 40


@pytest.mark.torch
def test_augment_and_loss():
    torch = _torch()
    from deadahead.baselines import dinov2_common as C
    x = torch.rand(4, 1, 32, 32)
    g1, g2 = torch.Generator().manual_seed(0), torch.Generator().manual_seed(0)
    K = C._blur_kernel(torch.device("cpu"))
    a, b = C.augment(x, g1, K), C.augment(x, g2, K)
    assert a.shape == x.shape and torch.equal(a, b) and float(a.min()) >= 0 and float(a.max()) <= 1
    z = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    assert float(C.nt_xent(z, z)) < float(C.nt_xent(z, torch.roll(z, 1, 0)))


@pytest.mark.torch
def test_adapt_embed_finetune_end_to_end(tmp_path, monkeypatch):
    torch = _torch()
    pytest.importorskip("tifffile")
    from deadahead.baselines import dinov2_adapt as AD
    from deadahead.baselines import dinov2_common as C
    from deadahead.baselines import dinov2_embed as EM
    from deadahead.baselines import dinov2_finetune as FT

    torch.set_num_threads(1)                       # repeated runs are compared bit for bit
    data = tmp_path / "data"
    monkeypatch.setenv("DEADAHEAD_DATA", str(data))
    write_movies(data)
    store = write_tables_and_crops(data)
    monkeypatch.setattr(AD, "TARGET", 30)
    TinyViT = tiny_vit_class(torch)
    monkeypatch.setattr(C, "load_encoder", lambda name, device, repo_dir=None: TinyViT().eval().to(device))
    common = ["--fold", HELD_OUT, "--steps", "2", "--batch", "4", "--res", "28", "--device", "cpu"]

    # adaptation: only the last two blocks and the final norm move
    assert AD.main(common) == 0
    ck = torch.load(C.adapted_checkpoint(HELD_OUT), weights_only=False)
    assert ck["recipe"]["fold"] == HELD_OUT and ck["recipe"]["hub_commit"] == C.HUB_COMMIT
    assert ck["recipe"]["n_tiles"] == 8 and np.load(AD.default_pool(HELD_OUT)).shape == (8, AD.SIZE, AD.SIZE)
    torch.manual_seed(AD.SEED)
    init = TinyViT().state_dict()
    sd = ck["state_dict"]
    unchanged = [k for k in sd if k.startswith(("patch_embed", "cls_token", "blocks.0."))]
    moved = [k for k in sd if k.startswith(("blocks.1.", "blocks.2.", "norm."))]
    assert unchanged and all(torch.equal(sd[k], init[k]) for k in unchanged)
    assert any(not torch.equal(sd[k], init[k]) for k in moved)

    # the same run again gives the same encoder
    again = tmp_path / "again.pt"
    assert AD.main(common + ["--out", str(again)]) == 0
    sd2 = torch.load(again, weights_only=False)["state_dict"]
    assert all(torch.equal(sd[k], sd2[k]) for k in sd)

    # embeddings of every movie by the fold's adapted encoder
    assert EM.main(["--fold", HELD_OUT, "--res", "28", "--batch", "5", "--device", "cpu"]) == 0
    out = C.embedding_store("adapted", HELD_OUT)
    for g in GROUPS:
        e = np.load(out / f"{g}_1.npy")
        n = len(np.load(store / f"{g}_1.npy"))
        assert e.shape == (n, 768) and e.dtype == np.float16 and np.isfinite(e).all()
        assert (out / f"{g}_1.parquet").read_bytes() == (store / f"{g}_1.parquet").read_bytes()
    model = TinyViT()
    C.load_adapted(model, C.adapted_checkpoint(HELD_OUT), torch.device("cpu"))
    direct = C.embed_movie(np.load(store / f"{GROUPS[1]}_1.npy"), model, torch.device("cpu"), 5, 28)
    assert np.array_equal(direct, np.load(out / f"{GROUPS[1]}_1.npy"))

    # fine-tuning on the 30-min labels of the training groups
    from deadahead.survival.config import head_config, windows_config
    tb = {n: pd.read_parquet(data / "tables" / f"{n}.parquet") for n in ("windows", "events", "movies")}
    X, Y, W, card = FT.build_labelled_set(HELD_OUT, 30, tb["windows"], tb["events"], tb["movies"], head_config(),
                                          windows_config(), store)
    assert card["groups_trained_on"] == ["AAA_do", "AAA_st"] and X.shape == (6, 32, 32)
    assert card["class_counts"] == {"NONE": 2, "APOPTOSIS": 2, "LYTIC": 2}
    assert np.allclose(sorted(W), sorted([0.6, 0.7, 0.8] * 2))
    assert FT.main(common) == 0
    fk = torch.load(C.finetuned_checkpoint(HELD_OUT), weights_only=False)
    assert fk["recipe"]["held_out_group"] == HELD_OUT and HELD_OUT not in fk["recipe"]["groups_trained_on"]
    assert fk["recipe"]["class_weights"] == pytest.approx([1.0, 1.0, 1.0])
    assert all(torch.equal(fk["state_dict"][k], init[k]) for k in unchanged)
    assert EM.main(["--fold", HELD_OUT, "--encoder", "finetuned", "--res", "28", "--device", "cpu"]) == 0
    assert len(list(C.embedding_store("finetuned", HELD_OUT).glob("*.npy"))) == len(GROUPS)


@pytest.mark.torch
def test_embedding_refuses_an_unknown_movie_before_it_loads_the_model(tmp_path, monkeypatch):
    _torch()
    from deadahead.baselines import dinov2_common as C
    from deadahead.baselines import dinov2_embed as EM

    crops, out = tmp_path / "crops", tmp_path / "out"
    crops.mkdir()
    np.save(crops / "m_1.npy", np.zeros((3, 8, 8), np.float16))
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"x")

    def no_model(*a, **k):
        raise AssertionError("the model was loaded")
    monkeypatch.setattr(C, "load_encoder", no_model)
    with pytest.raises(SystemExit, match="m_9"):
        EM.main(["--fold", "g", "--crops", str(crops), "--out-dir", str(out), "--weights", str(weights),
                 "--device", "cpu", "--movies", "m_1", "m_9"])
    assert not out.exists()


def test_embedding_resume_redoes_a_partial_movie(tmp_path, monkeypatch):
    _torch()
    from deadahead.baselines import dinov2_common as C
    from deadahead.baselines import dinov2_embed as EM

    crops, out = tmp_path / "crops", tmp_path / "out"
    crops.mkdir()
    np.save(crops / "m_1.npy", np.zeros((3, 8, 8), np.float16))
    pd.DataFrame({"cell_num": [1] * 3, "frame_index": [0, 5, 10], "row": [0, 1, 2]}).to_parquet(crops / "m_1.parquet")
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"x")
    monkeypatch.setattr(C, "load_encoder", lambda *a, **k: None)
    monkeypatch.setattr(C, "load_adapted", lambda *a, **k: {})
    monkeypatch.setattr(C, "embed_movie", lambda arr, *a: np.ones((len(arr), 768), np.float16))
    args = ["--fold", "g", "--crops", str(crops), "--out-dir", str(out), "--weights", str(weights), "--device", "cpu"]
    out.mkdir()
    (out / "m_1.npy.partial").write_bytes(b"cut short")           # what a killed run leaves
    assert EM.main(args) == 0
    assert np.load(out / "m_1.npy").shape == (3, 768) and (out / "m_1.parquet").is_file()


@pytest.mark.torch
def test_finetune_refuses_a_group_of_no_table(tmp_path, monkeypatch, capsys):
    _torch()
    from deadahead.baselines import dinov2_finetune as FT

    data = tmp_path / "data"
    monkeypatch.setenv("DEADAHEAD_DATA", str(data))
    write_movies(data)
    write_tables_and_crops(data)
    with pytest.raises(SystemExit) as e:
        FT.main(["--fold", HELD_OUT + "x", "--steps", "1", "--device", "cpu"])
    assert e.value.code == 2 and "no group" in capsys.readouterr().err


@pytest.mark.torch
def test_pinned_weights_are_checked(tmp_path):
    _torch()
    from deadahead.baselines import dinov2_common as C

    p = tmp_path / "w.pth"
    p.write_bytes(b"not the checkpoint")
    with pytest.raises(RuntimeError):
        C.check_weights(p)
    C.check_weights(tmp_path / "absent.pth")                      # nothing cached: the hub's download is checked next


@pytest.mark.torch
def test_config_agrees_with_the_constants():
    _torch()
    import yaml

    from deadahead.baselines import dinov2_adapt as AD
    from deadahead.baselines import dinov2_common as C
    from deadahead.baselines import dinov2_embed as EM
    from deadahead.baselines import dinov2_finetune as FT
    from deadahead.paths import REPO_ROOT

    cfg = yaml.safe_load((REPO_ROOT / "configs" / "baselines.yaml").read_text())["dinov2"]
    hub = cfg["hub"]
    assert (hub["repo"], hub["commit"], hub["entry_point"]) == (C.HUB_REPO, C.HUB_COMMIT, C.MODEL_NAME)
    assert hub["weights_sha256"] == C.WEIGHTS_SHA256 and hub["embed_dim"] == C.EMBED_DIM
    phase_map = (cfg["phase_map"]["lo"], cfg["phase_map"]["hi"])
    assert phase_map == (C.GLOBAL_LO, C.GLOBAL_HI) == (AD.GLOBAL_LO, AD.GLOBAL_HI)
    assert cfg["adapt"]["pool"]["tiles"] == AD.TARGET and cfg["embed"]["batch"] == EM.BATCH
    for block, m in ((cfg["adapt"], AD), (cfg["finetune"], FT)):
        assert (block["steps"], block["batch"], block["lr"], block["seed"]) == (m.STEPS, m.BATCH, m.LR, m.SEED)


@pytest.mark.torch
def test_finetune_loss_is_weighted_and_divided_by_the_batch_weight():
    torch = _torch()
    from deadahead.baselines import dinov2_finetune as FT

    torch.set_num_threads(2)
    TinyViT = tiny_vit_class(torch)
    rng = np.random.default_rng(0)
    X = rng.uniform(0.0, 2.0, size=(9, 32, 32)).astype(np.float16)
    Y = np.array([0, 1, 2] * 3, dtype=np.int64)
    W = rng.uniform(0.1, 3.0, size=9)

    def first_loss(weights):
        out = FT.finetune_encoder(X, Y, weights, steps=1, batch=9, res=28, dev=torch.device("cpu"),
                                  load_model=lambda d: TinyViT().to(d))
        return out["losses"][0]
    assert first_loss(W) == pytest.approx(first_loss(10.0 * W), rel=1e-5)     # the scale of the weights does not matter
    assert first_loss(W) != pytest.approx(first_loss(np.ones(9)), rel=1e-5)   # their shape does
