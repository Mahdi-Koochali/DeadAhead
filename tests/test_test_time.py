"""The test-time changes on synthetic clips: the numpy and torch transforms agree bit for bit, the masks, the
permutations, the mask store's checks, and the forward pass on a tiny model and store."""
import json
import types

import numpy as np
import pandas as pd
import pytest

from deadahead.video import test_time as TT


def test_permutations_never_identity_and_pinned():
    p = TT.permutations(2, 50)
    assert p.shape == (50, TT.FRAMES)
    assert all(sorted(r) == list(range(TT.FRAMES)) and not np.array_equal(r, np.arange(TT.FRAMES)) for r in p)
    assert np.array_equal(p, TT.permutations(2, 50))
    assert TT.permutations_sha256(TT.permutations(TT.FOLD_ORDER.index("PNT1A_bp"), 3)) != TT.PERM_SHA256["PNT1A_bp"]


def test_mask_planes_and_background():
    lab = np.zeros((40, 40), dtype=np.int32)
    lab[10:14, 10:14] = 3
    lab[30:33, 30:33] = 5
    planes = TT.mask_planes(lab, 3, 12.0, 12.0, size=16, radius=1)
    own = (planes & TT.BIT_OWN) != 0
    dil = (planes & TT.BIT_OWN_DIL) != 0
    assert own.sum() == 16 and dil.sum() > own.sum() and np.all(dil[own])
    assert ((planes & TT.BIT_ANY) != 0).sum() == 16
    edge = TT.mask_planes(lab, 3, 2.0, 2.0, size=16, radius=1)       # window partly outside the field
    assert ((edge & TT.BIT_OWN_DIL) != 0)[:6, :6].sum() == 0
    img = np.full((40, 40), 0.25)
    assert TT.background_level(img, lab, margin=3) == 0.25


def test_apply_condition_numpy():
    rng = np.random.default_rng(0)
    clips = rng.normal(size=(3, TT.FRAMES, 6, 6)).astype(np.float16)
    planes = rng.integers(0, 8, size=clips.shape).astype(np.uint8)
    perms = TT.permutations(0, 3)
    assert np.array_equal(TT.apply_condition("intact", clips), clips)
    hid = TT.apply_condition("hidden_cell", clips, planes)
    assert np.all(hid[(planes & TT.BIT_OWN_DIL) != 0] == 0) and hid.dtype == clips.dtype
    sh = TT.apply_condition("shuffled_frames", clips, perms=perms)
    assert np.array_equal(sh[1, 0], clips[1, perms[1, 0]])
    st = TT.apply_condition("repeated_last_frame", clips)
    assert all(np.array_equal(st[:, k], clips[:, -1]) for k in range(TT.FRAMES))
    with pytest.raises(ValueError):
        TT.apply_condition("unknown", clips)


def movie_files(tmp_path):
    """A two-frame movie on disk: image and mask stacks, its crop shard and the per-cell table."""
    tifffile = pytest.importorskip("tifffile")
    pytest.importorskip("scipy")
    from deadahead.video.crops import extract_movie
    rng = np.random.default_rng(3)
    n_frames, side = 2, 160
    lab = np.zeros((n_frames, side, side), dtype=np.float32)
    lab[:, 70:80, 70:80] = 1
    lab[:, 20:26, 120:130] = 2                       # near the border: its crop is clipped
    img = rng.normal(0.0, 0.01, size=lab.shape).astype(np.float32)
    img[lab > 0] += 1.0
    tifffile.imwrite(tmp_path / "img.tif", img, photometric="minisblack")
    tifffile.imwrite(tmp_path / "mask.tif", lab, photometric="minisblack")
    cells = []
    for f in range(n_frames):
        for c in (1, 2):
            ys, xs = np.nonzero(lab[f] == c)
            cells.append({"movie_id": "M", "cell_num": c, "track_id": f"M:{c}", "frame_index": f,
                          "centroid_y_px": float(ys.mean()), "centroid_x_px": float(xs.mean()),
                          "area_px": float(len(ys)), "mean_phase": float(img[f][lab[f] == c].mean())})
    cells = pd.DataFrame(cells)
    arr, idx = extract_movie(str(tmp_path / "img.tif"), "M", cells, TT.CROP, 1, 0.0)
    crops = tmp_path / "crops"
    crops.mkdir()
    np.save(crops / "M.npy", arr)
    idx.to_parquet(crops / "M.parquet", index=False)
    cells.to_parquet(tmp_path / "cells.parquet", index=False)
    return crops, cells, lab


def test_build_movie_masks(tmp_path):
    crops, _, _ = movie_files(tmp_path)
    out = tmp_path / "masks"
    out.mkdir()
    rec = TT.build_movie("M", "G", str(tmp_path / "img.tif"), str(tmp_path / "mask.tif"), str(out), str(crops),
                         str(tmp_path / "cells.parquet"))
    assert rec["rows"] == 4 and rec["area_check_failures"] == 0 and rec["mean_check_failures"] == 0
    assert rec["mean_check_rows"] == 2 and rec["rows_with_empty_central_mask_in_crop"] == 0
    planes = np.load(out / "M.npy")
    assert planes.shape == (4, TT.CROP, TT.CROP)
    assert int(((planes[0] & TT.BIT_OWN) != 0).sum()) == 100
    assert np.all(np.isfinite(np.load(out / "M.fill.npy").astype(float)))


def masks_command(tmp_path, crops):
    pd.DataFrame([{"movie_id": "M", "cell_line": "AAA", "treatment": "bp", "image_path": str(tmp_path / "img.tif"),
                   "mask_path": str(tmp_path / "mask.tif")}]).to_csv(tmp_path / "movies.csv", index=False)
    return ["masks", "--movie-table", str(tmp_path / "movies.csv"), "--cells", str(tmp_path / "cells.parquet"),
            "--store", str(crops), "--out-dir", str(tmp_path / "masks")]


def test_masks_command_passes_a_clean_store(tmp_path):
    crops, _, _ = movie_files(tmp_path)
    args = masks_command(tmp_path, crops)
    assert TT.main(args) == 0
    assert TT.main(args) == 0


def test_masks_command_fails_again_on_a_rerun(tmp_path, capsys):
    crops, cells, _ = movie_files(tmp_path)
    cells.loc[0, "area_px"] += 5                     # one wrong area
    cells.to_parquet(tmp_path / "cells.parquet", index=False)
    args = masks_command(tmp_path, crops)
    assert TT.main(args) == 1
    assert TT.main(args) == 1
    assert "area_check_failures" in capsys.readouterr().err


def test_masks_command_fails_on_non_integer_labels(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    crops, _, lab = movie_files(tmp_path)
    lab[0, 70, 70] = 1.5
    tifffile.imwrite(tmp_path / "mask.tif", lab, photometric="minisblack")
    assert TT.main(masks_command(tmp_path, crops)) == 1


def test_masks_command_fails_with_no_background_pixel(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    crops, _, lab = movie_files(tmp_path)
    lab[0, :, :] = np.where(lab[0] > 0, lab[0], 9)   # every pixel of one frame lies in some cell
    tifffile.imwrite(tmp_path / "mask.tif", lab, photometric="minisblack")
    assert TT.main(masks_command(tmp_path, crops)) == 1
    assert json.loads((tmp_path / "masks" / "M.json").read_text())["frames_with_no_background_pixel"] == 1


def test_numpy_and_torch_transforms_agree():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(1)
    clips = rng.normal(size=(4, TT.FRAMES, 5, 5)).astype(np.float16)
    planes = rng.integers(0, 8, size=clips.shape).astype(np.uint8)
    perms = TT.permutations(1, 4)
    for c in TT.CONDITIONS:
        want = TT.apply_condition(c, clips, planes, perms)
        got, share = TT.apply_condition_torch(c, torch.from_numpy(clips), torch.from_numpy(planes),
                                              torch.from_numpy(perms).long(), None)
        assert np.array_equal(got.numpy().view(np.uint16), want.view(np.uint16)), c
        assert (share is None) == (c not in TT.MASK_RULE)


def tiny_model(torch):
    """A stand-in classifier: three logits from the mean of each frame."""

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            with torch.random.fork_rng():                       # the global generator is left as it was
                torch.manual_seed(0)
                self.lin = torch.nn.Linear(TT.FRAMES, 3)

        def forward(self, pixel_values):
            return types.SimpleNamespace(logits=self.lin(pixel_values.mean(dim=(2, 3, 4))))
    return Tiny().eval()


def tiny_stores(tmp_path, n=24, side=8):
    rng = np.random.default_rng(2)
    store, masks = tmp_path / "store", tmp_path / "masks"
    store.mkdir()
    masks.mkdir()
    np.save(store / "M_1.npy", rng.uniform(-0.2, 3.0, size=(n, side, side)).astype(np.float16))
    planes = np.zeros((n, side, side), dtype=np.uint8)
    planes[:, 2:5, 2:5] = TT.BIT_OWN | TT.BIT_OWN_DIL | TT.BIT_ANY
    planes[:, 1:6, 1:6] |= TT.BIT_OWN_DIL
    np.save(masks / "M_1.npy", planes)
    np.save(masks / "M_1.fill.npy", np.zeros(n, dtype=np.float16))
    rows = pd.DataFrame({"movie_id": ["M_1"] * 3,
                         "frame_rows": [np.arange(k, k + TT.FRAMES) for k in range(3)]})
    return store, masks, rows


def test_forward_conditions_tiny(tmp_path):
    torch = pytest.importorskip("torch")
    from deadahead.video.predict import predict_probs
    store, masks, rows = tiny_stores(tmp_path)
    model, dev = tiny_model(torch), torch.device("cpu")
    mean, std = [0.45] * 3, [0.225] * 3
    perms = TT.permutations(0, len(rows))
    probs, filled = TT.forward_conditions(model, rows, store, masks, mean, std, dev, perms, batch=2)
    assert set(probs) == set(TT.CONDITIONS) and all(p.shape == (3, 3) for p in probs.values())
    assert np.allclose(probs["intact"].sum(axis=1), 1.0)
    assert np.array_equal(probs["intact"], predict_probs(model, rows, store, mean, std, dev, batch=2))
    assert not any(np.array_equal(probs[c], probs["intact"]) for c in TT.ALTERED)
    assert filled["hidden_cell"] == pytest.approx(25 / 64)
    empty = np.load(masks / "M_1.npy")
    empty[1] = 0
    np.save(masks / "M_1.npy", empty)
    with pytest.raises(LookupError):
        TT.forward_conditions(model, rows, store, masks, mean, std, dev, perms)
