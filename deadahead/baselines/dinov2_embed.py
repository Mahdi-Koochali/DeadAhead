"""Per-frame DINOv2 embeddings of every crop of every movie, by the encoder of one fold.

Every movie (the held-out group's included) is embedded by the encoder adapted (``--encoder adapted``) or fine-tuned
(``--encoder finetuned``) for the held-out group ``--fold``, through the global phase map at 224 px. Per movie the
store holds ``<movie>.npy``, float16 (N, 768), one row per crop of the 128-px store (every 5th frame), and
``<movie>.parquet``, the crop store's index (cell_num, frame_index, row). The LightGBM heads pool nine of these
rows per landmark (``deadahead.baselines.heads``). Resumable: a movie whose array exists is skipped.

    deadahead baseline-dinov2-embed --fold DU145_bp --encoder adapted
    deadahead baseline-dinov2-embed --fold DU145_bp --encoder finetuned
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

BATCH = 256
RES = 224


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Embed every crop with the DINOv2 encoder of one fold.")
    ap.add_argument("--fold", required=True, help="held-out group whose encoder embeds the movies")
    ap.add_argument("--encoder", choices=["adapted", "finetuned"], default="adapted",
                    help="the encoder adapted without labels or the one fine-tuned on the labels")
    ap.add_argument("--weights", type=Path, default=None, help="checkpoint (default: the fold's adapted or fine-tuned one)")
    ap.add_argument("--crops", type=Path, default=None, help="crop store (default: crops/frames128)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: runs/dinov2/embeddings/<fold> or runs/dinov2_finetuned/embeddings/<fold>")
    ap.add_argument("--movies", nargs="*", default=None)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--res", type=int, default=RES, help="a multiple of the patch size 14")
    ap.add_argument("--hub-repo-dir", type=Path, default=None,
                    help="local copy of the pinned facebookresearch/dinov2 commit (default: download it)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    from . import dinov2_common as C

    crops = a.crops or C.crops_store()
    if not crops.is_dir():
        raise SystemExit(f"{crops} is missing: build it with deadahead video-crops --stride 5 --out-dir {crops}")
    weights = a.weights or (C.adapted_checkpoint(a.fold) if a.encoder == "adapted" else C.finetuned_checkpoint(a.fold))
    if not weights.is_file():
        raise SystemExit(f"{weights} is missing")
    out = a.out_dir or C.embedding_store(a.encoder, a.fold)
    shards = sorted(crops.glob("*.npy"))
    if a.movies:
        unknown = sorted(set(a.movies) - {s.stem for s in shards})
        if unknown:
            raise SystemExit(f"no crops for movie(s) {unknown} in {crops}")
        shards = [s for s in shards if s.stem in a.movies]

    device = C.default_device(a.device)
    model = C.load_encoder(C.MODEL_NAME, device, a.hub_repo_dir)
    C.load_adapted(model, weights, device, expect_backbone="dinov2")
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for i, s in enumerate(shards, 1):
        dst = out / f"{s.stem}.npy"
        if dst.exists() and not a.force:
            continue
        arr = np.load(s, mmap_mode="r")
        t1 = time.time()
        emb = C.embed_movie(arr, model, device, a.batch, a.res)
        (out / f"{s.stem}.parquet").write_bytes((crops / f"{s.stem}.parquet").read_bytes())
        tmp = dst.with_name(f"{dst.name}.partial")               # written whole, then renamed: the array comes last
        with open(tmp, "wb") as f:
            np.save(f, emb)
        tmp.replace(dst)
        print(f"[{i}/{len(shards)}] {s.stem:<14} {emb.shape[0]:>7} x {emb.shape[1]}  "
              f"{time.time()-t1:5.1f}s  (elapsed {(time.time()-t0)/60:.1f} min)", flush=True)
    print(f"{len(list(out.glob('*.npy')))} movies in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
