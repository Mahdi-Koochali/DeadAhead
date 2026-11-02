"""Leaf-by-leaf comparison of two result trees, for the cross-checks between reproduce parts."""

from __future__ import annotations

import math


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def compare_tree(a, b, tol: float, ignore_keys=(), path: str = "", acc: dict | None = None) -> dict:
    """a (expected) against b (computed): the same keys and lengths, every number within tol (NaN only with NaN),
    every other leaf equal. Returns {max_abs_diff, n_leaves, n_problems, problems (the first 50), passes}."""
    top = acc is None
    acc = acc if acc is not None else {"max_abs_diff": 0.0, "n_leaves": 0, "n_problems": 0, "problems": []}

    def bad(msg):
        acc["n_problems"] += 1
        if len(acc["problems"]) < 50:
            acc["problems"].append(msg)
    if isinstance(a, dict) and isinstance(b, dict):
        ka, kb = set(a) - set(ignore_keys), set(b) - set(ignore_keys)
        if ka != kb:
            bad(f"{path}: keys differ (only expected {sorted(map(str, ka - kb))[:3]}, only computed "
                f"{sorted(map(str, kb - ka))[:3]})")
        for k in sorted(ka & kb, key=str):
            compare_tree(a[k], b[k], tol, ignore_keys, f"{path}.{k}", acc)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            bad(f"{path}: {len(a)} expected items, {len(b)} computed")
        for i, (x, y) in enumerate(zip(a, b)):
            compare_tree(x, y, tol, ignore_keys, f"{path}[{i}]", acc)
    elif _num(a) and _num(b):
        acc["n_leaves"] += 1
        fa, fb = float(a), float(b)
        if math.isnan(fa) or math.isnan(fb):
            if not (math.isnan(fa) and math.isnan(fb)):
                bad(f"{path}: {a!r} expected, {b!r} computed")
        elif math.isinf(fa) or math.isinf(fb):
            if fa != fb:
                bad(f"{path}: {a!r} expected, {b!r} computed")
        else:
            d = abs(fa - fb)
            acc["max_abs_diff"] = max(acc["max_abs_diff"], d)
            if not d <= tol:
                bad(f"{path}: |{a!r} - {b!r}| = {d:.3g} > {tol:g}")
    else:
        acc["n_leaves"] += 1
        if type(a) is not type(b) or a != b:
            bad(f"{path}: {a!r} expected, {b!r} computed")
    if top:
        acc["passes"] = acc["n_problems"] == 0
        acc["tolerance"] = tol
    return acc
