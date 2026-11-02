"""Reproduce the decomposition results (supplementary S4) from the bundle, on Table 1's rows and cohort.

Fine-tuning: the survival head with the handcrafted features, read on the fine-tuned VideoMAE features (DeadAhead)
against the same encoder's adapted features before fine-tuning. Test-time changes: the fine-tuned video model with
the tracked cell hidden, its 16 frames shuffled or its last frame repeated 16 times, against the intact model.
Both are scored as Table 1 is: AUPRC over P, Q and N per held-out group and seed, the seed mean and the macro.
The test-time changes are also scored on Table 1's original rows (every row with a known label, an untyped death a
negative), each file's AUPRC over all its rows, and the fine-tuning reading on the oracle landmark set (each held-out
landmark whose 30-min outcome is known in the complete movie, with its oracle label), when the bundle holds it.
"""

from __future__ import annotations

import numpy as np

from deadahead.evaluation.aggregate import outcome_text, paired, paired_both
from deadahead.evaluation.metrics import auprc
from deadahead.evaluation.rows import CAUSES, GROUPS, SEEDS, key_index, table1_rows
from deadahead.evaluation.table1 import table1
from deadahead.reproduce.table1 import prediction_scores

FINE_TUNED, ADAPTED = "deadahead", "videomae_adapted_survival"
ARMS = (FINE_TUNED, ADAPTED)              # x and y of the fine-tuning reading
INTACT = "videomae_finetuned"
CONDITIONS = ("hidden_cell", "shuffled_frames", "repeated_last_frame")
PRINT_ORDER = ("LYTIC", "APOPTOSIS")
CAUSE_IX = {"APOPTOSIS": 1, "LYTIC": 2}      # the oracle label: 0 no event, 1 apoptosis, 2 lytic death
CONDITION_LABELS = {"hidden_cell": "tracked cell hidden", "shuffled_frames": "16 frames shuffled",
                    "repeated_last_frame": "last frame repeated 16 times"}


def scores_of(df, rows) -> dict:
    """{(g, s): {cause: scores aligned to rows[rows.group == g]}} for one method's long predictions."""
    out = {}
    for g in GROUPS:
        rg = rows[rows["group"] == g].reset_index(drop=True)
        dg = df[df["group"] == g]
        for s in sorted(dg["seed"].unique()):
            out[(g, int(s))] = prediction_scores(dg, int(s), rg)
    return out


def methods_block(scores: dict, rows) -> dict:
    """Table 1's arithmetic: per method and cause the per-(group, seed) AUPRC over P, Q, N with the set counts, the
    seed means, the nine-group macro and the eight-group print."""
    return table1(scores, rows, list(scores))["methods"]


def per_group(methods: dict, m: str, c: str) -> dict:
    return methods[m][c]["nine_folds"]["per_group"]


def fine_tuning_readings(methods: dict) -> dict:
    """Fine-tuned minus adapted features under the same head, per cause: the paired comparison and both sides."""
    x, y = ARMS
    out = {}
    for c in PRINT_ORDER:
        px, py = per_group(methods, x, c), per_group(methods, y, c)
        out[c] = {"x": x, "y": y, "paired": paired_both(px, py), "x_values": methods[x][c],
                  "y_values": methods[y][c]}
    return out


def test_time_readings(methods: dict) -> dict:
    """Per cause: the intact model's AUPRC and, per test-time change, its AUPRC, the intact minus changed paired
    comparison and the groups where the AUPRC fell."""
    out = {}
    for c in PRINT_ORDER:
        pi = per_group(methods, INTACT, c)
        blk = {"intact": methods[INTACT][c], "changed": {}}
        for x in CONDITIONS:
            px = per_group(methods, x, c)
            pb = paired_both(pi, px)
            fell = sorted((g for g in pi if g in px and px[g] < pi[g]), key=GROUPS.index)
            blk["changed"][x] = {"auprc": methods[x][c], "intact_minus_changed": pb,
                                 "groups_where_it_fell": fell, "n_groups_where_it_fell": len(fell)}
        out[c] = blk
    return out


def original_row_scores(tt, intact, original) -> tuple[dict, dict]:
    """Per (condition, group, seed) and per (group, seed): {cause: (label == cause, score)} over every row of the
    stored file, in file order, with the labels of Table 1's original rows (an untyped death is a negative)."""
    idx = key_index(original)
    labels = original["label"].astype(str).to_numpy()

    def pairs(d):
        out = {}
        for c in CAUSES:
            x = d[d["cause"] == c]
            pos = idx.get_indexer(key_index(x))
            if (pos < 0).any():
                raise ValueError("a test-time prediction row is not one of Table 1's original rows")
            out[c] = (labels[pos] == c, x["score"].to_numpy(float))
        return out
    cond = {(x, g, int(s)): pairs(d) for (x, g, s), d in tt.groupby(["condition", "group", "seed"], sort=False)}
    base = {(g, int(s)): pairs(d) for (g, s), d in intact.groupby(["group", "seed"], sort=False)}
    return cond, base


def auprc_of(pair: tuple) -> float:
    """AUPRC of one cause's scores on all rows of a stored file."""
    return float(auprc(pair[0], pair[1]))


def original_rows_readings(cond: dict, intact: dict) -> dict:
    """Per cause, on the files' own rows: the intact model's AUPRC per group (seed mean) and macro, and per
    test-time change its AUPRC and the intact minus changed differences (per group, mean, groups ahead)."""
    out = {}
    for c in CAUSES:
        a_int = {g: float(np.mean([auprc_of(intact[(g, s)][c]) for s in SEEDS])) for g in GROUPS}
        conds = {}
        for x in CONDITIONS:
            A = {g: float(np.mean([auprc_of(cond[(x, g, s)][c]) for s in SEEDS])) for g in GROUPS}
            d = {g: a_int[g] - A[g] for g in GROUPS}
            dv = np.array(list(d.values()), float)
            conds[x] = {"macro_auprc": float(np.mean(list(A.values()))), "per_group_auprc": A,
                        "intact_minus_changed": {"per_group_delta": d, "delta_bar": float(dv.mean()),
                                                 "k_plus": int((dv > 0).sum()), "k_minus": int((dv < 0).sum()),
                                                 "ties": int((dv == 0).sum())}}
        out[c] = {"per_group_intact": a_int, "macro_intact": float(np.mean(list(a_int.values()))),
                  "conditions": conds}
    return out


def unit_metrics(y_cls: np.ndarray, p: np.ndarray, cause: str) -> dict | None:
    """One (group, seed, cause) on the oracle landmark set: the AUPRC; None when the group has no positive."""
    yt = (np.asarray(y_cls) == CAUSE_IX[cause]).astype(float)
    if yt.sum() == 0:
        return None
    p = np.asarray(p, dtype=float)
    return {"auprc": float(auprc(yt, p)), "n": int(len(yt)), "n_pos": int(yt.sum())}


def arm_values(per_seed: dict, metric: str = "auprc") -> dict:
    """per_seed: {seed: {group: metrics or None}} -> the per-group mean over the three seeds, the macro, the
    per-seed macro and its across-seed SD (ddof 0). A group with no positive is left out; a group with fewer than
    every seed raises."""
    seeds = sorted(per_seed)
    groups = sorted({g for s in seeds for g in per_seed[s]}, key=GROUPS.index)
    a, per_seed_macro = {}, {}
    for g in groups:
        vals = [per_seed[s].get(g) for s in seeds]
        if all(v is None for v in vals):
            continue
        if any(v is None for v in vals) or len(vals) != len(SEEDS):
            raise ValueError(f"group {g}: {metric} on fewer than three fits")
        a[g] = float(np.mean([v[metric] for v in vals]))
    for s in seeds:
        v = [per_seed[s][g][metric] for g in a]
        per_seed_macro[s] = float(np.mean(v)) if v else float("nan")
    m = list(per_seed_macro.values())
    return {"per_group": a, "macro": float(np.mean(list(a.values()))) if a else float("nan"),
            "per_seed_macro": per_seed_macro, "across_seed_sd_macro": float(np.std(m, ddof=0)) if m else float("nan")}


def oracle_units(bundle: dict) -> dict:
    """{(arm, g, s): {cause: (oracle labels, 30-min scores)}} on each group's oracle landmark set, in the stored
    prediction order."""
    lab = bundle["oracle_labels"]
    out = {}
    for arm, df in ((FINE_TUNED, bundle["predictions"][FINE_TUNED]), (ADAPTED, bundle["adapted_predictions"])):
        for g in GROUPS:
            lg = lab[lab["group"] == g]
            li = key_index(lg)
            dg = df[df["group"] == g]
            for s in sorted(dg["seed"].unique()):
                sc = {}
                for c in CAUSES:
                    d = dg[(dg["seed"] == s) & (dg["cause"] == c)]
                    pos = li.get_indexer(key_index(d))
                    keep = pos >= 0
                    if int(keep.sum()) != len(lg):
                        raise ValueError(f"{arm} {g} s{s}: the predictions do not cover the oracle landmark set")
                    sc[c] = (lg["label"].to_numpy()[pos[keep]], d["score"].to_numpy(float)[keep])
                out[(arm, g, int(s))] = sc
    return out


def oracle_readings(bundle: dict) -> dict:
    """Per cause, on the oracle landmark set: both sides' per-group AUPRC (seed mean), macro, per-seed macro and
    its SD, and the paired comparison (fine-tuned minus adapted)."""
    units = oracle_units(bundle)
    out = {}
    for c in PRINT_ORDER:
        per_seed = {a: {} for a in ARMS}
        for (arm, g, s), sc in units.items():
            per_seed[arm].setdefault(int(s), {})[g] = unit_metrics(*sc[c], c)
        xv, yv = arm_values(per_seed[ARMS[0]]), arm_values(per_seed[ARMS[1]])
        out[c] = {"x": ARMS[0], "y": ARMS[1], "paired": paired(xv["per_group"], yv["per_group"]),
                  "x_values": xv, "y_values": yv}
    return out


def part_decomposition(bundle: dict) -> dict:
    """The population counts, the fine-tuning reading and the test-time readings on the common cohort, the
    test-time readings on the original rows and, with the oracle labels in the bundle, the fine-tuning reading on
    the oracle landmark set."""
    preds = bundle["predictions"]
    if FINE_TUNED not in preds or INTACT not in preds or "adapted_predictions" not in bundle or \
            "test_time_predictions" not in bundle:
        raise ValueError("the bundle holds no decomposition predictions: the decomposition cannot be computed")
    tr = table1_rows(bundle["rows"], bundle["untyped"])
    cohort = tr["cohort"]
    ft = {FINE_TUNED: scores_of(preds[FINE_TUNED], cohort),
          ADAPTED: scores_of(bundle["adapted_predictions"], cohort)}
    tt = bundle["test_time_predictions"]
    st = {INTACT: scores_of(preds[INTACT], cohort)}
    for x in CONDITIONS:
        st[x] = scores_of(tt[tt["condition"] == x], cohort)
    return {"population": {"counts": tr["counts"], "n_rows": len(cohort)},
            "fine_tuning": methods_and_readings(ft, cohort, fine_tuning_readings),
            "test_time": methods_and_readings(st, cohort, test_time_readings),
            "test_time_original_rows": original_rows_readings(*original_row_scores(tt, preds[INTACT],
                                                                                   tr["original"])),
            **({"fine_tuning_oracle_landmarks": oracle_readings(bundle)} if "oracle_labels" in bundle else {})}


def methods_and_readings(scores: dict, rows, reader) -> dict:
    return {"readings": reader(methods_block(scores, rows))}


def format_decomposition(res: dict) -> str:
    """Both readings as text: macro AUPRC per side and change, the macro difference and the groups ahead."""
    out = ["Fine-tuning: DeadAhead (fine-tuned VideoMAE features) minus the adapted features, same head "
           "(macro AUPRC, 9 groups)"]
    for c, r in res["fine_tuning"]["readings"].items():
        p = r["paired"]["nine_folds"]
        out.append(f"  {c.lower():<10} {p['macro_x']:.3f} vs {p['macro_y']:.3f}: {p['macro_difference']:+.3f} "
                   f"({p['x_ahead']} of {p['n']} groups ahead; {outcome_text(p)})")
    out.append("Test-time changes to the fine-tuned video model (macro AUPRC, 9 groups)")
    for c, r in res["test_time"]["readings"].items():
        cells = [f"intact {r['intact']['nine_folds']['macro']:.3f}"]
        for x, v in r["changed"].items():
            cells.append(f"{CONDITION_LABELS[x]} {v['auprc']['nine_folds']['macro']:.3f} "
                         f"(fell in {v['n_groups_where_it_fell']} of 9)")
        out.append(f"  {c.lower():<10} " + ", ".join(cells))
    return "\n".join(out)
