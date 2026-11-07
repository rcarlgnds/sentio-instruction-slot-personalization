"""Single-field label control on the paired track, aggregated.

PRE-REGISTERED before any new cell ran (6 Sep 2026). The frozen wrong-label
control bounds the presence of the document label against its content but
cannot separate the program field from the semester field, because the
injected label carries both and, on this track, the semester alone
identifies the gold. This experiment injects labels carrying ONE field at
a time, in the slot-plus-text condition, for all three rerankers:

  program-only  label: [Source: <DEPT> degree program textbook]
  semester-only label: [Source: degree program textbook, semester <N>]

Reading rules, fixed in advance:
  1. every cell reports resolved wins via the production metrik_paired
     definition, over rows where gold and counterpart both resolve;
  2. each single-field cell is compared with its model's frozen both-field
     cell by the exact conditional McNemar test on discordant rows, and
     with the 48/96 lock by the one-sided exact binomial;
  3. interpretation, written before the numbers existed: if the
     program-only cell retains most of the both-field gain while the
     semester-only cell falls toward the lock, the program field carries
     the signal; the reverse implicates the semester field; both cells
     retaining the gain means either field suffices; both falling means
     the conjunction was doing the work;
  4. all six cells are reported whatever they show; no thresholds move.

  python single_field_label.py
"""

import os
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

for _stage in sorted(p for p in Path(__file__).resolve().parents[1].iterdir() if p.is_dir()):
    sys.path.insert(0, str(_stage))
from exact_tests import mcnemar_exact  # noqa: E402

ROOT = Path(os.environ.get("SENTIO_ROOT")
            or Path(__file__).resolve().parents[4])
Q2 = ROOT / "study/channel/results"
RAW = Q2 / "raw"

MODELS = {
    "Qwen3-Reranker-0.6B": "scores_qwen3rr_Qwen3-Reranker-0.6B_paired_profile_docmeta",
    "Qwen3-Reranker-4B": "scores_qwen3rr_Qwen3-Reranker-4B_paired_profile_docmeta",
    "zerank-2": "scores_zerank2_zerank-2_paired_profile_docmeta",
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rows(path: Path) -> dict:
    out = {}
    for l in path.open():
        r = json.loads(l)
        out[r["query_id"]] = r
    return out


def win(r) -> bool | None:
    gold = set(r["gold"] or [])
    other = set(r.get("counterpart") or [])
    pos = {c: i for i, c in enumerate(r["ranking"])}
    g = min((pos[c] for c in gold if c in pos), default=None)
    o = min((pos[c] for c in other if c in pos), default=None)
    if g is None or o is None:
        return None
    return g < o


def binom_one_sided_vs_half(wins: int, n: int) -> float:
    return sum(math.comb(n, i) for i in range(wins, n + 1)) / 2 ** n


def compute(rmap: dict) -> tuple:
    wins = [win(r) for r in rmap.values()]
    return (sum(1 for w in wins if w is not None),
            sum(1 for w in wins if w))


def main():
    table = {"dibuat": datetime.now(timezone.utc).isoformat(),
             "_apa_ini": "Single-field label control on the paired track, slot-plus-text condition, three "
                         "rerankers. The both cells are the saved slot-plus-text cells, and the program and "
                         "semester cells were scored for this control.",
             "praregistrasi": "the reading and interpretation rules are in the docstring of the script",
             "format_label": {
                 "both": "[Source: <DEPT> degree program textbook, semester <N>]",
                 "program": "[Source: <DEPT> degree program textbook]",
                 "semester": "[Source: degree program textbook, semester <N>]"},
             "model": {}, "sumber": {}}
    for model, base in MODELS.items():
        paths = {"both_frozen": RAW / f"{base}.jsonl",
                 "program_only": RAW / f"{base}_fieldprogram.jsonl",
                 "semester_only": RAW / f"{base}_fieldsemester.jsonl"}
        for name, p in paths.items():
            if not p.exists():
                raise SystemExit(f"ABORT: {model} cell {name} missing "
                                 f"({p.name})")
        per = {name: rows(p) for name, p in paths.items()}
        both = per["both_frozen"]
        entry = {}
        for name, rmap in per.items():
            n_cmp, n_win = compute(rmap)
            e = {"n_bandingkan": n_cmp, "menang": n_win,
                 "p_binomial_satu_sisi_vs_kunci": round(
                     binom_one_sided_vs_half(n_win, n_cmp), 6)}
            if name != "both_frozen":
                b = c = 0
                for qid, r in rmap.items():
                    if qid not in both:
                        continue
                    w_new, w_old = win(r), win(both[qid])
                    if w_new is None or w_old is None:
                        continue
                    if w_new and not w_old:
                        b += 1
                    elif w_old and not w_new:
                        c += 1
                e["diskordan_vs_both"] = {"menang_satu_field_saja": b,
                                          "menang_both_saja": c,
                                          "p_mcnemar_eksak": round(
                                              mcnemar_exact(b, c), 6)}
            entry[name] = e
            table["sumber"][f"{model}/{name}"] = {
                "path": str(paths[name].relative_to(ROOT)),
                "sha256": sha(paths[name])}
        table["model"][model] = entry
    out = Q2 / "tables/single_field_label.json"
    out.write_text(json.dumps(table, indent=2, ensure_ascii=False))
    manifest = {
        "run_id": "single_field_label_" + datetime.now(timezone.utc)
        .strftime("%Y%m%dT%H%M%SZ"),
        "script": "study/channel/scripts/6_label_check/single_field_label.py",
        "sumber": table["sumber"],
        "output": {"path": "results/tables/single_field_label.json",
                   "sha256": sha(out)},
        "label_source": "deterministic (aggregation of frozen scores)",
    }
    (Q2 / "tables/single_field_label.manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False))
    summary = {m: {k: v["menang"] for k, v in e.items()}
               for m, e in table["model"].items()}
    print(json.dumps(summary, indent=1))
    print(f"written: {out}")


if __name__ == "__main__":
    main()
