import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

for _stage in sorted(p for p in Path(__file__).resolve().parents[1].iterdir() if p.is_dir()):
    sys.path.insert(0, str(_stage))
from exact_tests import holm, mcnemar_exact  # noqa: E402

ROOT = Path(os.environ.get("SENTIO_ROOT")
            or Path(__file__).resolve().parents[4])
CH = ROOT / "study/channel/results"
TABLE = CH / "tables/surface_ladder.json"
OUT = CH / "tables/surface_ladder_p1_p2.json"
MAN = CH / "raw/surface_ladder_p1_p2.manifest.json"
MODELS = ["0.6B", "4B", "zerank-2", "FollowIR-7B", "ctxl-6B", "gemma"]
FIVE = MODELS[:5]


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def resolved(raw):
    out = {}
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        pos = {cid: i for i, cid in enumerate(r["ranking"])}
        g = [pos[c] for c in r["gold"] if c in pos]
        c = [pos[x] for x in r["counterpart"] if x in pos]
        if g and c:
            out[r["query_id"]] = int(min(g) < min(c))
    return out


def main():
    board = json.loads(TABLE.read_text(encoding="utf-8"))["papan"]
    if list(board) != MODELS:
        raise SystemExit(f"ABORT: the frozen table lists the models {list(board)}")
    per_model, inputs = {}, {"surface_ladder.json": sha256_bytes(TABLE.read_bytes())}
    for m in MODELS:
        cells = {}
        for key in ("p1_nama_en", "p2_nama_id"):
            cell = board[m][key]
            raw = (CH / "raw" / cell["berkas"]).read_bytes()
            if sha256_bytes(raw) != cell["sha256"]:
                raise SystemExit(f"ABORT: {m} {key}: sha256 differs from the frozen table")
            res = resolved(raw)
            if not (sum(res.values()) == cell["menang"] and len(res) == cell["n"] == 96):
                raise SystemExit(f"ABORT: {m} {key}: {sum(res.values())} resolved rows, not the count of "
                                 "the frozen table")
            cells[key] = (cell, res)
            inputs[cell["berkas"]] = cell["sha256"]
        (c1, r1), (c2, r2) = cells["p1_nama_en"], cells["p2_nama_id"]
        rows = sorted(set(r1) & set(r2))
        a = sum(1 for q in rows if r1[q] == 1 and r2[q] == 0)
        b = sum(1 for q in rows if r1[q] == 0 and r2[q] == 1)
        per_model[m] = {
            "n": len(rows),
            "p1_menang": sum(r1[q] for q in rows),
            "p2_menang": sum(r2[q] for q in rows),
            "p1_saja": a,
            "p2_saja": b,
            "p_mcnemar_eksak": round(mcnemar_exact(a, b), 6),
            "berkas_p1": c1["berkas"], "sha256_p1": c1["sha256"],
            "berkas_p2": c2["berkas"], "sha256_p2": c2["sha256"],
        }
    for m, h in zip(MODELS, holm([per_model[m]["p_mcnemar_eksak"] for m in MODELS])):
        per_model[m]["p_holm_keluarga_enam"] = round(h, 6)
        per_model[m]["signifikan_0.05"] = h < 0.05
    lower = [m for m in FIVE if per_model[m]["p2_menang"] < per_model[m]["p1_menang"]]
    significant = [m for m in FIVE if per_model[m]["signifikan_0.05"]]
    created = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    result = {
        "dibuat": created,
        "_apa_ini": ("Deciding contrast of the surface ladder: p1 (English program name in the label) against "
                     "p2 (Indonesian program name), both with an English instruction that names the program. "
                     "Exact paired McNemar, Holm within the family of six models."),
        "aturan_menang": ("a row is resolved when the best position of the asker's gold in the stored ranking is "
                          "above the best position of the rival's gold; a row with a gold missing is dropped"),
        "sumber": {"tabel_beku": "study/channel/results/tables/surface_ladder.json",
                   "skor_per_item": "study/channel/results/raw",
                   "cek": "every score file's sha256 and every p1 and p2 count equal the frozen table, six models"},
        "per_model": per_model,
        "ringkas_lima_model": {
            "model": FIVE,
            "p2_lebih_rendah_dari_p1": len(lower),
            "signifikan_holm": len(significant),
            "model_signifikan": significant,
            "holm_maks_di_antara_yang_signifikan": max((per_model[m]["p_holm_keluarga_enam"] for m in significant),
                                                       default=None),
        },
    }
    OUT.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    MAN.write_text(json.dumps({"script": Path(__file__).name, "dibuat": created, "masukan_sha256": inputs,
                               "output": {"path": "results/tables/surface_ladder_p1_p2.json",
                                          "sha256": sha256_bytes(OUT.read_bytes())}},
                              indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("model        n  p1  p2  p1only p2only  p_exact  Holm6    sig")
    for m in MODELS:
        v = per_model[m]
        print(f"{m:11s} {v['n']:3d} {v['p1_menang']:3d} {v['p2_menang']:3d}  {v['p1_saja']:5d} {v['p2_saja']:6d}  "
              f"{v['p_mcnemar_eksak']:.6f} {v['p_holm_keluarga_enam']:.6f} {v['signifikan_0.05']}")
    print("five models:", result["ringkas_lima_model"])
    print("wrote", OUT.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
