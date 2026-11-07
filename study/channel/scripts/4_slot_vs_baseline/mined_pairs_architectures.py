import os
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("SENTIO_ROOT")
            or Path(__file__).resolve().parents[4])

for _stage in sorted(p for p in Path(__file__).resolve().parents[1].iterdir() if p.is_dir()):
    sys.path.insert(0, str(_stage))
import metrics as M  # noqa: E402
from equivalence_bounds import analysis  # noqa: E402
from mcnemar import per_item_results, holm, mcnemar_exact  # noqa: E402

RAW = ROOT / "study/channel/results/raw"
OUT = ROOT / "study/channel/results/tables/channel_board_xpair_architectures.json"
MAN = RAW / "channel_board_xpair_architectures.manifest.json"
MODEL = {"followir": "FollowIR-7B", "ctxl": "ctxl-rerank-v2-instruct-multilingual-6b", "gemma": "bge-reranker-v2-gemma"}
CELLS = ["none", "slot", "both", "wrong"]
KEY = 600
N = 1621


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def contrast(wa, wb):
    ids = sorted(set(wa) & set(wb))
    n01 = sum(1 for q in ids if not wa[q] and wb[q])
    n10 = sum(1 for q in ids if wa[q] and not wb[q])
    p = mcnemar_exact(n01, n10)
    return {"n": len(ids), "a_kalah_b_menang": n01, "a_menang_b_kalah": n10, "selisih_bersih": n01 - n10,
            "selisih_pp": round((n01 - n10) / len(ids) * 100, 2) if ids else None,
            "p_mcnemar_exact": float(f"{p:.4g}"), "_p": p}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-missing", action="store_true")
    a = ap.parse_args()
    board, praw, source = {}, {}, {}
    for fam, mdl in MODEL.items():
        item, cell = {}, {}
        for c in CELLS:
            fp = RAW / f"scores_{fam}_{mdl}_xpair_{c}.jsonl"
            if not fp.with_suffix(".manifest.json").exists():
                if a.allow_missing:
                    cell[c] = "menunggu"
                    continue
                raise SystemExit(f"ABORT: cell {fam} xpair {c} is missing ({fp.name}); use --allow-missing for a dry pass.")
            w = per_item_results(fp); item[c] = w
            m = M.paired_metrics(M.read_ranking(fp))
            if len(w) != N:
                raise SystemExit(f"ABORT: {fp.name} has {len(w)} comparable rows, not {N}.")
            cell[c] = {"menang": sum(w.values()), "eksak": f"{sum(w.values())}/{N}", "pairwise": m["pairwise"],
                      "top1_gold": m["top1_gold"], "ndcg5": m["ndcg5"], "sha256": sha(fp)}
            source[fp.name] = cell[c]["sha256"]
        rec = {"keturunan": {"followir": "Mistral, tuned to follow retrieval instructions", "ctxl": "Mistral", "gemma": "Gemma (Google)"}[fam],
               "sel": cell}
        if all(c in item for c in ("none", "slot")):
            k = contrast(item["none"], item["slot"]); praw[mdl] = k.pop("_p")
            an = analysis(mdl, k["a_kalah_b_menang"], k["a_menang_b_kalah"], k["n"])
            rec["none_vs_slot"] = k
            rec["batas_efek_slot_poin_persen"] = an["batas_atas_poin_persen"]
            rec["ci_atas_poin_persen"] = round(an["dilaporkan95"][1] * 100, 2)
            rec["batas_ekuivalensi_slot"] = an
            if cell["none"]["menang"] != KEY:
                rec["peringatan_kunci"] = f"sel none {cell['none']['menang']} != structural key {KEY}"
        if all(c in item for c in ("slot", "both")):
            k = contrast(item["slot"], item["both"]); k.pop("_p"); rec["slot_vs_both"] = k
            rec["label_ditambahkan_pp"] = round((cell["both"]["menang"] - cell["slot"]["menang"]) / N * 100, 1)
            rec["label_ditambahkan_baris"] = cell["both"]["menang"] - cell["slot"]["menang"]
        if all(c in item for c in ("wrong", "both")):
            k = contrast(item["wrong"], item["both"]); k.pop("_p"); rec["wrong_vs_both"] = k
        board[mdl] = rec
    ph = holm(praw) if praw else {}
    for mdl, p in ph.items():
        board[mdl]["none_vs_slot"]["p_holm_keluarga_E"] = p
        board[mdl]["none_vs_slot"]["signifikan_0.05"] = p < 0.05
    result = {"dibuat": datetime.now(timezone.utc).isoformat(), "config_version_lock": "0a7b2e4de856",
             "praregistrasi": "PRAREGISTRASI_LINTAS_ARSITEKTUR_ADENDUM2.md section E",
             "_apa_ini": ("Channel board of the 1,621-row broad track (8 programs) for bge-gemma, ctxl-6B, and "
                          "FollowIR-7B. The structural key is 600, the slot-effect bound comes from the same "
                          "analysis() as the frozen board, and Holm uses m=3 over none against slot."),
             "n_populasi": N, "kunci_buta_struktural": KEY, "papan": board, "label_source": "deterministic (model scores)"}
    OUT.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    MAN.write_text(json.dumps({"skrip": Path(__file__).name, "dibuat": result["dibuat"], "masukan_sha256": source,
                               "keluaran": OUT.relative_to(ROOT).as_posix(), "keluaran_sha256": sha(OUT)}, indent=1, ensure_ascii=False))
    for mdl, r in board.items():
        s = r["sel"]; rows = "  ".join(f"{c} {s[c]['menang'] if isinstance(s[c], dict) else s[c]}" for c in CELLS)
        print(f"  {mdl:44} {rows}" + (f"  bound {r['batas_efek_slot_poin_persen']}pp ci_top {r['ci_atas_poin_persen']}" if 'batas_efek_slot_poin_persen' in r else "")
              + (f"  label +{r['label_ditambahkan_pp']}pp ({r['label_ditambahkan_baris']})" if 'label_ditambahkan_pp' in r else ""))
    print("written:", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
