import os
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("SENTIO_ROOT")
            or Path(__file__).resolve().parents[4])

for _stage in sorted(p for p in Path(__file__).resolve().parents[1].iterdir() if p.is_dir()):
    sys.path.insert(0, str(_stage))
from equivalence_bounds import analysis  # noqa: E402
from mcnemar import per_item_results, holm, mcnemar_exact  # noqa: E402

RAW = ROOT / "study/channel/results/raw"
OUT = ROOT / "study/channel/results/tables/vendor_style_slot.json"
MAN = RAW / "vendor_style_slot.manifest.json"
MODEL = {"followir": "FollowIR-7B", "ctxl": "ctxl-rerank-v2-instruct-multilingual-6b", "gemma": "bge-reranker-v2-gemma"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def contrast(wa, wb):
    ids = sorted(set(wa) & set(wb))
    n01 = sum(1 for q in ids if not wa[q] and wb[q])
    n10 = sum(1 for q in ids if wa[q] and not wb[q])
    return {"n": len(ids), "a_kalah_b_menang": n01, "a_menang_b_kalah": n10,
            "p_mcnemar_exact": float(f"{mcnemar_exact(n01, n10):.4g}"), "_p": mcnemar_exact(n01, n10)}


def main():
    cell, praw, source = {}, {}, {}
    for fam, mdl in MODEL.items():
        fv = RAW / f"scores_{fam}_{mdl}_paired_slot_vendor.jsonl"
        if not fv.with_suffix(".manifest.json").exists():
            cell[mdl] = {"status": "menunggu"}
            continue
        wv = per_item_results(fv)
        wn = per_item_results(RAW / f"scores_{fam}_{mdl}_paired_none.jsonl")
        ws = per_item_results(RAW / f"scores_{fam}_{mdl}_paired_slot.jsonl")
        man = json.loads(fv.with_suffix(".manifest.json").read_text())
        kn, ks = contrast(wn, wv), contrast(ws, wv)
        praw[mdl] = kn.pop("_p"); ks.pop("_p")
        cell[mdl] = {"status": "selesai", "menang_vendor": sum(wv.values()), "n_komparabel": len(wv),
                    "menang_none_beku": sum(wn.values()), "menang_slot_beku": sum(ws.values()),
                    "vendor_vs_none": kn, "vendor_vs_slot": ks,
                    "batas_efek_slot_vendor": analysis(mdl, kn["a_kalah_b_menang"], kn["a_menang_b_kalah"], kn["n"]),
                    "instruksi_vendor_kerangka": man.get("instruksi_vendor_kerangka"),
                    "penempatan": man.get("instruksi_vendor_penempatan")}
        source[fv.name] = sha(fv)
    ph = holm(praw) if praw else {}
    for mdl, p in ph.items():
        cell[mdl]["vendor_vs_none"]["p_holm_keluarga_V"] = p
        cell[mdl]["vendor_vs_none"]["signifikan_0.05"] = p < 0.05
    result = {"dibuat": datetime.now(timezone.utc).isoformat(), "config_version_lock": "0a7b2e4de856",
             "praregistrasi": "PRAREGISTRASI_LINTAS_ARSITEKTUR_ADENDUM3.md",
             "_apa_ini": ("The slot_vendor cells (the profile in the slot, in the sentence style each vendor "
                          "documents) against the frozen none and slot cells of the same model; exact McNemar, "
                          "Holm m=3 over vendor against none, slot-effect bound from the same analysis(). No "
                          "frozen cell rescored."),
             "sel": cell, "label_source": "deterministic (model scores)"}
    OUT.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    MAN.write_text(json.dumps({"skrip": Path(__file__).name, "dibuat": result["dibuat"], "masukan_sha256": source,
                               "keluaran": OUT.relative_to(ROOT).as_posix(), "keluaran_sha256": sha(OUT)}, indent=1, ensure_ascii=False))
    for mdl, v in cell.items():
        if v["status"] != "selesai":
            print(f"  {mdl:44} pending")
            continue
        print(f"  {mdl:44} vendor {v['menang_vendor']}/96  none {v['menang_none_beku']}  slot {v['menang_slot_beku']}"
              f"  vs none {v['vendor_vs_none']['a_kalah_b_menang']}/{v['vendor_vs_none']['a_menang_b_kalah']} p={v['vendor_vs_none']['p_mcnemar_exact']}"
              f" holm={v['vendor_vs_none'].get('p_holm_keluarga_V')}  bound {v['batas_efek_slot_vendor']['batas_atas_poin_persen']}pp")
    print("written:", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
