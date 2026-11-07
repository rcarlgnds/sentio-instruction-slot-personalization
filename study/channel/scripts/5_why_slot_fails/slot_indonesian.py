import os
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

for _stage in sorted(p for p in Path(__file__).resolve().parents[1].iterdir() if p.is_dir()):
    sys.path.insert(0, str(_stage))
from equivalence_bounds import analysis  # noqa: E402
from mcnemar import per_item_results, holm, mcnemar_exact  # noqa: E402

ROOT = Path(os.environ.get("SENTIO_ROOT")
            or Path(__file__).resolve().parents[4])
RAW = ROOT / "study/channel/results/raw"
N1 = ROOT / "study/retrieval/results/raw"
OUT = ROOT / "study/channel/results/tables/slot_id_paired.json"
MAN = RAW / "slot_id_paired.manifest.json"
CELLS = {
    "Qwen3-Reranker-0.6B": (RAW / "scores_qwen3rr_Qwen3-Reranker-0.6B_paired_profile_id.jsonl",
                            N1 / "qwen_scores_0.6b_paired_default.jsonl", N1 / "qwen_scores_0.6b_paired_profile.jsonl"),
    "Qwen3-Reranker-4B": (RAW / "scores_qwen3rr_Qwen3-Reranker-4B_paired_profile_id.jsonl",
                          N1 / "qwen_scores_4b_paired_default.jsonl", N1 / "qwen_scores_4b_paired_profile.jsonl"),
    "zerank-2": (RAW / "scores_zerank2official_zerank-2_paired_profile_id.jsonl",
                 RAW / "scores_zerank2official_zerank-2_paired_default.jsonl", RAW / "scores_zerank2official_zerank-2_paired_profile.jsonl"),
    "FollowIR-7B": (RAW / "scores_followir_FollowIR-7B_paired_slot_id.jsonl",
                    RAW / "scores_followir_FollowIR-7B_paired_none.jsonl", RAW / "scores_followir_FollowIR-7B_paired_slot.jsonl"),
    "ctxl-rerank-v2-instruct-multilingual-6b": (RAW / "scores_ctxl_ctxl-rerank-v2-instruct-multilingual-6b_paired_slot_id.jsonl",
                                                RAW / "scores_ctxl_ctxl-rerank-v2-instruct-multilingual-6b_paired_none.jsonl",
                                                RAW / "scores_ctxl_ctxl-rerank-v2-instruct-multilingual-6b_paired_slot.jsonl"),
    "bge-reranker-v2-gemma": (RAW / "scores_gemma_bge-reranker-v2-gemma_paired_slot_id.jsonl",
                              RAW / "scores_gemma_bge-reranker-v2-gemma_paired_none.jsonl", RAW / "scores_gemma_bge-reranker-v2-gemma_paired_slot.jsonl"),
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def contrast(wa, wb):
    ids = sorted(set(wa) & set(wb))
    n01 = sum(1 for q in ids if not wa[q] and wb[q])
    n10 = sum(1 for q in ids if wa[q] and not wb[q])
    p = mcnemar_exact(n01, n10)
    return {"n": len(ids), "a_kalah_b_menang": n01, "a_menang_b_kalah": n10, "p_mcnemar_exact": float(f"{p:.4g}"), "_p": p}


def main():
    cell, praw, source = {}, {}, {}
    for mdl, (fid, fnone, fslot) in CELLS.items():
        if not fid.with_suffix(".manifest.json").exists():
            cell[mdl] = {"status": "menunggu"}
            continue
        wid, wn, ws = per_item_results(fid), per_item_results(fnone), per_item_results(fslot)
        man = json.loads(fid.with_suffix(".manifest.json").read_text())
        kn, ks = contrast(wn, wid), contrast(ws, wid); praw[mdl] = kn.pop("_p"); ks.pop("_p")
        cell[mdl] = {"status": "selesai", "menang_id": sum(wid.values()), "n_komparabel": len(wid),
                    "menang_none_beku": sum(wn.values()), "menang_slot_inggris_beku": sum(ws.values()),
                    "id_vs_none": kn, "id_vs_slot_inggris": ks,
                    "batas_efek_slot_id": analysis(mdl, kn["a_kalah_b_menang"], kn["a_menang_b_kalah"], kn["n"]),
                    "instruksi_kerangka": man.get("instruksi_profil_id_kerangka"),
                    "berkas": {"id": fid.name, "none": fnone.name, "slot": fslot.name}}
        source[fid.name] = sha(fid)
    ph = holm(praw) if praw else {}
    for mdl, p in ph.items():
        cell[mdl]["id_vs_none"]["p_holm_keluarga_ID"] = p
        cell[mdl]["id_vs_none"]["signifikan_0.05"] = p < 0.05
    result = {"dibuat": datetime.now(timezone.utc).isoformat(), "config_version_lock": "0a7b2e4de856",
             "praregistrasi": "PRAREGISTRASI_LINTAS_ARSITEKTUR_ADENDUM4.md",
             "_apa_ini": ("Indonesian profile instruction in the slot, six models, paired track of 96 rows, "
                          "against the frozen none cells (Holm m=6) and the frozen English slot cells of the same "
                          "model; effect bound from the same analysis()."),
             "sel": cell, "label_source": "deterministic (model scores)"}
    OUT.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    MAN.write_text(json.dumps({"skrip": Path(__file__).name, "dibuat": result["dibuat"], "masukan_sha256": source,
                               "keluaran": OUT.relative_to(ROOT).as_posix(), "keluaran_sha256": sha(OUT)}, indent=1, ensure_ascii=False))
    for mdl, v in cell.items():
        if v["status"] != "selesai":
            print(f"  {mdl:42} pending")
            continue
        print(f"  {mdl:42} ID {v['menang_id']}/96  none {v['menang_none_beku']}  slot-EN {v['menang_slot_inggris_beku']}"
              f"  vs none {v['id_vs_none']['a_kalah_b_menang']}/{v['id_vs_none']['a_menang_b_kalah']} p={v['id_vs_none']['p_mcnemar_exact']} holm={v['id_vs_none'].get('p_holm_keluarga_ID')}"
              f"  bound {v['batas_efek_slot_id']['batas_atas_poin_persen']}pp")
    print("written:", OUT); return 0


if __name__ == "__main__":
    sys.exit(main())
