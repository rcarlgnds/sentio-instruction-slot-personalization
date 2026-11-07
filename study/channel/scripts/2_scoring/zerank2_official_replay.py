import os
import hashlib
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

for _stage in sorted(p for p in Path(__file__).resolve().parents[1].iterdir() if p.is_dir()):
    sys.path.insert(0, str(_stage))
import metrics as M  # noqa: E402
from mcnemar import mcnemar_exact  # noqa: E402

ROOT = Path(os.environ.get("SENTIO_ROOT")
            or Path(__file__).resolve().parents[4])
RAW = ROOT / "study/channel/results/raw"
TAB = ROOT / "study/channel/results/tables"
OUT = TAB / "zerank2_official_replay.json"
MAN = RAW / "zerank2_official_replay.manifest.json"
FROZEN = "scores_zerank2_zerank-2_"
OFFICIAL = "scores_zerank2official_zerank-2_"
TIE_TOL = 0.25


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return [json.loads(l) for l in Path(p).open() if l.strip()]


def positions(r):
    return {c: i for i, c in enumerate(r["ranking"])}


def wins(r):
    pos = positions(r)
    g = min((pos[c] for c in (r["gold"] or []) if c in pos), default=None)
    o = min((pos[c] for c in (r.get("counterpart") or []) if c in pos), default=None)
    if g is None or o is None:
        return None
    return g < o


def is_tie(r):
    pos = positions(r)
    sk = r.get("scores") or []
    g = [sk[pos[c]] for c in (r["gold"] or []) if c in pos and pos[c] < len(sk)]
    o = [sk[pos[c]] for c in (r.get("counterpart") or []) if c in pos and pos[c] < len(sk)]
    if not g or not o:
        return None
    return abs(max(g) - max(o)) <= TIE_TOL


def traindev_metrics(rows):
    nd, rr, r1 = [], [], []
    for r in rows:
        gold = set(r["gold"] or [])
        rank = r["ranking"]
        nd.append(M._ndcg_at_k(rank, gold))
        rr.append(next((1 / (i + 1) for i, c in enumerate(rank) if c in gold), 0.0))
        r1.append(1.0 if rank and rank[0] in gold else 0.0)
    return {"n": len(nd), "ndcg5": round(statistics.fmean(nd), 4),
            "mrr": round(statistics.fmean(rr), 4), "top1_hit": round(statistics.fmean(r1), 4)}


def holdout_metrics(path):
    from posthoc_holdout_metrics import metrics
    return metrics(Path(path))


def control_sign_test(rows):
    per_pair = {}
    for r in rows:
        pid = r["query_id"].rsplit("-", 1)[0]
        w = wins(r)
        if w is not None:
            per_pair[pid] = per_pair.get(pid, 0) + int(w)
    delta = [v - 1 for v in per_pair.values()]
    plus = sum(1 for d in delta if d > 0)
    minus = sum(1 for d in delta if d < 0)
    n = plus + minus
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(0, min(plus, minus) + 1)) / 2 ** n)
    return {"n_pasangan": len(per_pair), "menang_total": sum(per_pair.values()),
            "pasangan_naik": plus, "pasangan_turun": minus, "p_dua_sisi_exact": float(f"{p:.4g}")}


def compare_cells(suffix):
    fb, fr = RAW / f"{FROZEN}{suffix}.jsonl", RAW / f"{OFFICIAL}{suffix}.jsonl"
    mb, mr = fb.with_suffix(".manifest.json"), fr.with_suffix(".manifest.json")
    if not mr.exists():
        return {"status": "menunggu", "berkas_resmi": fr.name}
    if not mb.exists() or not fb.exists():
        return {"status": "berkas_beku_hilang", "berkas_resmi": fr.name, "berkas_beku": fb.name,
                "manifest_beku_ada": mb.exists(), "sha256_resmi": sha(fr)}
    A, B = {r["query_id"]: r for r in read(fb)}, {r["query_id"]: r for r in read(fr)}
    ids = sorted(set(A) & set(B))
    result = {"status": "selesai", "n_baris_beku": len(A), "n_baris_resmi": len(B), "n_baris_sama": len(ids),
             "ranking_identik": sum(1 for q in ids if A[q]["ranking"] == B[q]["ranking"]),
             "top1_identik": sum(1 for q in ids if A[q]["ranking"][:1] == B[q]["ranking"][:1]),
             "sha256_beku": sha(fb), "sha256_resmi": sha(fr)}
    has_counterpart = any(A[q].get("counterpart") for q in ids)
    if has_counterpart:
        wa = {q: wins(A[q]) for q in ids}
        wb = {q: wins(B[q]) for q in ids}
        comp = [q for q in ids if wa[q] is not None and wb[q] is not None]
        b = sum(1 for q in comp if wa[q] and not wb[q])
        c = sum(1 for q in comp if wb[q] and not wa[q])
        result.update({"n_komparabel": len(comp),
                      "menang_beku": sum(1 for q in comp if wa[q]), "menang_resmi": sum(1 for q in comp if wb[q]),
                      "diskordan_beku_saja": b, "diskordan_resmi_saja": c,
                      "p_mcnemar_beku_vs_resmi": float(f"{mcnemar_exact(b, c):.4g}"),
                      "seri_dalam_presisi_beku": sum(1 for q in comp if is_tie(A[q])),
                      "seri_dalam_presisi_resmi": sum(1 for q in comp if is_tie(B[q])),
                      "diskordan_yang_seri_di_beku": sum(1 for q in comp if wa[q] != wb[q] and is_tie(A[q]))})
        pb = M.paired_metrics([A[q] for q in ids]); pr = M.paired_metrics([B[q] for q in ids])
        result["metrik_paired"] = {"beku": pb, "resmi": pr}
        if suffix.startswith("control_"):
            result["uji_tanda_pasangan"] = {"beku": control_sign_test([A[q] for q in ids]),
                                           "resmi": control_sign_test([B[q] for q in ids])}
    if suffix.startswith("traindev_"):
        result["metrik_traindev"] = {"beku": traindev_metrics([A[q] for q in ids]),
                                    "resmi": traindev_metrics([B[q] for q in ids])}
        result["metrik_traindev_ir_measures"] = {"beku": holdout_metrics(fb), "resmi": holdout_metrics(fr)}
    if suffix.startswith("holdout_"):
        result["metrik_holdout_ir_measures"] = {"beku": holdout_metrics(fb), "resmi": holdout_metrics(fr)}
    return result


def compare_tables(frozen_name, official_name, key):
    pb, pr = TAB / frozen_name, TAB / official_name
    if not pr.exists():
        return {"status": "menunggu", "berkas_resmi": official_name}
    A, B = json.loads(pb.read_text()), json.loads(pr.read_text())
    out = {"status": "selesai", "sha256_beku": sha(pb), "sha256_resmi": sha(pr)}
    for k in key:
        out[k] = {"beku": A.get(k), "resmi": B.get(k)}
    return out


PAIRED = {"none": "paired_default", "slot": "paired_profile", "text": "paired_default_docmeta",
          "both": "paired_profile_docmeta", "wrong": "paired_profile_docmetasalah",
          "fmttag": "paired_profile_fmttag", "fmtlabeled": "paired_profile_fmtlabeled",
          "vmem": "paired_profile_vmem", "program_only": "paired_profile_docmeta_fieldprogram",
          "semester_only": "paired_profile_docmeta_fieldsemester"}
XPAIR = {"none": "xpair_default", "slot": "xpair_profile", "both": "xpair_profile_docmeta",
         "wrong": "xpair_profile_docmetasalah"}
CONTRASTS = [("paired", "none", "slot", True), ("paired", "slot", "both", False), ("paired", "text", "both", False),
           ("paired", "wrong", "both", False), ("paired", "slot", "fmttag", False), ("paired", "slot", "fmtlabeled", False),
           ("paired", "slot", "vmem", False), ("paired", "program_only", "both", False),
           ("paired", "semester_only", "both", False),
           ("xpair", "none", "slot", True), ("xpair", "slot", "both", False), ("xpair", "wrong", "both", False)]


def winning_items(prefix, suffix):
    fp = RAW / f"{prefix}{suffix}.jsonl"
    if not fp.exists() or not fp.with_suffix(".manifest.json").exists():
        return None
    from mcnemar import per_item_results
    return per_item_results(fp)


def track_contrast():
    from equivalence_bounds import analysis
    output = {}
    for track, a, b, has_bound in CONTRASTS:
        mapping = PAIRED if track == "paired" else XPAIR
        rec = {}
        for route, prefix in (("beku", FROZEN), ("resmi", OFFICIAL)):
            wa, wb = winning_items(prefix, mapping[a]), winning_items(prefix, mapping[b])
            if wa is None or wb is None:
                rec[route] = {"status": "menunggu"}; continue
            ids = sorted(set(wa) & set(wb))
            n01 = sum(1 for q in ids if not wa[q] and wb[q])
            n10 = sum(1 for q in ids if wa[q] and not wb[q])
            r = {"status": "selesai", "n": len(ids), "menang_a": sum(1 for q in ids if wa[q]),
                 "menang_b": sum(1 for q in ids if wb[q]), "a_kalah_b_menang": n01, "a_menang_b_kalah": n10,
                 "selisih_bersih_pp": round((n01 - n10) / len(ids) * 100, 2) if ids else None,
                 "p_mcnemar_exact": float(f"{mcnemar_exact(n01, n10):.4g}")}
            if has_bound:
                an = analysis("zerank-2", n01, n10, len(ids))
                r["batas_atas_poin_persen"] = an["batas_atas_poin_persen"]
                r["dilaporkan95_pp"] = [round(x * 100, 2) for x in an["dilaporkan95"]]
            rec[route] = r
        output[f"{track}:{a}_vs_{b}"] = rec
    return output


def path_controls():
    import aggregate_controls as AC
    pool_rows = M.read_ranking(AC.POOL)
    output = {}
    for tag, filtered_only in (("tersaring", True), ("penuh", False)):
        used_pairs, used_ids, _ = AC.population(tag, pool_rows, filtered_only)
        rec = {}
        for route, prefix in (("beku", FROZEN), ("resmi", OFFICIAL)):
            d, k = winning_items(prefix, "control_default"), winning_items(prefix, "control_kondisi")
            if d is None or k is None:
                rec[route] = {"status": "menunggu"}; continue
            gains = drops = ties = n_both = n_one = 0
            for pid in used_pairs:
                qs = [f"{pid}-contoh", f"{pid}-definisi"]
                if any(q not in d or q not in k for q in qs):
                    continue
                wd, wk = sum(1 for q in qs if d[q]), sum(1 for q in qs if k[q])
                n_both += wk == 2; n_one += wk == 1
                gains += wk > wd; drops += wk < wd; ties += wk == wd
            n = gains + drops + ties
            rec[route] = {"status": "selesai", "n_pasangan": n, "kondisi_lebih_baik": gains,
                          "kondisi_lebih_buruk": drops, "seri": ties,
                          "delta_pairwise_pp": round(((2 * n_both + n_one) / (2 * n) - 0.5) * 100, 2) if n else None,
                          "p_dua_sisi_exact": float(f"{AC.binom_two_sided(gains, drops):.4g}")}
        output[tag] = rec
    return output


def main():
    suffix = sorted({p.name[len(OFFICIAL):-len(".jsonl")] for p in RAW.glob(f"{OFFICIAL}*.jsonl")}
                    - {s for s in [] })
    suffix = [s for s in suffix if "smoke" not in s and "equivfull" not in s]
    cell = {s: compare_cells(s) for s in suffix}
    core17 = compare_tables("followir_Core17_zerank2_zerank-2.json", "followir_Core17_zerank2official_zerank-2.json",
                           ["p_MRR", "ndcg5_og", "ndcg5_changed", "n_kueri"])
    probe = compare_tables("injection_probe_zerank2_zerank-2.json", "injection_probe_zerank2official_zerank-2.json",
                          ["agregat_per_kombinasi"])
    if probe["status"] == "selesai":
        A = json.loads((TAB / "injection_probe_zerank2_zerank-2.json").read_text())["per_sampel"]
        B = json.loads((TAB / "injection_probe_zerank2official_zerank-2.json").read_text())["per_sampel"]
        bb = {r["query_id"]: r for r in B}
        agree = tot = 0
        for r in A:
            for k, v in r["kombinasi"].items():
                tot += 1
                agree += int(bb[r["query_id"]]["kombinasi"][k]["menang"] == v["menang"])
        probe["keputusan_menang_identik"] = f"{agree}/{tot}"
    contrast = track_contrast()
    control = path_controls()
    result = {
        "dibuat": datetime.now(timezone.utc).isoformat(),
        "config_version_lock": "0a7b2e4de856",
        "praregistrasi": "PRAREGISTRASI_LINTAS_ARSITEKTUR_ADENDUM2.md section (a); the rule for switching to the official path: PRAREGISTRASI_ADENDUM_KESETARAAN.md section 3",
        "_apa_ini": ("Frozen zerank-2 cells (the project's own bf16 scorer) against a replay through the "
                     "vendor's official CrossEncoder.predict(): the same summary recomputed from both files, "
                     "the per-row agreement, and the rows tied within precision (|gold score minus partner "
                     "score| <= 0.25). No GPU and no rescoring. The cells counted in n_sel_menunggu were "
                     "scored only on the vendor's path (the ladder cells, paired_profile_id, and "
                     "control_profile), so they have no frozen file of the project's scorer to compare with "
                     "(status berkas_beku_hilang)."),
        "ambang_seri": TIE_TOL,
        "n_sel_selesai": sum(1 for v in cell.values() if v["status"] == "selesai"),
        "n_sel_menunggu": sum(1 for v in cell.values() if v["status"] != "selesai"),
        "sel": cell, "kontras_naskah_per_jalur": contrast, "kontrol_tekstual_per_jalur": control,
        "core17": core17, "probe_injeksi": probe,
        "label_source": "deterministic (frozen scores of both paths)",
    }
    OUT.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    MAN.write_text(json.dumps({"skrip": Path(__file__).name, "dibuat": result["dibuat"],
                               "config_version_lock": "0a7b2e4de856", "keluaran": OUT.relative_to(ROOT).as_posix(),
                               "keluaran_sha256": sha(OUT),
                               "masukan_sha256": {k: {"beku": v.get("sha256_beku"), "resmi": v.get("sha256_resmi")}
                                                  for k, v in cell.items() if v["status"] == "selesai"}},
                              indent=1, ensure_ascii=False))
    for s, v in cell.items():
        if v["status"] != "selesai":
            print(f"  {s:44} {v['status']}"); continue
        records = f"  {s:44} rows {v['n_baris_sama']:5}  rank= {v['ranking_identik']:5}  top1= {v['top1_identik']:5}"
        if "menang_beku" in v:
            records += (f"  wins {v['menang_beku']}->{v['menang_resmi']}  disc {v['diskordan_beku_saja']}/{v['diskordan_resmi_saja']}"
                      f" p={v['p_mcnemar_beku_vs_resmi']}  ties {v['seri_dalam_presisi_beku']} (disc&tie {v['diskordan_yang_seri_di_beku']})")
        if "metrik_traindev" in v:
            records += f"  traindev {v['metrik_traindev']['beku']} -> {v['metrik_traindev']['resmi']}"
        if "metrik_holdout_ir_measures" in v:
            records += f"  holdout {v['metrik_holdout_ir_measures']['beku']} -> {v['metrik_holdout_ir_measures']['resmi']}"
        if "uji_tanda_pasangan" in v:
            records += f"  sign {v['uji_tanda_pasangan']['beku']} -> {v['uji_tanda_pasangan']['resmi']}"
        print(records)
    for k, v in contrast.items():
        if v["beku"].get("status") == "selesai" and v["resmi"].get("status") == "selesai":
            b, r = v["beku"], v["resmi"]
            print(f"  [{k:32}] frozen {b['menang_a']}->{b['menang_b']} disc {b['a_kalah_b_menang']}/{b['a_menang_b_kalah']} p={b['p_mcnemar_exact']}"
                  + (f" bound {b['batas_atas_poin_persen']}pp" if 'batas_atas_poin_persen' in b else "")
                  + f" | official {r['menang_a']}->{r['menang_b']} disc {r['a_kalah_b_menang']}/{r['a_menang_b_kalah']} p={r['p_mcnemar_exact']}"
                  + (f" bound {r['batas_atas_poin_persen']}pp" if 'batas_atas_poin_persen' in r else ""))
        else:
            print(f"  [{k:32}] pending")
    for k, v in control.items():
        print(f"  control {k:9} frozen {v['beku']} | official {v['resmi']}")
    print("  core17:", core17.get("status"), {k: core17[k] for k in core17 if k in ("p_MRR", "ndcg5_og")})
    print("  probe :", probe.get("status"), probe.get("keputusan_menang_identik"))
    print("written:", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
