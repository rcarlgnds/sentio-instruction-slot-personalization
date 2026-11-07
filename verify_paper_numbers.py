"""Checks the numbers printed in the paper against the files they come from.

Each check names a file and a key. It reads the value there, or recomputes it when the paper prints a
derived value such as a range, a difference, a sign test, or a Holm correction for multiple tests. The
exit code is 1 if any check fails.

    python verify_paper_numbers.py [package root] [report.md]
"""
import json
import math
import re
import sys
from pathlib import Path

import tempfile
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(tempfile.gettempdir()) / "verify_paper_numbers_report.md"
T = "study/channel/results/tables/"
R = "study/retrieval/results/tables/"
_cache = {}


def J(rel):
    if rel not in _cache:
        _cache[rel] = json.loads((ROOT / rel).read_text(encoding="utf-8"))
    return _cache[rel]


rows = []


def check(section, claim, paper_value, package_value, source, ok=None):
    if ok is None:
        ok = paper_value == package_value
    rows.append((section, claim, str(paper_value), str(package_value), source, "match" if ok else "DIFFER"))
    return ok


def rng(vals):
    return (min(vals), max(vals))


def binom_two_sided(k, n):
    p_le = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    p_ge = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * min(p_le, p_ge))


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    m = len(ps)
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, ps[i] * (m - rank))
        adj[i] = min(1.0, running)
    return adj


def pick(d, *names):
    for n in names:
        if n in d:
            return d[n]
    raise KeyError(names)


def find_key(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = find_key(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_key(v, key)
            if r is not None:
                return r
    return None


QW = ["Qwen3-Reranker-0.6B", "Qwen3-Reranker-4B"]
ARCH_MODELS = {"FollowIR-7B": "FollowIR-7B", "ctxl-6B": "ctxl-rerank-v2-instruct-multilingual-6b",
       "bge-gemma": "bge-reranker-v2-gemma"}
SIX = ["Qwen3-Reranker-0.6B", "Qwen3-Reranker-4B", "zerank-2", "FollowIR-7B", "ctxl-6B", "gemma"]

cs = J("study/_shared/corpus_snapshot.json")
check("3.1", "40 textbooks", 40, cs["n_documents"], "study/_shared/corpus_snapshot.json: n_documents")
check("3.1", "80,738 indexed chunks", 80738, cs["n_chunks"], "corpus_snapshot.json: n_chunks")
lk = J(T + "leakage_audit_v2.json")
check("3.1, Table 4 (b)", "64,108 child chunks", 64108, lk["korpus_penuh"]["n_chunk"], "leakage_audit_v2.json: korpus_penuh.n_chunk")
check("Table 4 (b), 1", "0.9% of child chunks carry a program marker", 0.009, lk["korpus_penuh"]["rate"], "leakage_audit_v2.json: korpus_penuh.rate")
check("3.5, Table 4 (b)", "four marker patterns (course code, program phrase, curriculum phrase, institution name)", 4, len(lk["definisi_penanda"]), "leakage_audit_v2.json: definisi_penanda")
dl = J("study/_shared/document_licenses.json")
check("3.1", "12 books in the releasable_with_conditions category of the license record", 12, dl["summary"]["releasable_with_conditions"]["documents"], "document_licenses.json: summary.releasable_with_conditions.documents")
check("3.1", "28 books in the releasable category of the license record (40 - 12)", 28, dl["summary"]["releasable"]["documents"], "document_licenses.json: summary.releasable.documents")
licenses = [d["license"] for d in dl["documents"]]
licenses_ok = all(("CC BY" in s) or ("MIT" in s) or ("Public Domain" in s) for s in licenses)
vers = sorted({m.group(1) for s in licenses for m in [re.search(r"CC BY (\d\.\d)", s)] if m})
check("3.1", "licenses are CC BY 2.0 to 4.0, MIT, or public domain", "CC BY/MIT/PD, versions 2.0..4.0",
      f"{'all match' if licenses_ok else 'mismatch'}, CC BY versions {vers}", "document_licenses.json: documents[].license", ok=licenses_ok and vers[0] == "2.0" and vers[-1] == "4.0")
ab = J(T + "benchmark_provenance.json")
check("3.1", "2,776 benchmark queries", 2776, ab["n_kueri_diterima"], "benchmark_provenance.json: n_kueri_diterima")
check("3.1", "generation records name claude-fable-5 for all 2,776", 2776, ab["model"]["claude-fable-5"], "benchmark_provenance.json: model.claude-fable-5")
check("3.1", "written between July 25 and August 2, 2026", "2026-07-25 .. 2026-08-02",
      ab["generasi_paling_awal_utc"][:10] + " .. " + ab["generasi_paling_akhir_utc"][:10], "benchmark_provenance.json: generasi_paling_awal_utc, generasi_paling_akhir_utc",
      ok=ab["generasi_paling_awal_utc"].startswith("2026-07-25") and ab["generasi_paling_akhir_utc"].startswith("2026-08-02"))
check("3.4", "first gate version admitted 1,412 of 2,776", 1412, ab["protokol_gerbang_qc"]["v1-cps-gate"], "benchmark_provenance.json: protokol_gerbang_qc.v1-cps-gate")
cl = J("study/_shared/config_lock.json")
check("3.4", "gate threshold 0.55", 0.55, cl["static_fingerprint"]["gates"][0], "config_lock.json: static_fingerprint.gates[0]")
sg = J(T + "gate_strata.json")
check("3.4", "mining cap reached before any second-version query", "last source query position < 1412",
      f"position {sg['urutan_berkas_kueri']['posisi_kueri_asal_terakhir_yang_dipakai']} < 1412",
      "gate_strata.json: urutan_berkas_kueri.posisi_kueri_asal_terakhir_yang_dipakai", ok=sg["urutan_berkas_kueri"]["posisi_kueri_asal_terakhir_yang_dipakai"] < 1412)

sp = J(T + "pool_statistics.json")
tb, tl = sp["track_berpasangan"], sp["track_luas"]
check("3.2", "400 cross-book topic pairs", 400, tb["pasangan_topik_kandidat"], "pool_statistics.json: track_berpasangan.pasangan_topik_kandidat")
check("3.2", "52 pairs", 52, tb["pasangan"], "pool_statistics.json: track_berpasangan.pasangan")
check("3.2", "48 pairs with both golds in the pool, 96 comparable rows", (48, 96), (tb["pasangan_kedua_gold_di_pool"], tb["baris_komparabel"]), "pool_statistics.json: pasangan_kedua_gold_di_pool, baris_komparabel")
check("3.2", "four pairs dropped", 4, len(tb["pasangan_gugur_karena_gold_di_luar_pool"]), "pool_statistics.json: pasangan_gugur_karena_gold_di_luar_pool")
check("3.3", "50 passages per search branch, RRF k = 60, cut to 80", (50, 60, 80), (tb["batas_pencarian_per_cabang_di_kode"][0], tb["rrf_k_config_lock"], tb["retrieval_pool_size_config_lock"]), "pool_statistics.json: batas_pencarian_per_cabang_di_kode, rrf_k_config_lock, retrieval_pool_size_config_lock")
check("3.3", "64 to 80 unique child chunks per question", (64, 80), (tb["kandidat_unik_per_baris"]["min"], tb["kandidat_unik_per_baris"]["max"]), "pool_statistics.json: track_berpasangan.kandidat_unik_per_baris")
check("3.3 (provenance, not printed in the paper)", "longest candidate 1,847 characters, under the 4,000 limit", (1847, True), (max(tb["teks_kandidat_terpanjang_karakter"], tl["teks_kandidat_terpanjang_karakter"]), max(tb["teks_kandidat_terpanjang_karakter"], tl["teks_kandidat_terpanjang_karakter"]) < 4000), "pool_statistics.json: teks_kandidat_terpanjang_karakter (both tracks)")
lim = sum(1 for p in (ROOT / "study/channel/scripts").rglob("*.py") if "[:4000]" in p.read_text(encoding="utf-8", errors="ignore"))
check("3.3", "4,000-character text limit in the scoring scripts", ">= 1 script", f"{lim} scripts truncate at [:4000]", "study/channel/scripts/*/*.py", ok=lim >= 1)
check("3.4", "600 groups, 1,621 rows", (600, 1621), (tl["grup"], tl["baris"]), "pool_statistics.json: track_luas.grup, baris")
check("3.4", "179 pairs and 421 triples", (179, 421), (tl["ukuran_grup"]["2"], tl["ukuran_grup"]["3"]), "pool_statistics.json: track_luas.ukuran_grup")
check("3.4", "68 to 70 candidates per row", (68, 70), (tl["kandidat_unik_per_baris"]["min"], tl["kandidat_unik_per_baris"]["max"]), "pool_statistics.json: track_luas.kandidat_unik_per_baris")
src = (ROOT / "study/channel/scripts/1_tracks/expand_paired_track.py").read_text(encoding="utf-8", errors="ignore")
check("3.4", "similarity window 0.66 to 0.89, word overlap (Jaccard) at most 0.36", True, all(s in src for s in ("default=0.66", "default=0.89", "default=0.36")), "study/channel/scripts/1_tracks/expand_paired_track.py: --ambang, --ambang-atas, --jaccard-maks")
dx = J(T + "dits_xpair.json")
check("3.4", "blind baseline 600 of 1,621 (0.3701)", (600, 0.3701), (int(dx["arms"]["S1"]["menang_eksak"].split("/")[0]), round(dx["arms"]["S1"]["pairwise"], 4)), "dits_xpair.json: arms.S1.menang_eksak, pairwise")

tm = {m["repo"].split("/")[-1].lower(): m for m in J(T + "model_table.json")["model"]}
table2 = [("Qwen3-Reranker-0.6B", "Qwen/Qwen3-Reranker-0.6B", "e61197ed4502", "float16", 2048, "apache-2.0"),
          ("Qwen3-Reranker-4B", "Qwen/Qwen3-Reranker-4B", "22e683669bc0", "float16", 2048, "apache-2.0"),
          ("zerank-2", "zeroentropy/zerank-2", "5eae30d5ee3c", "bfloat16", 1536, "apache-2.0"),
          ("FollowIR-7B", "jhu-clsp/FollowIR-7B", "4d25d437e38b", "bfloat16", 1536, "apache-2.0"),
          ("ctxl-rerank-v2-instruct-multilingual-6b", "ContextualAI/ctxl-rerank-v2-instruct-multilingual-6b", "f14ca1a2fc20", "bfloat16", 1536, "cc-by-nc-sa-4.0"),
          ("bge-reranker-v2-gemma", "BAAI/bge-reranker-v2-gemma", "1787044f8b6f", "bfloat16", 1536, "apache-2.0")]
for name, repo, rev, prec, tok, lic in table2:
    e = tm[repo.split("/")[-1].lower()]
    got = (e["repo"], e["revisi"][:12], e["presisi"], e["anggaran_token"], e["lisensi_kartu_model"].lower())
    check("Table 2", f"{name}: repo, revision, precision, tokens, license", (repo, rev, prec, tok, lic), got, "model_table.json: model[].repo, revisi, presisi, anggaran_token, lisensi_kartu_model")
z = J(T + "zerank2_official_replay.json")
diffs = {k: abs(v["menang_beku"] - v["menang_resmi"]) for k, v in z["sel"].items() if "menang_resmi" in v and "menang_beku" in v}
check("3.5", "zerank-2 resolved counts on the model-card path differ from the harness by at most two per condition", "<= 2", f"max {max(diffs.values())} ({len(diffs)} cells)", "zerank2_official_replay.json: sel.*.menang_beku vs menang_resmi", ok=max(diffs.values()) <= 2)
bg = J(T + "gpu_cost.json")["papan"]["paired"]
med = [bg[m]["median_menit_per_sel"] for m in bg]
check("3.3 (provenance, not printed in the paper)", "on one 16 GB GPU: median 1.4 to 9.4 minutes per paired-track condition", (1.4, 9.4), rng(med), "gpu_cost.json: papan.paired.*.median_menit_per_sel")
lm = J(T + "machine_environment.json")
check("3.1", "AMD Ryzen 9 9950X3D, 96 GB, RTX 5080 16 GB, Ubuntu 24.04", True,
      ("9950X3D" in lm["prosesor"]) and lm["memori_terpasang_gb"] == 96 and ("5080" in lm["gpu"]) and round(lm["vram_mib"] / 1024) == 16 and lm["os"].startswith("Ubuntu 24.04"),
      "machine_environment.json: prosesor, memori_terpasang_gb, gpu, vram_mib, os")

fk = J(T + "followir_calibration.json")
check("3.5", "208 query-instruction instances from 104 topics", (208, 104), (sum(v["n_kueri"] for v in fk["setelan_per_koleksi"].values()), sum(v["n_kueri"] for v in fk["setelan_per_koleksi"].values()) // 2), "followir_calibration.json: setelan_per_koleksi.*.n_kueri (40 + 64 + 104)")
check("Table 4 (a)", "all six models get a positive p-MRR on all three collections (18 of 18 cells)", (18, 18), (fk["n_sel"], fk["n_sel_positif"]), "followir_calibration.json: n_sel, n_sel_positif")
pm = []
for col, blk in fk["papan"].items():
    pm.append(blk["acuan_p_mrr"])
    pm += [m["p_mrr"] for m in blk["model"].values()]
check("Table 4 (a)", "p-MRR from +0.013 to +0.147", (0.013, 0.147), (round(min(pm), 3), round(max(pm), 3)), "followir_calibration.json: papan.*.acuan_p_mrr and model.*.p_mrr")
fi7 = tuple(round(fk["papan"][c]["model"]["FollowIR-7B"]["p_mrr"], 3) for c in ("Robust04", "News21", "Core17"))
check("Table 4 (a)", "our code gives FollowIR-7B +0.135, +0.049, +0.140 on Robust04, News21, Core17", (0.135, 0.049, 0.14), fi7, "followir_calibration.json: papan.{Robust04,News21,Core17}.model.FollowIR-7B.p_mrr")
gm = tuple(round(pick(fk["papan"][c]["model"], "gemma", "bge-gemma", ARCH_MODELS["bge-gemma"])["p_mrr"], 3) for c in ("Robust04", "News21", "Core17"))
_least = all(round(pick(fk["papan"][c]["model"], "gemma", "bge-gemma", ARCH_MODELS["bge-gemma"])["p_mrr"], 6)
             == round(min(m["p_mrr"] for m in fk["papan"][c]["model"].values()), 6) for c in ("Robust04", "News21", "Core17"))
check("Table 4 (a)", "bge-gemma responds the least, at +0.013 to +0.049", (0.013, 0.049, True), (min(gm), max(gm), _least), "followir_calibration.json: papan.{Robust04,News21,Core17}.model.*.p_mrr (bge-gemma lowest of the six on each collection)")
sb = J(T + "separability_probe.json")
check("Table 4 (b)", "TF-IDF classifier 95% on sections it has not seen, 56% by always guessing the larger book", (95, 56), (round(100 * sb["cv_grouped_by_section"]["akurasi_mean"]), round(100 * sb["baseline_kelas_mayoritas"])), "separability_probe.json: cv_grouped_by_section.akurasi_mean, baseline_kelas_mayoritas")
se = J(T + "separability_probe_embedding.json")
check("Table 4 (b)", "linear classifier on BGE-M3 embeddings 78%", 78, round(100 * se["probe_linear_logreg"]["akurasi_mean"]), "separability_probe_embedding.json: probe_linear_logreg.akurasi_mean")
ct = J(T + "control_textual.json")
check("3.5, Table 4 (c)", "58 topic pairs, 36 filtered", (58, 36), (ct["populasi_penuh"]["n_pasangan"], ct["populasi_tersaring"]["n_pasangan"]), "control_textual.json: populasi_penuh.n_pasangan, populasi_tersaring.n_pasangan")
zt = z["kontrol_tekstual_per_jalur"]
d36 = (round(100 * ct["ringkasan"]["tersaring"]["delta_pairwise"]["Qwen3-Reranker-0.6B"], 1), round(100 * ct["ringkasan"]["tersaring"]["delta_pairwise"]["Qwen3-Reranker-4B"], 1), round(zt["tersaring"]["resmi"]["delta_pairwise_pp"], 1))
check("Table 4 (c)", "worked-example gains 6.9 to 34.7 points on the 36 filtered pairs", (6.9, 34.7), (min(d36), max(d36)), "control_textual.json: ringkasan.tersaring.delta_pairwise (Qwen3); zerank2_official_replay.json: kontrol_tekstual_per_jalur.tersaring.resmi.delta_pairwise_pp")
d58 = (round(100 * ct["ringkasan"]["penuh"]["delta_pairwise"]["Qwen3-Reranker-0.6B"], 1), round(100 * ct["ringkasan"]["penuh"]["delta_pairwise"]["Qwen3-Reranker-4B"], 1), round(zt["penuh"]["resmi"]["delta_pairwise_pp"], 1))
check("Table 4 (c)", "on all 58 pairs 7.8, 16.4, 30.2 points", (7.8, 16.4, 30.2), d58, "control_textual.json: ringkasan.penuh.delta_pairwise; zerank2_official_replay.json: kontrol_tekstual_per_jalur.penuh.resmi")
ut = ct["populasi_tersaring"]["model"]
ps = [binom_two_sided(ut[m]["uji_tanda_pasangan"]["kondisi_lebih_buruk"], ut[m]["uji_tanda_pasangan"]["kondisi_lebih_baik"] + ut[m]["uji_tanda_pasangan"]["kondisi_lebih_buruk"]) for m in QW]
zr = zt["tersaring"]["resmi"]
ps.append(binom_two_sided(zr["kondisi_lebih_buruk"], zr["kondisi_lebih_baik"] + zr["kondisi_lebih_buruk"]))
hp = holm(ps)
check("Table 4 (c)", "on the 36 filtered pairs the smallest model falls short of significance (Holm p = 0.125)", 0.125, round(hp[0], 3), "recomputed from control_textual.json: populasi_tersaring.model.*.uji_tanda_pasangan and zerank2_official_replay.json (official path), Holm m = 3")
ut58 = ct["populasi_penuh"]["model"]
ps58 = [binom_two_sided(ut58[m]["uji_tanda_pasangan"]["kondisi_lebih_buruk"], ut58[m]["uji_tanda_pasangan"]["kondisi_lebih_baik"] + ut58[m]["uji_tanda_pasangan"]["kondisi_lebih_buruk"]) for m in QW]
zr58 = zt["penuh"]["resmi"]
ps58.append(binom_two_sided(zr58["kondisi_lebih_buruk"], zr58["kondisi_lebih_baik"] + zr58["kondisi_lebih_buruk"]))
hp58 = holm(ps58)
check("Table 4 (c)", "on all 58 pairs all three gains are significant, Holm sign-test p at most 0.012", (0.012, True), (round(max(hp58), 3), all(p < 0.05 for p in hp58)), "recomputed from control_textual.json: populasi_penuh.model.*.uji_tanda_pasangan and zerank2_official_replay.json (official path), Holm m = 3")
ul = J(T + "channel_tests_full.json")
bw = {r["model"]: r for r in ul["uji"] if r["track"] == "xpair" and r["kontras"] == "both_lawan_wrong"}
FOUR = ("Qwen3-Reranker-0.6B", "zerank-2", "FollowIR-7B", "ctxl-6B")
loss4 = tuple(bw[m]["menang_a"] - bw[m]["menang_b"] for m in FOUR)
check("4.4", "broad track, wrong program in the label costs Qwen3-0.6B, zerank-2, FollowIR-7B, ctxl-6B 52, 114, 29, 216 rows", (52, 114, 29, 216), loss4, "channel_tests_full.json: uji[track=xpair, kontras=both_lawan_wrong].menang_a - menang_b (zerank-2 on the official path, as channel_tests_full.zerank2 states)")
_p4 = max(bw[m]["p_holm_enam_model"] for m in FOUR)
check("4.4", "those four losses at Holm p below 0.01 (the largest is FollowIR-7B's, 0.0093)", "< 0.01", f"max {_p4:.5f}", "channel_tests_full.json: max p_holm_enam_model over the four models", ok=_p4 < 0.01)
check("4.4", "a net 29 rows for FollowIR-7B (60 lost, 31 gained); the other three lose without gaining", ((60, 31, 29), True),
      ((bw["FollowIR-7B"]["n10_a_menang_b_kalah"], bw["FollowIR-7B"]["n01_a_kalah_b_menang"], bw["FollowIR-7B"]["menang_a"] - bw["FollowIR-7B"]["menang_b"]),
       all(bw[m]["n01_a_kalah_b_menang"] == 0 for m in ("Qwen3-Reranker-0.6B", "zerank-2", "ctxl-6B"))),
      "channel_tests_full.json: uji[track=xpair, kontras=both_lawan_wrong].n10_a_menang_b_kalah, n01_a_kalah_b_menang")
check("4.4", "Qwen3-4B resolved slightly more rows with the wrong program, 653 against 641, Holm p 0.25", (653, 641, 0.25), (bw["Qwen3-Reranker-4B"]["menang_b"], bw["Qwen3-Reranker-4B"]["menang_a"], round(bw["Qwen3-Reranker-4B"]["p_holm_enam_model"], 2)), "channel_tests_full.json: uji[track=xpair, kontras=both_lawan_wrong, model=Qwen3-Reranker-4B] menang_b, menang_a, p_holm_enam_model")
for f, lab in (("injection_probe_qwen3rr_Qwen3-Reranker-0.6B.json", "Qwen3-0.6B"), ("injection_probe_qwen3rr_Qwen3-Reranker-4B.json", "Qwen3-4B"), ("injection_probe_zerank2official_zerank-2.json", "zerank-2")):
    pi = J(T + f)["agregat_per_kombinasi"]
    within = all(abs(v["menang"] - 10) <= 2 for k, v in pi.items() if not k.startswith("K5"))
    check("3.5, Table 4 (d)", f"format sweep {lab}: 7 formats, 10 pairs per cell, non-label formats within two rows of baseline", (7, 20, True), (len(pi), pi["K1_default"]["n_komparabel"], within), f"{f}: agregat_per_kombinasi")
kn = z["kontras_naskah_per_jalur"]
_g = [kn[k]["resmi"] for k in ("paired:slot_vs_fmttag", "paired:slot_vs_fmtlabeled")]
check("Table 4 (d)", "the zerank-2 guide formats do no better than a plain prefix", (True, True),
      (all(r["menang_b"] <= r["menang_a"] for r in _g), all(r["p_mcnemar_exact"] >= 0.05 for r in _g)),
      "zerank2_official_replay.json: kontras_naskah_per_jalur paired:slot_vs_fmttag and paired:slot_vs_fmtlabeled .resmi (47 and 46 against 48, p 1.0 and 0.83)")
vs = J(T + "vendor_style_slot.json")["sel"]
si = J(T + "slot_id_paired.json")["sel"]
order_id = ["Qwen3-Reranker-0.6B", "Qwen3-Reranker-4B", "zerank-2", "FollowIR-7B", ARCH_MODELS["ctxl-6B"], ARCH_MODELS["bge-gemma"]]
_cells = [r["menang_b"] for r in _g] + [vs[ARCH_MODELS[m]]["menang_vendor"] for m in ("FollowIR-7B", "ctxl-6B", "bge-gemma")] \
    + [si[m]["menang_id"] for m in order_id]
check("Table 4 (d)", "the guide formats, the vendor styles, and the Indonesian instruction land between 46 and 51 of 96", (46, 51, 11),
      (min(_cells), max(_cells), len(_cells)), "zerank2_official_replay.json (2 cells); vendor_style_slot.json: sel.*.menang_vendor (3); slot_id_paired.json: sel.*.menang_id (6)")
hp_all = [si[m]["id_vs_none"]["p_holm_keluarga_ID"] for m in order_id] + [vs[m]["vendor_vs_none"]["p_holm_keluarga_V"] for m in vs]
check("Table 4 (d)", "these cells sit at Holm p = 1", 1.0, min(hp_all), "slot_id_paired.json: id_vs_none.p_holm_keluarga_ID; vendor_style_slot.json: vendor_vs_none.p_holm_keluarga_V")
bounds = [si[m]["batas_efek_slot_id"]["batas_atas_poin_persen"] for m in order_id] + [vs[m]["batas_efek_slot_vendor"]["batas_atas_poin_persen"] for m in vs]
check("Table 4 (d)", "slot bounds between 7.2 and 12.0 percentage points", (7.2, 12.0), (round(min(bounds), 1), round(max(bounds), 1)), "slot_id_paired.json: batas_efek_slot_id.batas_atas_poin_persen; vendor_style_slot.json: batas_efek_slot_vendor.batas_atas_poin_persen")
check("4.2", "zerank-2 with a vendor-style memory block 47", 47, kn["paired:slot_vs_vmem"]["resmi"]["menang_b"], "zerank2_official_replay.json: kontras_naskah_per_jalur paired:slot_vs_vmem.resmi.menang_b")

check("3.6", "weights 0.5, 0.3, 0.2", [0.5, 0.3, 0.2], cl["static_fingerprint"]["weights"][:3], "config_lock.json: static_fingerprint.weights")
check("3.6 (provenance, not printed in the paper)", "calibration set 155 in-domain and 64 out-of-domain", (155, 64), (cl["bounds"]["n_queries"], cl["bounds"]["n_ood_queries"]), "config_lock.json: bounds.n_queries, n_ood_queries")
cq = J("system/data/calibration_queries.json")
check("3.6 (provenance, not printed in the paper)", "calibration queries shipped (155 + 64)", (155, 64), (len(cq["in_domain"]), len(cq["ood_probe"])), "system/data/calibration_queries.json: in_domain, ood_probe")
rp = J(T + "program_relations.json")
check("3.6", "program weight 1.0 same program, else 0.7, 0.5, 0.3", (1.0, ["0.3", "0.5", "0.7"]), (rp["nilai_prodi_sama"], sorted(rp["tingkat_lintas_prodi"].keys())), "program_relations.json: nilai_prodi_sama, tingkat_lintas_prodi")
pp = J(T + "channel_board_paired.json")["arm_referensi_terbit"]
check("4.5, Fig. 3", "DITS 93 of 96, hard filter 96 of 96", ("93/96", "96/96"), (pp["B_DITS"]["eksak"], pp["filter_metadata"]["eksak"]), "channel_board_paired.json: arm_referensi_terbit.B_DITS.eksak, filter_metadata.eksak")
check("4.5", "broad track: similarity alone 600 and full DITS 779 (similarity plus proximity, also 779, is not printed)", ("600/1621", "779/1621", "779/1621"), (dx["arms"]["S1"]["menang_eksak"], dx["arms"]["S2"]["menang_eksak"], dx["arms"]["B"]["menang_eksak"]), "dits_xpair.json: arms.S1, S2, B.menang_eksak")
dep = dx["arms"]["B"]["per_departemen"]
above = [d for d in dep if dep[d]["pairwise"] > dx["arms"]["S1"]["per_departemen"][d]["pairwise"]]
check("4.5 (provenance, not printed in the paper)", "DITS above the baseline in all eight programs", (8, 8), (len(dep), len(above)), "dits_xpair.json: arms.B.per_departemen vs arms.S1.per_departemen")
sgb = sg["strata"]["v1-cps-gate"]["sel"]["both"]
check("4.5", "three rerankers beat DITS on the broad track: 819, 841, 921", (819, 841, 921), (sgb["zerank-2"], sgb["FollowIR-7B"], sgb["ctxl-6B"]), "gate_strata.json: strata.v1-cps-gate.sel.both")
hf = sg["hard_filter_track_luas"]
check("4.5", "hard filter resolves all 1,621 broad rows", (1621, 1621), (hf["menang"], hf["n"]), "gate_strata.json: hard_filter_track_luas")
rc = J(T + "reachability.json")
check("5", "293 of 2,493 queries (11.75%) have a gold in another program's book", (293, 2493, 0.1175), (rc["gold_lintas_program"]["n"], rc["n_kueri_diperiksa"], rc["gold_lintas_program"]["rate"]), "reachability.json: gold_lintas_program.n, rate; n_kueri_diperiksa")
check("5", "none of them loses all of its gold under the filter", 0, rc["gold_tak_terjangkau_filter_keras"]["n"], "reachability.json: gold_tak_terjangkau_filter_keras.n")

u = J(T + "channel_tests_full.json")
check("3.7", "Holm family of six models (m = 6)", 6, u["uji"][0]["m_enam_model"], "channel_tests_full.json: uji[].m_enam_model")
bk = J(T + "cluster_bootstrap.json")
check("3.7", "10,000 whole-group resamples give narrower intervals on all six models", (10000, 6, 0), (bk["resample"], len(bk["papan"]), len(bk["model_yang_selangnya_melebar"])), "cluster_bootstrap.json: resample, papan, model_yang_selangnya_melebar")

uj = {(x["track"], x["kontras"], x["model"]): x for x in u["uji"]}
slot_paired = [uj[("paired", "none_lawan_slot", m)]["menang_b"] for m in SIX]
slot_broad = [uj[("xpair", "none_lawan_slot", m)]["menang_b"] for m in SIX]
check("Table 3, Fig. 3", "slot cells paired 49, 50, 48, 47, 52, 43", [49, 50, 48, 47, 52, 43], slot_paired, "channel_tests_full.json: uji[track=paired, kontras=none_lawan_slot].menang_b")
check("4.2 (per-model values behind the printed range, provenance)", "slot cells broad 597, 598, 597, 593, 603, 602", [597, 598, 597, 593, 603, 602], slot_broad, "channel_tests_full.json: uji[track=xpair, kontras=none_lawan_slot].menang_b")
check("4.2", "blind cells 48 and 600 for every model", ([48] * 6, [600] * 6), ([uj[("paired", "none_lawan_slot", m)]["menang_a"] for m in SIX], [uj[("xpair", "none_lawan_slot", m)]["menang_a"] for m in SIX]), "channel_tests_full.json: uji[...].menang_a")
check("4.2", "slot resolves 43 to 52 of 96", (43, 52), rng(slot_paired), "derived from the paired slot cells")
check("4.2", "broad slot 593 to 603", (593, 603), rng(slot_broad), "derived from the broad slot cells")
pmc = [uj[("paired", "none_lawan_slot", m)]["p_mentah_mcnemar_eksak"] for m in SIX]
check("4.2", "exact McNemar p from 0.180 to 1.0", (0.18, 1.0), (round(min(pmc), 3), round(max(pmc), 3)), "channel_tests_full.json: uji[paired, none_lawan_slot].p_mentah_mcnemar_eksak")
bs = u["batas_slot"]
t3 = {"paired": {"Qwen3-Reranker-0.6B": (7.81, 6.10), "Qwen3-Reranker-4B": (10.74, 7.22), "zerank-2": (8.45, 8.45), "FollowIR-7B": (6.10, 7.81), "ctxl-6B": (13.71, 6.25), "gemma": (1.88, 11.24)},
      "xpair": {"Qwen3-Reranker-0.6B": (0.56, 0.89), "Qwen3-Reranker-4B": (0.64, 0.87), "zerank-2": (0.58, 0.91), "FollowIR-7B": (0.89, 1.73), "ctxl-6B": (0.82, 0.49), "gemma": (0.58, 0.37)}}
for trk in ("paired", "xpair"):
    got = {m: (round(bs[trk][m]["batas_gain_poin_persen"], 2), round(bs[trk][m]["batas_rugi_poin_persen"], 2)) for m in SIX}
    check("Table 3" if trk == "paired" else "4.2 (per-model values behind the printed limit, provenance)",
          f"largest gain and largest loss columns, {trk} (12 values)", t3[trk], got, f"channel_tests_full.json: batas_slot.{trk}.*.batas_gain_poin_persen, batas_rugi_poin_persen")
chg3 = {"paired": ["+1", "+2", "0", "−1", "+4", "−5"], "xpair": ["−3", "−2", "−3", "−7", "+3", "+2"]}
p3 = {"paired": ["1.00", "0.81", "1.00", "1.00", "0.52", "0.18"], "xpair": ["0.73", "0.87", "0.74", "0.57", "0.70", "0.79"]}


def _chg(n):
    return "0" if n == 0 else (f"+{n}" if n > 0 else f"−{-n}")


for trk in ("paired", "xpair"):
    check("Table 3" if trk == "paired" else "4.2 (per-model values, provenance)", f"change from the blind baseline, {trk} (6 values)", chg3[trk],
          [_chg(uj[(trk, "none_lawan_slot", m)]["menang_b"] - uj[(trk, "none_lawan_slot", m)]["menang_a"]) for m in SIX],
          f"channel_tests_full.json: uji[track={trk}, kontras=none_lawan_slot].menang_b - menang_a")
    check("Table 3, 4.2" if trk == "paired" else "4.2 (per-model values behind the printed result, provenance)",
          f"exact McNemar p of the slot against no profile, {trk} (6 values, two decimals; all above 0.05)", p3[trk],
          [f"{uj[(trk, 'none_lawan_slot', m)]['p_mentah_mcnemar_eksak']:.2f}" for m in SIX],
          f"channel_tests_full.json: uji[track={trk}, kontras=none_lawan_slot].p_mentah_mcnemar_eksak")
check("4.2", "widest paired gain bound 13.7 (ctxl-6B); broad gain bound at most 0.89", (13.71, 0.89), (u["ringkasan_batas"]["paired"]["gain_terbesar_poin_persen"], max(bs["xpair"][m]["batas_gain_poin_persen"] for m in SIX)), "channel_tests_full.json: ringkasan_batas.paired.gain_terbesar_poin_persen; batas_slot.xpair.*.batas_gain_poin_persen")
check("4.2", "zerank-2: 48 with the profile in the slot, 597 on the broad track", (48, 597), (z["sel"]["paired_profile"]["menang_resmi"], z["sel"]["xpair_profile"]["menang_resmi"]), "zerank2_official_replay.json: sel.paired_profile, xpair_profile.menang_resmi")
won_p = [uj[("paired", "none_lawan_slot", m)]["n01_a_kalah_b_menang"] for m in SIX]
lost_p = [uj[("paired", "none_lawan_slot", m)]["n10_a_menang_b_kalah"] for m in SIX]
won_b = [uj[("xpair", "none_lawan_slot", m)]["n01_a_kalah_b_menang"] for m in SIX]
lost_b = [uj[("xpair", "none_lawan_slot", m)]["n10_a_menang_b_kalah"] for m in SIX]
check("Fig. 4", "paired rows the slot won / lost: 6/5, 10/8, 8/8, 5/6, 13/9, 2/7", ([6, 10, 8, 5, 13, 2], [5, 8, 8, 6, 9, 7]), (won_p, lost_p), "channel_tests_full.json: uji[paired, none_lawan_slot].n01_a_kalah_b_menang, n10_a_menang_b_kalah")
check("Fig. 4", "broad rows the slot won / lost: 15/18, 17/19, 16/19, 51/58, 15/12, 8/6", ([15, 17, 16, 51, 15, 8], [18, 19, 19, 58, 12, 6]), (won_b, lost_b), "channel_tests_full.json: uji[xpair, none_lawan_slot].n01, n10")
check("4.1", "rows changed: 9 to 22 paired, 14 to 109 broad", ((9, 22), (14, 109)), (rng([a + b for a, b in zip(won_p, lost_p)]), rng([a + b for a, b in zip(won_b, lost_b)])), "derived from the discordant counts")
check("4.1", "largest net change a loss of 5 (paired) and 7 (broad)", (-5, -7), (min(a - b for a, b in zip(won_p, lost_p)), min(a - b for a, b in zip(won_b, lost_b))), "derived from the discordant counts")
ph = [uj[(t, "none_lawan_slot", m)]["p_holm_enam_model"] for t in ("paired", "xpair") for m in SIX]
check("4.1", "no net change is significant (Holm p >= 0.05 in all 12 slot contrasts)", True, min(ph) >= 0.05, "channel_tests_full.json: uji[none_lawan_slot].p_holm_enam_model")
hm = J(T + "asker_hit_mrr.json")
fl = hm["laju_flip_top1_sel_slot"]
check("Abstract, 4.1, 7", "top passage differs in 15 to 29% of the 52 pairs", (15, 29), (math.floor(100 * min(v["laju"] for v in fl["paired"].values())), round(100 * max(v["laju"] for v in fl["paired"].values()))), "asker_hit_mrr.json: laju_flip_top1_sel_slot.paired.*.laju (grup 52)")
check("4.1 (provenance, not printed in the paper)", "and in 6.2 to 15.5% of the 600 groups", (6.2, 15.5), (round(100 * min(v["laju"] for v in fl["xpair"].values()), 1), round(100 * max(v["laju"] for v in fl["xpair"].values()), 1)), "asker_hit_mrr.json: laju_flip_top1_sel_slot.xpair.*.laju (grup 600)")
fn = hm["laju_flip_top1_sel_none"]
check("4.1", "without the profile the share is zero on both tracks", 0, max(v["laju"] for t in fn for v in fn[t].values()), "asker_hit_mrr.json: laju_flip_top1_sel_none")
fr = J(T + "flip_rate.json")["per_arm"]
check("4.1 (provenance, not printed in the paper)", "Qwen3 models: every order changed while the mean Kendall tau was 0.91 and 0.93", (1.0, 0.91, 0.93), (min(fr["C3_0.6b_profil"]["flip_urutan_penuh"], fr["C4_4b_profil"]["flip_urutan_penuh"]), round(fr["C3_0.6b_profil"]["kendall_tau_mean"], 2), round(fr["C4_4b_profil"]["kendall_tau_mean"], 2)), "flip_rate.json: per_arm.C3_0.6b_profil, C4_4b_profil (flip_urutan_penuh, kendall_tau_mean)")
hs = hm["hasil"]["paired"]["slot"]
check("4.1", "with the slot the asker's gold comes first in 17 to 30 rows, the rival's in 15 to 29", ((17, 30), (15, 29)), (rng([v["hit1_baris"] for v in hs.values()]), rng([v["rival1_baris"] for v in hs.values()])), "asker_hit_mrr.json: hasil.paired.slot.*.hit1_baris, rival1_baris")
dsp = hm["selisih_slot_lawan_none"]
check("4.1 (provenance, not printed in the paper)", "Hit@1 shift at most 7.3 points paired and 1.8 broad", (7.3, 1.8), (round(100 * max(abs(v["hit1_slot_dikurangi_none"]) for v in dsp["paired"].values()), 1), round(100 * max(abs(v["hit1_slot_dikurangi_none"]) for v in dsp["xpair"].values()), 1)), "asker_hit_mrr.json: selisih_slot_lawan_none.*.hit1_slot_dikurangi_none")

ar = J(T + "channel_board_paired_architectures.json")
qb = ar["papan_beku_keturunan_qwen3"]
both_p = [int(qb["Qwen3-Reranker-0.6B"]["slot_plus_teks"].split("/")[0]), int(qb["Qwen3-Reranker-4B"]["slot_plus_teks"].split("/")[0]), z["sel"]["paired_profile_docmeta"]["menang_resmi"],
          ar["papan"]["FollowIR-7B"]["both"]["menang"], ar["papan"][ARCH_MODELS["ctxl-6B"]]["both"]["menang"], ar["papan"][ARCH_MODELS["bge-gemma"]]["both"]["menang"]]
check("Table 3, Fig. 3", "slot and label paired 88, 83, 96, 91, 96, 49", [88, 83, 96, 91, 96, 49], both_p, "channel_board_paired_architectures.json: papan_beku_keturunan_qwen3.*.slot_plus_teks, papan.*.both.menang; zerank2_official_replay.json: sel.paired_profile_docmeta.menang_resmi")
both_b = [sgb[m] for m in SIX]
check("4.4 (per-model values behind the printed range, provenance)", "slot and label broad 697, 641, 819, 841, 921, 603", [697, 641, 819, 841, 921, 603], both_b, "gate_strata.json: strata.v1-cps-gate.sel.both")
five = [0, 1, 2, 3, 4]
check("4.4", "five models reach 83 to 96, 33 to 48 rows above their slot cells", ((83, 96), (33, 48)), (rng([both_p[i] for i in five]), rng([both_p[i] - slot_paired[i] for i in five])), "derived from the both and slot cells")
check("4.4", "on 1,621 rows the same five reach 641 to 921", (641, 921), rng([both_b[i] for i in five]), "derived from gate_strata.json")
mp = J(T + "mcnemar_paired.json")["kontras_mcnemar"]
c3 = mp["C3_profil vs C3_docmeta_profil"]
check("4.4", "Qwen3-0.6B: 39 rows toward a win against 0, Holm p < 0.001", (39, 0, True), (c3["diskordan_a_kalah_b_menang"], c3["diskordan_a_menang_b_kalah"], c3["p_holm_keluarga_posthoc"] < 0.001), "mcnemar_paired.json: kontras_mcnemar['C3_profil vs C3_docmeta_profil']")
ko = ar["mcnemar_kontras_lain_p_exact"]
fi, cx = ko["FollowIR-7B slot vs both"], ko[ARCH_MODELS["ctxl-6B"] + " slot vs both"]
check("4.4", "FollowIR-7B 45 against 1 and ctxl-6B 44 against 0, raw p < 1e-11", (45, 1, 44, 0, True), (fi["diskordan_a_kalah_b_menang"], fi["diskordan_a_menang_b_kalah"], cx["diskordan_a_kalah_b_menang"], cx["diskordan_a_menang_b_kalah"], max(fi["p_mcnemar_exact_penuh"], cx["p_mcnemar_exact_penuh"]) < 1e-11), "channel_board_paired_architectures.json: mcnemar_kontras_lain_p_exact")
xa = J(T + "channel_board_xpair_architectures.json")["papan"][ARCH_MODELS["ctxl-6B"]]["slot_vs_both"]
check("4.4 (provenance, not printed in the paper)", "ctxl-6B gains 318 broad rows while losing none", (318, 0), (xa["a_kalah_b_menang"], xa["a_menang_b_kalah"]), "channel_board_xpair_architectures.json: papan.ctxl.slot_vs_both")
hb = hm["hasil"]["paired"]["both"]
five_names = ["Qwen3-Reranker-0.6B", "Qwen3-Reranker-4B", "zerank-2", "FollowIR-7B", "ctxl-6B"]
check("4.4", "with both channels the asker's gold comes first in 25 to 75% of rows, the rival's in at most 4", ((25, 75), 4), ((round(100 * min(hb[m]["hit1"] for m in five_names)), round(100 * max(hb[m]["hit1"] for m in five_names))), max(hb[m]["rival1_baris"] for m in five_names)), "asker_hit_mrr.json: hasil.paired.both.*.hit1, rival1_baris")
hn = hm["hasil"]["paired"]["none"]
check("4.4 (provenance, not printed in the paper)", "MRR@10 rises from 0.29 to 0.53 (neither) to 0.46 to 0.85 (both)", ((0.29, 0.53), (0.46, 0.85)), ((round(min(hn[m]["mrr10"] for m in five_names), 2), round(max(hn[m]["mrr10"] for m in five_names), 2)), (round(min(hb[m]["mrr10"] for m in five_names), 2), round(max(hb[m]["mrr10"] for m in five_names), 2))), "asker_hit_mrr.json: hasil.paired.none/both.*.mrr10")
lf = J(T + "single_field_label.json")["model"]
sem = [lf[m]["semester_only"]["menang"] for m in QW + ["zerank-2"]] + [ar["papan"]["FollowIR-7B"]["both_semester"]["menang"], ar["papan"][ARCH_MODELS["ctxl-6B"]]["both_semester"]["menang"]]
prg = [lf[m]["program_only"]["menang"] for m in QW + ["zerank-2"]] + [ar["papan"]["FollowIR-7B"]["both_program"]["menang"], ar["papan"][ARCH_MODELS["ctxl-6B"]]["both_program"]["menang"]]
check("4.4, Table 5", "semester-only label 78 to 96, program-only 51 to 89", ((78, 96), (51, 89)), (rng(sem), rng(prg)), "single_field_label.json: model.*.semester_only/program_only.menang; channel_board_paired_architectures.json: papan.*.both_semester/both_program.menang")
check("4.4", "zerank-2 one-field cells agree on the model-card path", (z["sel"]["paired_profile_docmeta_fieldsemester"]["menang_resmi"], z["sel"]["paired_profile_docmeta_fieldprogram"]["menang_resmi"]), (lf["zerank-2"]["semester_only"]["menang"], lf["zerank-2"]["program_only"]["menang"]), "zerank2_official_replay.json: sel.paired_profile_docmeta_fieldsemester/fieldprogram.menang_resmi")
pj = J(T + "channel_board_paired.json")["papan"]
wrong5 = [pj["Qwen3-Reranker-0.6B"]["label_salah"]["menang"], pj["Qwen3-Reranker-4B"]["label_salah"]["menang"],
          z["sel"]["paired_profile_docmetasalah"]["menang_resmi"],
          int(ar["ringkas"]["FollowIR-7B"]["wrong"].split("/")[0]), int(ar["ringkas"][ARCH_MODELS["ctxl-6B"]]["wrong"].split("/")[0])]
check("4.4, Table 5", "wrong program in every label, semester kept: the five models reach 72 to 94 of 96", (72, 94), rng(wrong5), "channel_board_paired.json: papan.Qwen3-*.label_salah.menang; zerank2_official_replay.json: sel.paired_profile_docmetasalah.menang_resmi; channel_board_paired_architectures.json: ringkas.FollowIR-7B/ctxl.wrong")
check("4.4", "bge-gemma reaches 49 of 96 with both channels against 43 with the slot alone (p = 0.263, slot vs both), and 603 of 1,621", (49, 0.263, 603), (both_p[5], round(ko["bge-reranker-v2-gemma slot vs both"]["p_mcnemar_exact"], 3), both_b[5]), "channel_board_paired_architectures.json: papan.gemma.both, mcnemar_kontras_lain_p_exact; gate_strata.json")
kl = J(T + "profile_collinearity.json")["track"]["paired"]
check("3.2, 6", "on the paired track program and semester change together", True, kl["prodi_dan_semester_kolinear_sempurna"], "profile_collinearity.json: track.paired.prodi_dan_semester_kolinear_sempurna")
prov = (ROOT / "study/channel/provenance/README.md").read_text(encoding="utf-8", errors="ignore")
check("3.2 (the workflow itself, named in the provenance note)", "paired-track questions written by agents through a Claude Code workflow", True, "Claude Code workflow" in " ".join(prov.split()) and (ROOT / "study/channel/provenance/paired_set_generation_workflow.js").exists(), "study/channel/provenance/README.md, paired_set_generation_workflow.js")

ep = J("study/retrieval/results/raw/eval_paired.json")
s2 = find_key(ep, "S2")
bb = find_key(ep, "B")
s2p = find_key(s2, "pairwise") if isinstance(s2, dict) else None
bp = find_key(bb, "pairwise") if isinstance(bb, dict) else None
check("4.5 (provenance, not printed in the paper)", "paired track: similarity plus proximity alone also gives 93, and the prerequisite term adds nothing (S2 = B = 93/96)", (93, 93),
      (round(s2p * 96) if s2p is not None else "S2 not found", round(bp * 96) if bp is not None else "B not found"),
      "study/retrieval/results/raw/eval_paired.json: S2.pairwise x 96, B.pairwise x 96", ok=(s2p is not None and bp is not None and round(s2p * 96) == 93 and round(bp * 96) == 93))

js = (ROOT / "study/channel/provenance/paired_set_generation_workflow.js").read_text(encoding="utf-8")
batches = re.search(r"const BATCHES = \[([0-9, ]+)\]", js)
n_batch = len([x for x in batches.group(1).split(",") if x.strip()]) if batches else None
per_batch = re.search(r"bi \* 25", js)
check("3.2", "agents took the 150 most similar of the 400 pairs (6 batches of 25 rows)", 150, (n_batch * 25) if (n_batch and per_batch) else "BATCHES not found",
      "study/channel/provenance/paired_set_generation_workflow.js: const BATCHES (6 batches) x 25 rows per batch")
prov = (ROOT / "study/channel/provenance/README.md").read_text(encoding="utf-8")
m115 = re.search(r"(\d+) candidates and (\d+) accepted", prov)
check("3.2", "115 candidates in all, 52 accepted", (115, 52), (int(m115.group(1)), int(m115.group(2))) if m115 else "not found",
      "study/channel/provenance/README.md: 115 candidates and 52 accepted pairs")
cfg = (ROOT / "system/app/config.py").read_text(encoding="utf-8")
cps_src = (ROOT / "system/app/retrieval/cps.py").read_text(encoding="utf-8")
check("3.6", "a candidate below the calibrated floor (sim_norm 0) scores zero", True,
      ("cps_zero_sim_not_candidate: bool = True" in cfg) and ("r.cps_score = 0.0" in cps_src),
      "system/app/config.py: cps_zero_sim_not_candidate = True; system/app/retrieval/cps.py: rerank() sets cps_score 0.0 when sim_norm is 0")

_pp = J(T + "channel_board_paired.json")["papan"]
_pa = J(T + "channel_board_paired_architectures.json")["papan"]
_zs = J(T + "zerank2_official_replay.json")["sel"]
_lo = (_pp["Qwen3-Reranker-0.6B"]["teks_saja"]["menang"], _pp["Qwen3-Reranker-4B"]["teks_saja"]["menang"],
       _zs["paired_default_docmeta"]["menang_resmi"], _pa["FollowIR-7B"]["text"]["menang"],
       _pa["ctxl-rerank-v2-instruct-multilingual-6b"]["text"]["menang"], _pa["bge-reranker-v2-gemma"]["text"]["menang"])
check("4.2", "the label-only cell sits at the blind baseline, 48 of 96, on all six models", (48,) * 6, _lo,
      "channel_board_paired.json: papan.*.teks_saja.menang (Qwen3); zerank2_official_replay.json: sel.paired_default_docmeta.menang_resmi; channel_board_paired_architectures.json: papan.*.text.menang")
_sg = J(T + "semester_gold_paired.json")["semester"]
check("4.4", "the semester was right in all 96 paired rows", (96, 96), (_sg["gold_cocok_semester_mahasiswa"], _sg["penyebut"]),
      "semester_gold_paired.json: semester.gold_cocok_semester_mahasiswa, semester.penyebut")

bc = J(R + "benchmark_composition.json")
_g = {}
for _sp in ("traindev", "test_holdout"):
    for _k, _v in bc["per_split"][_sp]["queries_by_gold_count"].items():
        _g[_k] = _g.get(_k, 0) + _v
check("3.1 (the 526 are provenance, not printed in the paper)", "each query has one or two gold passages, and the 526 with two are the prerequisite class", (["1", "2"], 526, 526),
      (sorted(_g), _g.get("2"), ab["kelas"]["prerequisite_chain_depth"]), "benchmark_composition.json: per_split.*.queries_by_gold_count; benchmark_provenance.json: kelas.prerequisite_chain_depth")
_cs = (ROOT / "system/app/benchmark/rads_synthesis/chunk_selector.py").read_text(encoding="utf-8")
check("3.1, 6", "queries written from one to three passages (class 1 from two or three, class 2 from two, classes 3 and 4 from one)", True,
      "self.rng.choice([2, 2, 3])" in _cs and "chunks=[chunk_a, chunk_b]" in _cs and _cs.count("chunks=[chunk]") >= 2,
      "system/app/benchmark/rads_synthesis/chunk_selector.py: k for class 1, chunks= for classes 2 to 4")
_tp = (ROOT / "system/app/benchmark/rads_synthesis/templates.py").read_text(encoding="utf-8")
check("3.1", "every passage went to the generator with its course, program, and semester", True,
      "[course: {c.course_code} {c.course_name} | program: {c.department_name} | semester: {c.semester}]" in _tp,
      "system/app/benchmark/rads_synthesis/templates.py: _chunk_block")
_bp = (ROOT / "system/scripts/build_paired_set.py").read_text(encoding="utf-8")
check("3.2 (stated in the script that built the track, when the benchmark had 1,412 queries)", "no benchmark question is asked by two different students", True,
      re.search(r"zero queries in which the same\s+question is asked by different students", _bp) is not None,
      "system/scripts/build_paired_set.py: module docstring")
_pm = J("study/channel/results/raw/pools_paired_expanded.manifest.json")
check("3.4, Fig. 2", "2,493 train and dev queries read (test untouched), 2,200 with the gold in the asker's program, 274 without a partner, 179 pairs and 421 triples, 1,621 rows",
      ("traindev 2.493", True, 2200, 274, 179, 421, 1621),
      (_pm["populasi_sumber"].split(",")[0], "test_holdout TIDAK disentuh" in _pm["populasi_sumber"], _pm["n_kueri_kandidat"],
       _pm["rincian"]["tanpa_padanan"], _pm["rincian"]["pasangan"], _pm["rincian"]["triple"], _pm["n_baris"]),
      "study/channel/results/raw/pools_paired_expanded.manifest.json: populasi_sumber, n_kueri_kandidat, rincian, n_baris")
_ub = J(T + "gate_strata.json")["urutan_berkas_kueri"]
check("3.4", "the query file begins with the 1,412 first-gate queries, and the miner stopped before any second-gate query", ([0, 1411], True),
      (_ub["posisi_baris_per_protokol_min_maks"]["v1-cps-gate"], _ub["posisi_kueri_asal_terakhir_yang_dipakai"] < _ub["posisi_baris_per_protokol_min_maks"]["v2-multiarm"][0]),
      "gate_strata.json: urutan_berkas_kueri")
_ep = (ROOT / "study/channel/scripts/1_tracks/expand_paired_track.py").read_text(encoding="utf-8")
check("3.4", "one shared pool per group, built around the query's gold", True, "v = X[pos[cid]]" in _ep and '"cands": cands' in _ep,
      "study/channel/scripts/1_tracks/expand_paired_track.py: pool from the gold of the source query, the same cands for every row")
_ip = (ROOT / "study/channel/scripts/2_scoring/injection_probe.py").read_text(encoding="utf-8")
_kb = _ip[_ip.index("K = ["):_ip.index("]\n", _ip.index("K = ["))]
_ks = re.findall(r'\("(K\d_\w+)",\s*(\w+),\s*(\w+),\s*(True|False)\)', _kb)
_kn = [k for k, inst, q, _ in _ks if inst == "None" and q == "plain_q"]
check("3.5", "the format sweep runs seven conditions, five of which deliver the profile", (7, 5), (len(_ks) + 1, len(_ks) + 1 - len(_kn)),
      "study/channel/scripts/2_scoring/injection_probe.py: K1 to K6 plus K7 (K1 default and K4 label only carry no profile)", ok=len(_ks) == 6 and len(_kn) == 2)
check("3.7", "every channel test, the Qwen3 cells included, has a six-model Holm family, and the worked-example control a family of its three models",
      (True, True), (all(t["m_enam_model"] == 6 for t in u["uji"]), "family of three models" in ct["uji_primer"]),
      "channel_tests_full.json: uji[].m_enam_model; control_textual.json: uji_primer")
check("Table 4 (c)", "the textual control compares the passage-type instruction with the default instruction", True, "condition minus default" in ct["uji_primer"],
      "control_textual.json: uji_primer")
_ti = J(T + "marker_ladder.json")["papan"]
_tp = J(T + "surface_ladder.json")["papan"]
_five = ["0.6B", "4B", "zerank-2", "FollowIR-7B", "ctxl-6B"]
check("4.4, Table 5", "English program name in instruction and label: five models reach 57 to 87 of 96", (57, 87),
      rng([_tp[m]["p1_nama_en"]["menang"] for m in _five]), "surface_ladder.json: papan.*.p1_nama_en.menang")
check("4.4", "close to the code: the English name differs significantly from the code label on none of the five (Holm)", 0,
      sum(bool(_tp[m]["p1_nama_en"]["vs_a_kode_instruksi_kode"]["signifikan_0.05"]) for m in _five),
      "surface_ladder.json: papan.*.p1_nama_en.vs_a_kode_instruksi_kode.signifikan_0.05 (Holm, family of six)")
_pp = J(T + "surface_ladder_p1_p2.json")
_ppm = _pp["per_model"]
_pmap = {"0.6B": "0.6B", "4B": "4B", "zerank-2": "zerank-2", "FollowIR-7B": "FollowIR-7B", "ctxl-6B": "ctxl-6B", "gemma": "gemma"}
_praw = [binom_two_sided(min(_ppm[m]["p1_saja"], _ppm[m]["p2_saja"]), _ppm[m]["p1_saja"] + _ppm[m]["p2_saja"]) for m in _pmap]
_pholm = dict(zip(_pmap, holm(_praw)))
_lower = sum(_ppm[m]["p2_menang"] < _ppm[m]["p1_menang"] for m in _five)
_sig = [m for m in _five if _pholm[m] < 0.05]
check("4.4", "changing only the label's language (English name to Indonesian) lowers the count on all five models, significantly on three (Holm p at most 0.004)",
      (5, 3, 0.004), (_lower, len(_sig), round(max(_pholm[m] for m in _sig), 3)),
      "surface_ladder_p1_p2.json: per_model.*.{p1_menang,p2_menang,p1_saja,p2_saja}; exact McNemar and Holm (six models) recomputed here",
      ok=_lower == 5 and len(_sig) == 3 and max(_pholm[m] for m in _sig) <= 0.004)
check("4.4", "the p1 and p2 totals of the deciding-contrast table equal the frozen surface ladder", True,
      all(_ppm[m]["p1_menang"] == _tp[m]["p1_nama_en"]["menang"] and _ppm[m]["p2_menang"] == _tp[m]["p2_nama_id"]["menang"]
          and _ppm[m]["sha256_p1"] == _tp[m]["p1_nama_en"]["sha256"] and _ppm[m]["sha256_p2"] == _tp[m]["p2_nama_id"]["sha256"] for m in _pmap),
      "surface_ladder_p1_p2.json against surface_ladder.json: papan.*.{p1_nama_en,p2_nama_id}.{menang,sha256}")
check("4.4", "the other ladder cannot be interpreted by its own rule: its opaque-code control misses 48 by more than two rows on three of the six models",
      (3, 6), (sum(abs(v["opak"]["menang"] - 48) > 2 for v in _ti.values()), len(_ti)),
      "marker_ladder.json: papan.*.opak.menang")
_ab = J(T + "channel_board_paired_architectures.json")
_hf = {v["m"] for k, v in _ab.items() if k.startswith("mcnemar_keluarga_") and isinstance(v, dict) and isinstance(v.get("m"), int)}
check("3.7", "the protocols set Holm families of three to nine tests", (3, 9), (min(_hf), max(_hf)),
      "channel_board_paired_architectures.json: mcnemar_keluarga_*.m")
_pc = {e["dokumen"]: e for e in J(T + "protocol_commits.json")["protokol"]}
_gov = [e for e in _pc.values() if e.get("sel_pertama_utc")]
check("3.5, 4.4", "both label ladders were committed before their first cell, and the first channel protocol is the cross-architecture one",
      (True, "PRAREGISTRASI_LINTAS_ARSITEKTUR.md"),
      (all(_pc[d]["protokol_mendahului_sel"] for d in ("PRAREGISTRASI_TANGGA_JARAK_INFERENSI.md",
                                                      "PRAREGISTRASI_TANGGA_ADENDUM5_PERMUKAAN.md")),
       min(_gov, key=lambda e: e["commit_utc"])["dokumen"]),
      "protocol_commits.json: protokol[].protokol_mendahului_sel, commit_utc")

_env = J(T + "deployment_envelope.json")
_peak = {m["model"]: round(m["vram_puncak_mib"]) for m in _env["per_model"]}
check("4.5 (provenance, not printed in the paper)", "Qwen3-0.6B, zerank-2, and Qwen3-4B peaked at 5,155 to 9,716 MiB of GPU memory", (5155, 9716),
      (min(_peak.values()), max(_peak.values())), "deployment_envelope.json: per_model.*.vram_puncak_mib (rounded)")
check("4.5 (provenance, not printed in the paper)", "the memory peaks come from the paired run with both channels", "paired_profile_docmeta",
      _env["kondisi"] if all(m["kondisi"] == _env["kondisi"] for m in _env["per_model"]) else "mixed",
      "deployment_envelope.json: kondisi and per_model.*.kondisi")
check("4.5 (provenance, not printed in the paper)", "the three measured models are Qwen3-0.6B, zerank-2, and Qwen3-4B",
      ["Qwen3-Reranker-0.6B", "Qwen3-Reranker-4B", "zerank-2"], sorted(_peak), "deployment_envelope.json: per_model.*.model")
check("4.5 (provenance, not printed in the paper)", "SentIO's answer generator leaves at most 2,553 MiB free on the 16 GB card", (2553, 16303 - 13750),
      (_env["headroom"]["sisa_mib"], _env["headroom"]["kartu_total_mib"] - _env["headroom"]["generator_reservation_mib"]),
      "deployment_envelope.json: headroom.sisa_mib = kartu_total_mib - generator_reservation_mib")


_cal = _pm["kalibrasi"]
check("3.4", "the partner window is the similarity range of the 400 paired-track topic pairs, 0.66 to 0.89", (400, 0.66, 0.89),
      (int(re.search(r"dari (\d+) kandidat pasangan ASLI", _cal).group(1)),
       round(float(re.search(r"sim (\d,\d+) sampai", _cal).group(1).replace(",", ".")), 2),
       round(float(re.search(r"sampai (\d,\d+) \(median", _cal).group(1).replace(",", ".")), 2)),
      "pools_paired_expanded.manifest.json: kalibrasi (sim 0.6594 to 0.8858 over the 400 original pairs)")
check("3.4, Fig. 2", "the miner read 874 queries: 600 groups plus 274 without a partner", 874,
      _pm["n_pasangan_dibangun"] + _pm["rincian"]["tanpa_padanan"],
      "pools_paired_expanded.manifest.json: n_pasangan_dibangun + rincian.tanpa_padanan; expand_paired_track.py counts one query per group or per miss")
_man = []
for _mp in sorted((ROOT / "study/channel/results/raw").glob("scores_*.manifest.json")):
    _m = json.loads(_mp.read_text(encoding="utf-8"))
    if isinstance(_m.get("peta_label_sumber"), dict):
        _man.append(_m)
_idm = [m for m in _man if m["peta_label_sumber"].get("peta_label") == "id"]
_oct = all(re.search(r"_202610\d\dT", m["run_id"]) for m in _idm)
check("Table 3, 4.4 (provenance, not printed)", "the label counts use labels read from each passage's own record", (97, True),
      (len(_idm), _oct), "study/channel/results/raw/scores_*.manifest.json: peta_label_sumber.peta_label = id and run_id dated 2026-10")
FIGURE_LABELS = {
 "fig2": [
  "Original paired track",
  "two statistics books, one course pair",
  "First broad sample",
  "eight degree programs",
  "400 topic pairs",
  "cross-book, matched by embedding similarity",
  "150 most similar pairs",
  "115 candidate questions written",
  "52 pairs kept",
  "judged answerable by agents",
  "48 pairs, both golds in the pool",
  "96 rows, baseline 48",
  "2,493 queries",
  "training and development splits",
  "2,200 queries",
  "gold in the asker's own program",
  "600 groups mined",
  "874 queries read, 179 pairs and 421 triples",
  "1,621 rows",
  "blind baseline 600, one per group"
 ],
 "fig1": [
  "counted out of 96,\rblind baseline 48",
  "3. Metadata read directly",
  "1. One question, two students (pair 1 of the paired track)",
  "Asked by both students (translated from Indonesian): “What does simple random sampling mean, and what must a sample satisfy to count as simple random?”",
  "CSCI, semester 1. Gold in the Lane book",
  "“Such sampling requires every member of the population to have an equal chance of being selected into the sample.”",
  "STAT, semester 2. Gold in the OpenStax book",
  "“Any group of n individuals is equally likely to be chosen as any other group of n individuals if the simple random sampling technique is used.”",
  "2. Two channels, crossed 2 × 2",
  "Candidate pool",
  "64 to 80 passages, same for both students",
  "Instruction slot",
  "the student profile",
  "(channel under test)",
  "Label in each passage",
  "program and semester",
  "(sensitivity check)",
  "six models,",
  "three families",
  "Resolved row",
  "asker’s gold passage ranks above the other student’s gold",
  "Hard filter (program) or DITS (program and semester), both read from metadata"
 ]
}


def _labels(name):
    return FIGURE_LABELS[name]


def _nums(labels):
    found = []
    for t in labels:
        t = re.sub(r"Section \d+(?:\.\d+)?|Rule \d|^\d\. |pair \d of|semester \d|\d × \d", "", t)
        found += [int(x.replace(",", "")) for x in re.findall(r"\d+(?:,\d{3})*", t)]
    return sorted(set(found))


_f2 = _labels("fig2")
_want2 = {tb["pasangan_topik_kandidat"]: 400, n_batch * 25: 150, int(m115.group(1)): 115, tb["pasangan"]: 52,
          tb["pasangan_kedua_gold_di_pool"]: 48, tb["baris_komparabel"]: 96, rc["n_kueri_diperiksa"]: 2493,
          _pm["n_kueri_kandidat"]: 2200, tl["grup"]: 600, _pm["n_pasangan_dibangun"] + _pm["rincian"]["tanpa_padanan"]: 874,
          tl["ukuran_grup"]["2"]: 179, tl["ukuran_grup"]["3"]: 421, tl["baris"]: 1621}
check("Fig. 2", "every number in the figure matches the data (400, 150, 115, 52, 48, 96, 2,493, 2,200, 600, 874, 179, 421, 1,621)",
      sorted(_want2.values()), _nums(_f2), "FIGURE_LABELS (Fig. 2) against pool_statistics.json, pools manifest, reachability.json, provenance",
      ok=_nums(_f2) == sorted(_want2) == sorted(_want2.values()))
_f1 = _labels("fig1")
_want1 = sorted({tb["kandidat_unik_per_baris"]["min"], tb["kandidat_unik_per_baris"]["max"], tb["baris_komparabel"], tb["baris_komparabel"] // 2})
check("Fig. 1", "every number in the figure matches the data (64 to 80 passages, 96 rows, blind baseline 48)", [48, 64, 80, 96], _nums(_f1),
      "FIGURE_LABELS (Fig. 1) against pool_statistics.json", ok=_nums(_f1) == _want1 == [48, 64, 80, 96])

# Release 1.4.0: the paper's numbers (extended paired track, rebuilt broad track, follow-up analyses)
from decimal import Decimal as _D, ROUND_HALF_UP as _HU


def _r(x, n):
    return str(_D(str(x)).quantize(_D(1) / (_D(10) ** n), rounding=_HU))


def _side(s, k):
    lims = (s["row_interval"], s["group_cr1_interval"], s["group_interval"])
    return 100 * (max(x[1] for x in lims) if k == "gain" else max(-x[0] for x in lims))


_FIVE = ["Qwen3-0.6B", "Qwen3-4B", "zerank-2", "FollowIR-7B", "ctxl-6B"]
_SIXR = _FIVE + ["bge-gemma"]

_ap = J(T + "r1_paired_provenance.json")
check("3.2", "the 150 most similar topic pairs went to the writers", 150, _ap["flow"]["pairs_sent_to_writers"],
      "r1_paired_provenance.json: flow.pairs_sent_to_writers")
check("3.2", "115 questions written, 35 pairs declined", (115, 35), (_ap["flow"]["questions_written"], _ap["flow"]["refused_by_writers"]),
      "r1_paired_provenance.json: flow.questions_written, refused_by_writers")
check("3.2", "52 passed the screen, the other 63 failed with a logged reason", (52, 63), (_ap["flow"]["passed"], _ap["flow"]["failed"]),
      "r1_paired_provenance.json: flow.passed, failed")
check("3.2", "the 52 questions are, word for word, the questions used in the study", True,
      _ap["checks"]["passing_questions_equal_release_text"], "r1_paired_provenance.json: checks.passing_questions_equal_release_text")

_og = J(T + "r1_slot_orthogonal.json")["cells"]
check("4.2", "six more slot conditions", 6, len(_og), "r1_slot_orthogonal.json: cells")
check("4.2", "the six rerankers resolved 44 to 53 of 96 across these conditions", (44, 53),
      rng([x["resolved_B"] for c in _og.values() for x in c["vs_blind"].values()]), "r1_slot_orthogonal.json: cells.*.vs_blind.*.resolved_B")
_ph = [x["p_holm_mcnemar_exact"] for c in _og.values() for k in ("vs_blind", "vs_full_profile_slot") for x in c[k].values()]
check("4.2", "none differed from the blind baseline or the full profile (Holm p = 1.0 in every case)", 1.0, min(_ph),
      "r1_slot_orthogonal.json: cells.*.{vs_blind, vs_full_profile_slot}.*.p_holm_mcnemar_exact")
check("4.2", "largest gain 13.0 percentage points, or 16.1 family-wise", ("13.0", "16.1"),
      (_r(max(c["summary_vs_blind"]["largest_gain_pp_max_over_models"] for c in _og.values()), 1),
       _r(max(c["summary_vs_blind"]["family_wise_largest_gain_pp_max_over_models"] for c in _og.values()), 1)),
      "r1_slot_orthogonal.json: cells.*.summary_vs_blind.{largest_gain_pp_max_over_models, family_wise_largest_gain_pp_max_over_models}")
check("4.2", "bge-gemma's one-field rows sit at its appended placement and span 44 to 48; the other five span 45 to 53", ((44, 48), (45, 53)),
      (rng([c["vs_blind"]["bge-gemma"]["resolved_B"] for c in _og.values()]),
       rng([c["vs_blind"][m]["resolved_B"] for c in _og.values() for m in _FIVE])),
      "r1_slot_orthogonal.json: cells.*.vs_blind.*.resolved_B")

_gv = J(T + "r1_gemma_vendor.json")["tracks"]["paired"]["documented_placement"]
check("4.4", "bge-gemma at its model-card placement resolves 46 of 96 with the slot and 49 with both channels", (46, 49),
      (_gv["slot_vendor_vs_none"]["resolved_B"], _gv["both_vendor_vs_slot_vendor"]["resolved_B"]),
      "r1_gemma_vendor.json: tracks.paired.documented_placement.{slot_vendor_vs_none, both_vendor_vs_slot_vendor}.resolved_B")
check("4.4", "neither change significant (p = 0.73 and 0.45), family-wise limits 5.2 and 9.7 percentage points",
      ("0.73", "0.45", "5.2", "9.7"),
      (_r(_gv["slot_vendor_vs_none"]["p_mcnemar_exact"], 2), _r(_gv["both_vendor_vs_slot_vendor"]["p_mcnemar_exact"], 2),
       _r(_gv["slot_vendor_vs_none"]["family_wise"]["largest_gain_pp"], 1), _r(_gv["both_vendor_vs_slot_vendor"]["family_wise"]["largest_gain_pp"], 1)),
      "r1_gemma_vendor.json: ...{p_mcnemar_exact, family_wise.largest_gain_pp}")
_pm = tuple(_r(J(T + f"followir_{c}_gemma_bge-reranker-v2-gemma_vendor.json")["p_MRR"], 3) for c in ("Robust04", "News21", "Core17"))
check("4.4, Table 4 (a)", "p-MRR at the model-card placement -0.002, -0.010, -0.027 on Robust04, News21, Core17",
      ("-0.002", "-0.010", "-0.027"), _pm, "followir_{Robust04,News21,Core17}_gemma_bge-reranker-v2-gemma_vendor.json: p_MRR")

_fa = J(T + "r1_label_factorial.json")
_rr = _fa["resolved_rows_out_of_96"]
check("4.4, Table 5", "right program, wrong semester: the five models fall to 29 to 61 of 96", (29, 61),
      rng([_rr[m]["both_semsalah"] for m in _FIVE]), "r1_label_factorial.json: resolved_rows_out_of_96.*.both_semsalah")
_dw = rng([_rr[m]["both_duasalah"] for m in _FIVE])
check("4.4, Table 5", "both fields wrong: 31 to 39, below the blind baseline of 48", (31, 39), _dw,
      "r1_label_factorial.json: resolved_rows_out_of_96.*.both_duasalah", ok=_dw == (31, 39) and _dw[1] < 48)
for _c in ("semester_correct_vs_wrong_at_correct_program", "semester_correct_vs_wrong_at_wrong_program"):
    _ms = _fa["contrasts"][_c]["models"]
    _p = max(_ms[m]["p_holm_mcnemar_exact"] for m in _FIVE)
    check("4.4", f"fixing the semester raises the count on all five models, Holm p below 0.001 ({_c})", "< 0.001",
          f"largest Holm p {_p:.6f}", f"r1_label_factorial.json: contrasts.{_c}.models.*",
          ok=_p < 0.001 and all(_ms[m]["resolved_B"] > _ms[m]["resolved_A"] for m in _FIVE))
_prog = ("program_correct_vs_wrong_at_correct_semester", "program_correct_vs_wrong_at_wrong_semester")
check("4.4", "fixing the program is significant at both semester levels only on Qwen3-0.6B", ["Qwen3-0.6B"],
      [m for m in _FIVE if all(m in _fa["contrasts"][c]["summary"]["significant_after_holm_mcnemar"] for c in _prog)],
      "r1_label_factorial.json: contrasts.program_correct_vs_wrong_at_*.summary.significant_after_holm_mcnemar")
check("4.4", "where correcting the program is not significant, its family-wise limit reaches 17.0 percentage points", "17.0",
      _r(max(_fa["contrasts"][c]["models"][m]["family_wise"]["largest_gain_pp"]
             for c in ("program_correct_vs_wrong_at_correct_semester", "program_correct_vs_wrong_at_wrong_semester")
             for m in _FIVE if m not in _fa["contrasts"][c]["summary"]["significant_after_holm_mcnemar"]), 1),
      "r1_label_factorial.json: contrasts.program_correct_vs_wrong_*.models.*.family_wise.largest_gain_pp, models not significant after Holm")
_same = all(_fa["contrasts"][c]["without_coincident_rows"][m]["resolved_B"] > _fa["contrasts"][c]["without_coincident_rows"][m]["resolved_A"]
            for c in _prog + ("semester_correct_vs_wrong_at_correct_program", "semester_correct_vs_wrong_at_wrong_program")
            for m in _fa["contrasts"][c]["summary"]["significant_after_holm_mcnemar"])
check("4.4", "without the rows where a wrong label matched the other student's value, every significant contrast points the same way "
      "(the pre-registered sensitivity rule, 11.1 b)", True, _same, "r1_label_factorial.json: contrasts.*.without_coincident_rows")
check("4.4, Table 5", "control labels: random 43 to 49, the asker's own label on every candidate 46 to 48", ((43, 49), (46, 48)),
      (rng([_rr[m]["label_acak"] for m in _FIVE]), rng([_rr[m]["label_seragam"] for m in _FIVE])),
      "r1_label_factorial.json: resolved_rows_out_of_96.*.{label_acak, label_seragam}")
check("4.4", "neither control label differs from the slot alone", [],
      _fa["contrasts"]["random_label_vs_slot"]["summary"]["significant_after_holm_mcnemar"]
      + _fa["contrasts"]["uniform_label_vs_slot"]["summary"]["significant_after_holm_mcnemar"],
      "r1_label_factorial.json: contrasts.{random,uniform}_label_vs_slot.summary.significant_after_holm_mcnemar")
check("4.4", "on the five models the two control labels have family-wise limits of 14.9 and 12.1 percentage points", ("14.9", "12.1"),
      tuple(_r(max(_fa["contrasts"][c]["models"][m]["family_wise"]["largest_gain_pp"] for m in _FIVE), 1)
            for c in ("random_label_vs_slot", "uniform_label_vs_slot")),
      "r1_label_factorial.json: contrasts.{random,uniform}_label_vs_slot.models.*.family_wise.largest_gain_pp")

_la = J(T + "r1_random_broad.json")
check("3.4", "second broad sample: 600 groups and 1,617 rows", (600, 1617), (_la["n_groups"], _la["n_rows"]), "r1_random_broad.json: n_groups, n_rows")
check("3.4", "286 groups of gate version 1 and 314 of version 2", (286, 314),
      (_la["by_gate"]["slot_vs_none"]["v1-cps-gate"]["n_groups"], _la["by_gate"]["slot_vs_none"]["v2-multiarm"]["n_groups"]),
      "r1_random_broad.json: by_gate.slot_vs_none.*.n_groups")
check("4.2", "limits of the first and the second broad sample, 0.89 and 1.42", ("0.89", "1.42"),
      (_r(J(T + "r1_cluster_broad.json")["contrasts"]["slot_vs_none"]["summary"]["largest_gain_pp_max_over_models"], 2),
       _r(_la["contrasts"]["slot_vs_none"]["summary"]["largest_gain_pp_max_over_models"], 2)),
      "r1_cluster_broad.json, r1_random_broad.json: contrasts.slot_vs_none.summary.largest_gain_pp_max_over_models")
check("4.2", "family-wise limits of the first and the second broad sample, 1.31 and 1.81", ("1.31", "1.81"),
      (_r(J(T + "r1_cluster_broad.json")["contrasts"]["slot_vs_none"]["summary"]["family_wise_largest_gain_pp_max_over_models"], 2),
       _r(_la["contrasts"]["slot_vs_none"]["summary"]["family_wise_largest_gain_pp_max_over_models"], 2)),
      "r1_cluster_broad.json, r1_random_broad.json: contrasts.slot_vs_none.summary.family_wise_largest_gain_pp_max_over_models")

_p2 = J(T + "p2_paired.json")
_sd = _p2["sample_design"]
_pub = J(T + "p2_population_public.json")["records"]
check("3.2", "2,587 topic pairs sent over six rounds", (2587, 6), (len(_pub), len(_sd["stopping"]["rounds"])),
      "p2_population_public.json: records; p2_paired.json: sample_design.stopping.rounds")
check("3.2", "the writers declined 1,307; 1,280 questions written; the screen passed 616",
      (1307, 1280, 616), (sum(1 for r in _pub if not r["written"]), sum(1 for r in _pub if r["written"]), sum(1 for r in _pub if r["passed_screen"])),
      "p2_population_public.json: records[].written, passed_screen")
check("3.2", "480 groups, two rows each (960 rows)", (480, 960), (_p2["n_groups"], _p2["n_rows"]), "p2_paired.json: n_groups, n_rows")
check("6", "the judge rated both passages of 436 of the 480 extended-track groups as partial answers", (436, 480),
      (sum(1 for r in _pub if r.get("group") and r.get("class") == "sebagian"), _p2["n_groups"]),
      "p2_population_public.json: records[].group, class; p2_paired.json: n_groups")
check("3.2", "five program pairs: ISYS-MGMT 251, ACCT-MGMT 89, CSCI-MGMT 69, CSCI-ISYS 44, ACCT-ISYS 27",
      {"ACCT-ISYS": 27, "ACCT-MGMT": 89, "CSCI-ISYS": 44, "CSCI-MGMT": 69, "ISYS-MGMT": 251}, _sd["groups_by_program_pair"],
      "p2_paired.json: sample_design.groups_by_program_pair")
_pp2 = [len(json.loads(x)["cands"]) for x in (ROOT / "study/channel/results/raw_p2/pools/pools_p2_ids.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
check("3.2", "one shared pool of 71 candidates (the two passages plus 69 chunks)", (71, (71, 71)), (_sd["rules"]["pool_size"], rng(_pp2)),
      "p2_paired.json: sample_design.rules.pool_size; raw_p2/pools/pools_p2_ids.jsonl: cands of every row")
check("3.2", "the stricter judge rule gave 13 groups in the first round; 43 groups meet it in all", (13, 43),
      (_sd["stopping"]["rounds"][0]["strict_groups"], _p2["strict_subset"]["n_groups"]),
      "p2_paired.json: sample_design.stopping.rounds[0].strict_groups, strict_subset.n_groups")
check("3.2", "the blind baseline is 480 of 960 for every model", {480},
      {v["resolved"] for v in _p2["blind_invariant"].values()}, "p2_paired.json: blind_invariant.*.resolved")
_sv = _p2["contrasts"]["slot_vs_none"]["models"]
check("4.2", "with the slot the six rerankers resolve 476 to 494 of 960", (476, 494), rng([_sv[m]["resolved_B"] for m in _SIXR]),
      "p2_paired.json: contrasts.slot_vs_none.models.*.resolved_B")
check("4.2", "no slot change is significant after Holm", [], _p2["contrasts"]["slot_vs_none"]["summary"]["significant_after_holm_mcnemar"],
      "p2_paired.json: contrasts.slot_vs_none.summary.significant_after_holm_mcnemar")
check("4.2, 7", "largest gain 4.40 percentage points, on zerank-2", ("4.40", "zerank-2"),
      (_r(_p2["contrasts"]["slot_vs_none"]["summary"]["largest_gain_pp_max_over_models"], 2), max(_SIXR, key=lambda m: _sv[m]["largest_gain_pp"])),
      "p2_paired.json: contrasts.slot_vs_none.summary.largest_gain_pp_max_over_models")
check("abstract", "the gain's 95 percent upper limit is at most 4.4 percentage points, or 5.4 family-wise", ("4.4", "5.4"),
      (_r(_p2["contrasts"]["slot_vs_none"]["summary"]["largest_gain_pp_max_over_models"], 1),
       _r(_p2["contrasts"]["slot_vs_none"]["summary"]["family_wise_largest_gain_pp_max_over_models"], 1)),
      "p2_paired.json: contrasts.slot_vs_none.summary.{largest_gain_pp_max_over_models, family_wise_largest_gain_pp_max_over_models}")
check("4.2, 7", "family-wise, the largest gain is 5.36 on the extended paired track and 16.19 on the rebuilt broad track", ("5.36", "16.19"),
      (_r(_p2["contrasts"]["slot_vs_none"]["summary"]["family_wise_largest_gain_pp_max_over_models"], 2),
       _r(J(T + "r2_broad.json")["contrasts"]["slot_vs_none"]["summary"]["family_wise_largest_gain_pp_max_over_models"], 2)),
      "p2_paired.json, r2_broad.json: contrasts.slot_vs_none.summary.family_wise_largest_gain_pp_max_over_models")
_kp = J(T + "r1_cluster_paired.json")["contrasts"]["slot_vs_none"]
check("4.2", "on the 52 original pairs the largest gain, 13.71 or 17.01 family-wise, belongs to ctxl-6B", ("13.71", "17.01", "ctxl-6B", "ctxl-6B"),
      (_r(_kp["summary"]["largest_gain_pp_max_over_models"], 2), _r(_kp["summary"]["family_wise_largest_gain_pp_max_over_models"], 2),
       max(_SIXR, key=lambda m: _kp["models"][m]["largest_gain_pp"]), max(_SIXR, key=lambda m: _kp["models"][m]["family_wise"]["largest_gain_pp"])),
      "r1_cluster_paired.json: contrasts.slot_vs_none.{summary, models.*.largest_gain_pp, models.*.family_wise.largest_gain_pp}")
_bpp = _p2["by_program_pair"]["slot_vs_none"]
_small = min(_bpp, key=lambda k: _bpp[k]["n_groups"])
check("4.2", "the smallest program pair, ACCT-ISYS with 27 groups, allows the largest gain of any pair, 17.75 percentage points or 22.92 "
      "family-wise", ("ACCT-ISYS", 27, "17.75", "22.92", True),
      (_small, _bpp[_small]["n_groups"], _r(_bpp[_small]["summary"]["largest_gain_pp_max_over_models"], 2),
       _r(_bpp[_small]["summary"]["family_wise_largest_gain_pp_max_over_models"], 2),
       all(_bpp[_small]["summary"]["largest_gain_pp_max_over_models"] >= v["summary"]["largest_gain_pp_max_over_models"] for v in _bpp.values())),
      "p2_paired.json: by_program_pair.slot_vs_none.*.{n_groups, summary}")
check("4.2", "13.71 on the 52 original pairs", "13.71", _r(_p2["paired_52_questions_reference"]["largest_gain_pp_max_over_models"], 2),
      "p2_paired.json: paired_52_questions_reference.largest_gain_pp_max_over_models")
check("4.2", "no program pair shows a significant gain", [],
      sorted(k for k, v in _p2["by_program_pair"]["slot_vs_none"].items() if v.get("summary", {}).get("significant_after_holm_mcnemar")),
      "p2_paired.json: by_program_pair.slot_vs_none.*.summary.significant_after_holm_mcnemar")
check("4.2", "the 43 groups of the stricter rule: largest gain 11.98, or 15.32 family-wise", ("11.98", "15.32"),
      (_r(_p2["strict_subset"]["families"]["slot_vs_none"]["summary"]["largest_gain_pp_max_over_models"], 2),
       _r(_p2["strict_subset"]["families"]["slot_vs_none"]["summary"]["family_wise_largest_gain_pp_max_over_models"], 2)),
      "p2_paired.json: strict_subset.families.slot_vs_none.summary")
_sf = _p2["strict_subset"]["families"]
check("4.4", "in the 43 strict groups the label lift stays significant on the same five models, and a wrong program lowers four of them "
      "significantly, with Qwen3-4B in place of FollowIR-7B", (43, sorted(_FIVE), ["Qwen3-0.6B", "Qwen3-4B", "ctxl-6B", "zerank-2"], True),
      (_p2["strict_subset"]["n_groups"], sorted(_sf["both_vs_slot"]["summary"]["significant_after_holm_mcnemar"]),
       sorted(_sf["wrong_vs_both"]["summary"]["significant_after_holm_mcnemar"]),
       all(_sf["wrong_vs_both"]["models"][m]["resolved_B"] < _sf["wrong_vs_both"]["models"][m]["resolved_A"]
           for m in _sf["wrong_vs_both"]["summary"]["significant_after_holm_mcnemar"])),
      "p2_paired.json: strict_subset.{n_groups, families.both_vs_slot, families.wrong_vs_both}")
check("4.2", "zerank-2: 494 of 960, 14 rows above the baseline, not significant", (494, 14, False),
      (_sv["zerank-2"]["resolved_B"], _sv["zerank-2"]["resolved_B"] - 480, _sv["zerank-2"]["p_holm_mcnemar_exact"] < 0.05),
      "p2_paired.json: contrasts.slot_vs_none.models.zerank-2")
_bs = _p2["contrasts"]["both_vs_slot"]["models"]
_lift = {m: (_bs[m]["resolved_B"] - _bs[m]["resolved_A"]) / _p2["n_rows"] * 100 for m in _SIXR}
check("4.4, 7", "with the label the five models reach 691 to 929 of 960", (691, 929), rng([_bs[m]["resolved_B"] for m in _FIVE]),
      "p2_paired.json: contrasts.both_vs_slot.models.*.resolved_B")
check("4.4", "21.7 to 45.4 percentage points above the slot alone (Holm p < 0.001)", ("21.7", "45.4", True),
      (_r(min(_lift[m] for m in _FIVE), 1), _r(max(_lift[m] for m in _FIVE), 1), max(_bs[m]["p_holm_mcnemar_exact"] for m in _FIVE) < 0.001),
      "p2_paired.json: contrasts.both_vs_slot.models.*")
check("4.4", "bge-gemma at its model-card placement moves 1.7 points (Holm p = 0.085)", ("1.7", "0.085"),
      (_r(_lift["bge-gemma"], 1), _r(_bs["bge-gemma"]["p_holm_mcnemar_exact"], 3)), "p2_paired.json: contrasts.both_vs_slot.models.bge-gemma")
check("4.4", "bge-gemma at its model-card placement: family-wise limit 4.1", "4.1",
      _r(_bs["bge-gemma"]["family_wise"]["largest_gain_pp"], 1), "p2_paired.json: contrasts.both_vs_slot.models.bge-gemma.family_wise.largest_gain_pp")
_wb = _p2["contrasts"]["wrong_vs_both"]["models"]
_drop = {m: (_wb[m]["resolved_A"] - _wb[m]["resolved_B"]) / _p2["n_rows"] * 100 for m in _FIVE}
_four = [m for m in _FIVE if m != "Qwen3-4B"]
check("4.4", "a wrong program lowers four of the five models by 4.6 to 15.9 points (Holm p < 0.001)", ("4.6", "15.9", True),
      (_r(min(_drop[m] for m in _four), 1), _r(max(_drop[m] for m in _four), 1), max(_wb[m]["p_holm_mcnemar_exact"] for m in _four) < 0.001),
      "p2_paired.json: contrasts.wrong_vs_both.models.*")
check("4.4", "and Qwen3-4B by 1.0 point (Holm p = 0.43)", ("1.0", "0.43"),
      (_r(_drop["Qwen3-4B"], 1), _r(_wb["Qwen3-4B"]["p_holm_mcnemar_exact"], 2)), "p2_paired.json: contrasts.wrong_vs_both.models.Qwen3-4B")
check("4.4", "Qwen3-4B with a wrong program: family-wise limit of the drop 4.2", "4.2",
      _r(_wb["Qwen3-4B"]["family_wise"]["largest_loss_pp"], 1), "p2_paired.json: contrasts.wrong_vs_both.models.Qwen3-4B.family_wise.largest_loss_pp")
check("4.5", "DITS resolves 903 of 960", (903, 960), (_p2["metadata_references"]["DITS_resolved"], _p2["metadata_references"]["DITS_rows"]),
      "p2_paired.json: metadata_references")
check("4.5", "with the label, zerank-2 (929) and ctxl-6B (928) resolve more rows than DITS on the extended track", (929, 928, True),
      (_bs["zerank-2"]["resolved_B"], _bs["ctxl-6B"]["resolved_B"],
       min(_bs["zerank-2"]["resolved_B"], _bs["ctxl-6B"]["resolved_B"]) > _p2["metadata_references"]["DITS_resolved"]),
      "p2_paired.json: contrasts.both_vs_slot.models, metadata_references.DITS_resolved")

_r2 = J(T + "r2_broad.json")
_pop = _r2["sample_design"]["population"]
_mr2 = J("study/channel/results/raw_r2/pools_xpair_r2.manifest.json")
check("3.4", "2,021 queries with a single gold; 1,850 judged clear; 541 with a full-answer gold",
      (2021, 1850, 541), (_pop["eligible_r1_rule"] - _pop["exclusions"]["more_than_one_gold"], _pop["question_clear"]["ya"],
                          _pop["question_clear"]["ya"] - _pop["exclusions"]["own_gold_not_full"]),
      "r2_broad.json: sample_design.population")
check("3.4", "26 groups and 55 rows; 11 from gate version 1 and 15 from version 2", (26, 55, 11, 15),
      (_r2["n_groups"], _r2["n_rows"], _pop["groupable_by_gate"]["v1-cps-gate"], _pop["groupable_by_gate"]["v2-multiarm"]),
      "r2_broad.json: n_groups, n_rows, sample_design.population.groupable_by_gate")
_pr2 = [len(json.loads(x)["cands"]) for x in (ROOT / "study/channel/results/raw_r2/pools/pools_xpair_r2_ids.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
check("3.4", "pools of 68 to 70 candidates, at most two partners, five closest chunks per direction", ((68, 70), 2, True),
      (rng(_pr2), _mr2["rules"]["max_partners"], "top 5" in _mr2["rules"]["candidates_per_program"]),
      "raw_r2/pools/pools_xpair_r2_ids.jsonl: cands of every row; raw_r2/pools_xpair_r2.manifest.json: rules")
_ps = _mr2["prompt_stability"]
check("3.4", "asked again in the other wording, 300 random verdicts kept their class in 91% of cases (Cohen's kappa 0.66)",
      (300, "91", "0.66"), (_ps["n"], _r(100 * _ps["agreement_three_classes"], 0), _r(_ps["kappa_three_classes"], 2)),
      "raw_r2/pools_xpair_r2.manifest.json: prompt_stability.{n, agreement_three_classes, kappa_three_classes}")
check("3.2, 3.4", "the judge is Qwen3-14B-AWQ", "Qwen/Qwen3-14B-AWQ", _r2["sample_design"]["judge"]["model"],
      "r2_broad.json: sample_design.judge.model")
_kit = _mr2["first_kit_judged"]["by_role"]
check("3.4", "on 55 groups of the first broad sample the judge rated 17 of the 55 original golds and none of the 94 mined partner golds "
      "a full answer under both wordings", (55, 17, 94, 0),
      (_kit["original_gold"]["items"], _kit["original_gold"]["full_both_wordings"], _kit["mined_partner_gold"]["items"],
       _kit["mined_partner_gold"]["full_both_wordings"]), "raw_r2/pools_xpair_r2.manifest.json: first_kit_judged.by_role")
check("3.4", "the blind baseline is 26 of 55 for every model", {26}, {v["resolved"] for v in _r2["blind_invariant"].values()},
      "r2_broad.json: blind_invariant.*.resolved")
_sv2 = _r2["contrasts"]["slot_vs_none"]
check("4.2", "the slot lands at 25 to 28 of 55, no change significant", ((25, 28), []),
      (rng([_sv2["models"][m]["resolved_B"] for m in _SIXR]), _sv2["summary"]["significant_after_holm_mcnemar"]),
      "r2_broad.json: contrasts.slot_vs_none")
check("4.2", "largest gain 12.50 percentage points, on zerank-2", ("12.50", "zerank-2"),
      (_r(_sv2["summary"]["largest_gain_pp_max_over_models"], 2), max(_SIXR, key=lambda m: _sv2["models"][m]["largest_gain_pp"])),
      "r2_broad.json: contrasts.slot_vs_none.summary.largest_gain_pp_max_over_models")
_eb = _r2["earlier_broad_samples_unchecked_partners"]
check("4.2", "beside it: the first broad sample's limit 0.89 and the second sample's 1.42", ("0.89", "1.42"),
      (_r(_eb["original_sample"]["largest_gain_pp_max_over_models"], 2), _r(_eb["r1_resample"]["largest_gain_pp_max_over_models"], 2)),
      "r2_broad.json: earlier_broad_samples_unchecked_partners")
_bs2 = _r2["contrasts"]["both_vs_slot"]
_lift2 = {m: _r((_bs2["models"][m]["resolved_B"] - _bs2["models"][m]["resolved_A"]) / _r2["n_rows"] * 100, 1) for m in _SIXR}
check("4.4", "the label lifts Qwen3-0.6B, zerank-2, and ctxl-6B significantly, by 23.6, 38.2, and 29.1 points",
      (["Qwen3-0.6B", "zerank-2", "ctxl-6B"], ("23.6", "38.2", "29.1")),
      (_bs2["summary"]["significant_after_holm_mcnemar"], (_lift2["Qwen3-0.6B"], _lift2["zerank-2"], _lift2["ctxl-6B"])),
      "r2_broad.json: contrasts.both_vs_slot")
_wb2 = _r2["contrasts"]["wrong_vs_both"]["models"]["zerank-2"]
check("4.4", "the wrong program cost zerank-2 9 of 55 rows (Holm p = 0.023)", (9, "0.023"),
      (_wb2["resolved_A"] - _wb2["resolved_B"], _r(_wb2["p_holm_mcnemar_exact"], 3)), "r2_broad.json: contrasts.wrong_vs_both.models.zerank-2")
check("4.5", "DITS resolves 46 of 55; with the label zerank-2 passes it at 48", (46, 55, 48, True),
      (_r2["metadata_references"]["DITS_resolved"], _r2["metadata_references"]["DITS_rows"], _bs2["models"]["zerank-2"]["resolved_B"],
       _bs2["models"]["zerank-2"]["resolved_B"] > _r2["metadata_references"]["DITS_resolved"]), "r2_broad.json: metadata_references, contrasts.both_vs_slot")

_T3 = {
    "p2": {"Qwen3-0.6B": ("476", "-4", "0.81", "2.21", "3.02", "827"), "Qwen3-4B": ("483", "+3", "0.88", "3.00", "2.38", "691"),
           "zerank-2": ("494", "+14", "0.36", "4.40", "1.53", "929"), "FollowIR-7B": ("481", "+1", "1.00", "2.55", "2.35", "720"),
           "ctxl-6B": ("492", "+12", "0.41", "4.03", "1.57", "928"), "bge-gemma": ("479", "-1", "1.00", "1.30", "1.50", "495")},
    "r2": {"Qwen3-0.6B": ("25", "-1", "1.00", "6.42", "9.77", "38"), "Qwen3-4B": ("26", "0", "1.00", "7.13", "7.13", "33"),
           "zerank-2": ("27", "+1", "1.00", "12.50", "9.43", "48"), "FollowIR-7B": ("27", "+1", "1.00", "9.77", "6.42", "36"),
           "ctxl-6B": ("28", "+2", "0.50", "8.75", "2.49", "44"), "bge-gemma": ("25", "-1", "1.00", "1.94", "5.57", "27")},
}
for _track, _tab, _blind in (("p2", _p2, 480), ("r2", _r2, 26)):
    for _m, _want in _T3[_track].items():
        _s = _tab["contrasts"]["slot_vs_none"]["models"][_m]
        _b = _tab["contrasts"]["both_vs_slot"]["models"][_m]
        _ch = _s["resolved_B"] - _blind
        _got = (str(_s["resolved_B"]), ("+" if _ch > 0 else "") + str(_ch), _r(_s["p_mcnemar_exact"], 2),
                _r(_side(_s, "gain"), 2), _r(_side(_s, "loss"), 2), str(_b["resolved_B"]))
        check("Table 3", f"{'extended paired' if _track == 'p2' else 'rebuilt broad'} track, {_m}", _want, _got,
              f"{_track}_{'paired' if _track == 'p2' else 'luas'}.json: contrasts.{{slot_vs_none, both_vs_slot}}.models.{_m}")

import ast as _ast
_spool = J(T + "pool_statistics.json")
_tpair, _tbroad = _spool["track_berpasangan"], _spool["track_luas"]
_cluster_broad = J(T + "r1_cluster_broad.json")["contrasts"]
check("abstract, 1, 4.1, 4.2, 4.3, 4.4, 4.5, 5, 6, 7, Table 3, Table 4",
      "track sizes where the text repeats them: 52 pairs and 96 rows (original paired track), 480 pairs and 960 rows (extended "
      "paired track), 26 groups and 55 rows (rebuilt broad track), 600 groups and 1,621 rows (first broad sample)",
      (52, 96, 480, 960, 26, 55, 600, 1621),
      (_tpair["pasangan"], _tpair["baris_komparabel"], _p2["n_groups"], _p2["n_rows"], _r2["n_groups"], _r2["n_rows"],
       _tbroad["grup"], _tbroad["baris"]),
      "pool_statistics.json: track_berpasangan.{pasangan, baris_komparabel}, track_luas.{grup, baris}; p2_paired.json, r2_broad.json: n_groups, n_rows")
check("4.2 (provenance, not a sentence of the paper)", "the extended paired track has ten times as many rows as the original one", 10, _p2["n_rows"] / _tpair["baris_komparabel"],
      "p2_paired.json: n_rows; pool_statistics.json: track_berpasangan.baris_komparabel")
check("3.2, 4.2, 6, 7, Table 3", "blind baselines where the text repeats them: 48 of 96, 480 of 960, 26 of 55, 600 of 1,621",
      ([48], [480], [26], [600]),
      (sorted({x["resolved_A"] for c in _og.values() for x in c["vs_blind"].values()}),
       sorted({v["resolved"] for v in _p2["blind_invariant"].values()}), sorted({v["resolved"] for v in _r2["blind_invariant"].values()}),
       sorted({_cluster_broad["slot_vs_none"]["models"][m]["resolved_A"] for m in _SIXR})),
      "r1_slot_orthogonal.json: cells.*.vs_blind.*.resolved_A; p2_paired.json, r2_broad.json: blind_invariant; "
      "r1_cluster_broad.json: contrasts.slot_vs_none.models.*.resolved_A")
check("abstract, 7, Table 5", "with the label the five models reach 83 to 96 of 96", (83, 96), rng([_rr[m]["both"] for m in _FIVE]),
      "r1_label_factorial.json: resolved_rows_out_of_96.*.both")
check("1", "64 to 80 candidate passages per question", (64, 80),
      (_tpair["kandidat_unik_per_baris"]["min"], _tpair["kandidat_unik_per_baris"]["max"]), "pool_statistics.json: track_berpasangan.kandidat_unik_per_baris")
_bcomp = J(R + "benchmark_composition.json")
check("3.4", "the first broad sample read the 2,493 training and development queries", 2493,
      sum(_bcomp["per_split"]["traindev"]["queries_by_gold_count"].values()), "benchmark_composition.json: per_split.traindev.queries_by_gold_count")

_cos = [r["cosine"] for r in _pub]
_jac = [r["jaccard"] for r in _pub]
check("3.2", "every topic pair of the extended track lies within the limits of Section 3.4 (similarity 0.66 to 0.89, word overlap at most 0.36)",
      (0.66, 0.89, 0.36), (min(_cos), max(_cos), max(_jac)), "p2_population_public.json: records[].cosine, jaccard (observed minimum, maximum, maximum)",
      ok=min(_cos) >= 0.66 and max(_cos) <= 0.89 and max(_jac) <= 0.36)
_anchors = set()
for _line in (ROOT / "study/channel/provenance/paired_queries.jsonl").read_text(encoding="utf-8").splitlines():
    if _line.strip():
        _q = json.loads(_line)
        for _k in ("gold_anchor", "counterpart_anchor"):
            _a = _q.get(_k) or []
            _a = _ast.literal_eval(_a) if isinstance(_a, str) else _a
            _anchors |= {x["canonical_id"] for x in _a if x.get("canonical_id")}
_p2_chunks = {p["chunk"] for r in _pub for p in r["passages"]}
_tpm = J("study/channel/results/raw_p2/pasangan_topik_p2.manifest.json")
check("3.2", "no passage of the original paired track was reused (and the pairing step dropped those passages by id or text hash)",
      (0, 104, True), (len(_p2_chunks & _anchors), len(_anchors), _tpm["counts"]["chunks_dropped"]["paired_track_chunk"] > 0),
      "provenance/paired_queries.jsonl: gold_anchor, counterpart_anchor; p2_population_public.json: records[].passages; "
      "raw_p2/pasangan_topik_p2.manifest.json: counts.chunks_dropped.paired_track_chunk")
_acct = [r["order"] for r in _pub if "ACCT" in r["program_pair"]]
_rest = [r["order"] for r in _pub if "ACCT" not in r["program_pair"]]
check("3.2", "ACCT pairs were sent only after the pairs of the other three programs", True, min(_acct) > max(_rest),
      "p2_population_public.json: records[].order, program_pair")
_pools = [json.loads(x) for x in (ROOT / "study/channel/results/raw_p2/pools/pools_p2_ids.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
_others = {len(set(r["cands"]) - set(r["gold"]) - set(r["counterpart"])) for r in _pools}
check("3.2", "each pool holds the two passages plus the 69 chunks closest to either", ({69}, True), (_others, all(
      set(r["gold"]) | set(r["counterpart"]) <= set(r["cands"]) and len(r["cands"]) == 71 for r in _pools)),
      "raw_p2/pools/pools_p2_ids.jsonl: cands, gold, counterpart")

_ml = set()
for _c in ("Robust04", "News21", "Core17"):
    for _q in ("Qwen3-Reranker-0.6B", "Qwen3-Reranker-4B"):
        _ml |= set(re.findall(r"--max-len\s+(\d+)", J(f"study/channel/results/raw/followir_{_c}_qwen3rr_{_q}.manifest.json")["perintah_persis"]))
check("Table 2", "the two Qwen3 models ran FollowIR with 1,024 tokens", {"1024"}, _ml,
      "raw/followir_{Robust04,News21,Core17}_qwen3rr_Qwen3-Reranker-{0.6B,4B}.manifest.json: perintah_persis")
_tmod = {m["model"]: m for m in J(T + "model_table.json")["model"]}
_family = {"qwen3": "Qwen3", "mistral": "Mistral", "gemma": "Gemma"}
check("Table 2", "Family column: Qwen3 for Qwen3-0.6B, Qwen3-4B, and zerank-2, Mistral for FollowIR-7B and ctxl-6B, Gemma for bge-gemma",
      ["Qwen3", "Qwen3", "Qwen3", "Mistral", "Mistral", "Gemma"],
      [_family.get(_tmod[k]["model_type"]) for k in ("Qwen3-Reranker-0.6B", "Qwen3-Reranker-4B", "zerank-2", "FollowIR-7B", "ctxl-6B", "gemma")],
      "model_table.json: model[].model_type")
_judges = {json.loads(x)["model"] for x in (ROOT / "study/channel/results/raw_p2/juri_p2.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()}
check("6", "the judge of both new tracks is from the same model family as three of the rerankers (Table 2)", (3, True),
      (sum(1 for m in _tmod.values() if m["model_type"] == "qwen3"),
       _judges == {_r2["sample_design"]["judge"]["model"]} and _r2["sample_design"]["judge"]["model"].startswith("Qwen/Qwen3-")),
      "model_table.json: model[].model_type; raw_p2/juri_p2.jsonl: model; r2_broad.json: sample_design.judge.model")
_pairs_per_cell = set()
for _f in ("injection_probe_qwen3rr_Qwen3-Reranker-0.6B.json", "injection_probe_qwen3rr_Qwen3-Reranker-4B.json", "injection_probe_zerank2official_zerank-2.json"):
    _pairs_per_cell |= {v["n_komparabel"] / 2 for v in J(T + _f)["agregat_per_kombinasi"].values()}
check("3.5", "the format sweep has 10 pairs per condition", {10}, _pairs_per_cell, "injection_probe_*.json: agregat_per_kombinasi.*.n_komparabel / 2")
_fcal_board = J(T + "followir_calibration.json")["papan"]
_coll = ("Robust04", "News21", "Core17")
_five_pos = all(_fcal_board[c]["model"][m]["positif"] for c in _coll for m in ("Qwen3-Reranker-0.6B", "Qwen3-Reranker-4B", "zerank-2", "FollowIR-7B", "ctxl-6B"))
_gemma_doc = [J(T + f"followir_{c}_gemma_bge-reranker-v2-gemma_vendor.json")["p_MRR"] for c in _coll]
check("4.3", "all six models get a positive p-MRR on FollowIR with the prompts of the runs, and bge-gemma at its model-card "
      "placement is positive on none", (True, True, True),
      (_five_pos, all(_fcal_board[c]["model"]["bge-reranker-v2-gemma"]["positif"] for c in _coll), max(_gemma_doc) <= 0),
      "followir_calibration.json: papan.*.model.*.positif; followir_*_gemma_bge-reranker-v2-gemma_vendor.json: p_MRR")
_ctx = J(T + "control_textual.json")
check("4.3", "the textual control: three models, the full set of 58 pairs, and 36 filtered pairs whose section titles pass four rules",
      (3, 58, 36, 4), (len(_ctx["populasi_penuh"]["model"]), _ctx["populasi_penuh"]["n_pasangan"], _ctx["populasi_tersaring"]["n_pasangan"],
                       len(set(re.findall(r"\bT(\d)\b", _ctx["filter_rule_verbatim_en"])))),
      "control_textual.json: populasi_penuh, populasi_tersaring, filter_rule_verbatim_en")

_g0c = J(T + "r1_gate_g0.json")["cells"]
_refs = {k: v for k, v in _g0c.items() if k.split("|")[0].startswith("Qwen3") and k.split("|")[1] in ("none", "slot")}
_gates = {k: v for k, v in _g0c.items() if k not in _refs}
check("3.7 (provenance, not printed in the paper)", "seven existing conditions were scored again first, and every score came back identical", (7, True),
      (len(_gates), all(v["flips"] == 0 and v["max_abs_score_diff"] == 0 for v in _gates.values())), "r1_gate_g0.json: cells (the gate cells)")
_fw = [v["family_wise"] for _t in (_p2, _r2) for v in _t["contrasts"]["slot_vs_none"]["models"].values()]
check("3.7", "a family-wise bound splits the 5% error level over the tests of its family (six models: 0.05 / 6 each)", ("5", {6}),
      (_r(100 * max(f["m"] * f["alpha_each"] for f in _fw), 0), {f["m"] for f in _fw}),
      "p2_paired.json, r2_broad.json: contrasts.slot_vs_none.models.*.family_wise.{m, alpha_each}")
check("3.7 (provenance, not printed in the paper)", "four more Qwen3 conditions without a label were scored again, and their scores moved by at most 0.015", (4, "0.015"),
      (len(_refs), _r(max(v["max_abs_score_diff"] for v in _refs.values()), 3)), "r1_gate_g0.json: cells.Qwen3-*|{none, slot}")
check("3.7 (provenance, not printed in the paper)", "the Qwen3 conditions without a label, scored again, are the references of the new Qwen3 conditions; one paired row of "
      "Qwen3-0.6B with the slot changed its outcome, so that reference is 50 of 96, against 49 in Table 3", (1, 50, 49, True),
      (sum(v["flips"] for v in _refs.values()), _refs["Qwen3-0.6B|slot"]["resolved_rerun"], _refs["Qwen3-0.6B|slot"]["resolved_reference"],
       _rr["Qwen3-0.6B"]["slot"] == 50 and all(c["vs_full_profile_slot"]["Qwen3-0.6B"]["resolved_A"] == 50 for c in _og.values())),
      "r1_gate_g0.json: cells.Qwen3-*|{none, slot}; r1_label_factorial.json: resolved_rows_out_of_96.Qwen3-0.6B.slot; "
      "r1_slot_orthogonal.json: cells.*.vs_full_profile_slot.Qwen3-0.6B.resolved_A")

check("4.2", "the 43 groups of the stricter judge rule", 43, _p2["strict_subset"]["n_groups"], "p2_paired.json: strict_subset.n_groups")
check("4.2", "the six conditions: the program or the semester alone, the same semester (1 or 2), the same program (CSCI or STAT)",
      ["slot_prodtetap_csci", "slot_prodtetap_stat", "slot_program", "slot_semester", "slot_semtetap1", "slot_semtetap2"], sorted(_og),
      "r1_slot_orthogonal.json: cells")
_zb = _cluster_broad["slot_vs_none"]["models"]["zerank-2"]
check("4.2", "zerank-2 on the first broad sample: 597 of 1,621, three rows below the baseline", (597, 3), (_zb["resolved_B"], _zb["resolved_A"] - _zb["resolved_B"]),
      "r1_cluster_broad.json: contrasts.slot_vs_none.models.zerank-2")
check("4.4", "bge-gemma resolves 43 of 96 with the slot alone", 43, _rr["bge-gemma"]["slot"], "r1_label_factorial.json: resolved_rows_out_of_96.bge-gemma.slot")
_tracks = {"original paired": J(T + "r1_cluster_paired.json"), "extended paired": _p2, "rebuilt broad": _r2,
           "first broad sample": J(T + "r1_cluster_broad.json"), "second broad sample": _la}
check("4.2", "a sign-flip test on whole groups agrees on all three tracks and both earlier broad samples, with no significant slot change",
      {k: [] for k in _tracks}, {k: v["contrasts"]["slot_vs_none"]["summary"]["significant_after_holm_group_test"] for k, v in _tracks.items()},
      "r1_cluster_paired.json, p2_paired.json, r2_broad.json, r1_cluster_broad.json, r1_random_broad.json: "
      "contrasts.slot_vs_none.summary.significant_after_holm_group_test")
_lab_p2, _lab_r2 = _p2["contrasts"]["both_vs_slot"], _r2["contrasts"]["both_vs_slot"]
check("4.4", "on whole groups the sign-flip test also finds the small lift of bge-gemma (Holm p = 0.007) and, on the rebuilt track, that "
      "of FollowIR-7B (Holm p = 0.023); McNemar's test finds neither", (["bge-gemma"], ["FollowIR-7B"], "0.007", "0.023"),
      (sorted(set(_lab_p2["summary"]["significant_after_holm_group_test"]) - set(_lab_p2["summary"]["significant_after_holm_mcnemar"])),
       sorted(set(_lab_r2["summary"]["significant_after_holm_group_test"]) - set(_lab_r2["summary"]["significant_after_holm_mcnemar"])),
       _r(_lab_p2["models"]["bge-gemma"]["p_holm_signflip_group"], 3), _r(_lab_r2["models"]["FollowIR-7B"]["p_holm_signflip_group"], 3)),
      "p2_paired.json, r2_broad.json: contrasts.both_vs_slot.summary.significant_after_holm_{group_test, mcnemar}, models.*.p_holm_signflip_group")
_sem_kept, _prog_lost = True, []
for _c in _prog + ("semester_correct_vs_wrong_at_correct_program", "semester_correct_vs_wrong_at_wrong_program"):
    for _m in _fa["contrasts"][_c]["summary"]["significant_after_holm_mcnemar"]:
        if _fa["contrasts"][_c]["without_coincident_rows"][_m]["p_holm_mcnemar_exact"] >= 0.05:
            if _c.startswith("semester"):
                _sem_kept = False
            else:
                _prog_lost.append(_m)
check("4.4", "without those rows the semester contrasts stay significant, and three program contrasts do not, among them Qwen3-0.6B "
      "at the right semester (Holm p = 0.31)", (True, 3, "0.31"),
      (_sem_kept, len(_prog_lost),
       _r(_fa["contrasts"]["program_correct_vs_wrong_at_correct_semester"]["without_coincident_rows"]["Qwen3-0.6B"]["p_holm_mcnemar_exact"], 2)),
      "r1_label_factorial.json: contrasts.*.without_coincident_rows.*.p_holm_mcnemar_exact")
_ref = J(T + "channel_board_paired.json")["arm_referensi_terbit"]
check("4.5", "with metadata, DITS resolves 93 of 96 and a hard filter 96", ("93/96", "96/96"), (_ref["B_DITS"]["eksak"], _ref["filter_metadata"]["eksak"]),
      "channel_board_paired.json: arm_referensi_terbit.B_DITS.eksak, filter_metadata.eksak")

_gi = {}
for _k, _t in (("extended paired", _p2), ("rebuilt broad", _r2)):
    _ms = _t["contrasts"]["slot_vs_none"]["models"]
    _gi[_k] = (_r(100 * max(_ms[m]["group_interval"][1] for m in _SIXR), 2),
               _r(100 * max(_ms[m]["family_wise"]["group_interval"][1] for m in _SIXR), 2))
check("4.2", "over whole groups the 95% limit of the gain is at most 2.92 percentage points on the extended paired track and "
      "8.75 on the rebuilt broad track, and 3.44 and 10.97 with a Bonferroni correction over the six models",
      {"extended paired": ("2.92", "3.44"), "rebuilt broad": ("8.75", "10.97")}, _gi,
      "p2_paired.json, r2_broad.json: contrasts.slot_vs_none.models.*.group_interval[1] and family_wise.group_interval[1]")
check("4.2", "with the books of a group as clusters, the group limit on the extended paired track is 2.99", "2.99",
      _r(max(_sv[m]["coarse_clusters"]["largest_gain_pp"] for m in _SIXR), 2),
      "p2_paired.json: contrasts.slot_vs_none.models.*.coarse_clusters.largest_gain_pp")
_rule = all(abs(v["largest_gain_pp"] - round(100 * max(v["row_interval"][1], v["group_interval"][1],
                                                        v["group_cr1_interval"][1]), 3)) < 0.002
            and (v["group_interval_method"].startswith("CR1 with t") == (v["groups_nonzero"] < 20))
            for _t in (_p2, _r2) for v in _t["contrasts"]["slot_vs_none"]["models"].values())
check("3.7", "on the extended and rebuilt tracks each bound in Table 3 is the widest gain-side limit of the row-level, "
      "cluster-robust, and group intervals, and the group interval resamples whole groups 10,000 times or uses a t interval "
      "when fewer than 20 groups change", True, _rule,
      "p2_paired.json, r2_broad.json: contrasts.slot_vs_none.models.*.{largest_gain_pp, row_interval, group_interval, "
      "group_cr1_interval, group_interval_method, groups_nonzero}")
check("3.7, Table 3", "each stored bound of the extended and rebuilt tracks equals the widest of its three intervals, read unrounded", True,
      all(abs(round(_side(v, "gain"), 3) - v["largest_gain_pp"]) < 0.0011 and abs(round(_side(v, "loss"), 3) - v["largest_loss_pp"]) < 0.0011
          for _t in (_p2, _r2) for v in _t["contrasts"]["slot_vs_none"]["models"].values()),
      "p2_paired.json, r2_broad.json: contrasts.slot_vs_none.models.*.{largest_gain_pp, largest_loss_pp, row_interval, group_interval, "
      "group_cr1_interval}")

_hr = J(T + "human_ratings.json")
_hk, _ht, _h3, _hs = _hr["kit"], _hr["tracks"], _hr["third_rater"], _hr["statistics_raters"]
_cols = ("p2", "r2", "paired")
_cons = {t: _ht[t]["validity"]["consensus"] for t in _cols}
_vv = {t: _ht[t]["agreement"]["valid_vs_not"] for t in _cols}
_v3 = {t: _ht[t]["agreement"]["three_classes"] for t in _cols}


def _rescaled(u, e, key):
    return {"bound_pp": u, "no_signal": u / (1 - e), "reversed": u / (1 - 2 * e) if e < 0.5 else None,
            "worst_case": (u + 100 * e) / (1 - e)}[key]


def _top(t, rule, key):
    e = _ht[t]["validity"][rule]["e_rows"]["from_one_sided_upper"]
    return max(x for x in (_rescaled(v["bound_pp"], e, key) for v in _ht[t]["sensitivity_bounds"][rule].values())
               if x is not None)


def _share(v):
    return f"{v['count']}/{v['n']}"


_kit = (_hk["p2"]["groups"], _hk["p2"]["pool_groups"], _hk["audit"]["groups"])
check("3.8", "the rating kit holds every gold of the original paired and rebuilt broad tracks, 60 of the 480 extended groups "
      "with all their golds, and 11 groups of the first broad sample", (60, 480, 11), _kit,
      "human_ratings.json: kit.*.{groups, pool_groups, every_gold_of_the_rated_groups, every_group_of_the_pool}; "
      "pool_statistics.json: track_luas.grup",
      ok=_kit == (60, 480, 11) and _hk["audit"]["pool_groups"] == _tbroad["grup"]
      and all(v["every_gold_of_the_rated_groups"] for v in _hk.values())
      and _hk["paired"]["every_group_of_the_pool"] and _hk["r2"]["every_group_of_the_pool"])
check("6", "the raters read every gold of the original paired and rebuilt broad tracks but only samples of the other two",
      (True, True, False, False), tuple(_hk[t]["every_group_of_the_pool"] for t in ("paired", "r2", "p2", "audit")),
      "human_ratings.json: kit.{paired, r2, p2, audit}.every_group_of_the_pool")
_set = _hr["settlement"]
check("3.8", "where the first two non-statistics raters disagreed, the median of the three answers decided, and the "
      "statistics raters did not disagree on validity", (True, 0),
      (_set["settled_by_median_of_three"], _set["statistics_items_validity_differs"]),
      "human_ratings.json: settlement.{settled_by_median_of_three, statistics_items_validity_differs}")
check("4.6, 6, Table 6", "after settling disagreements the raters found no invalid gold on the two paired tracks and one on "
      "the rebuilt broad track", ("0/120", "1/55", "0/104"), tuple(_share(_cons[t]["invalid_golds"]) for t in _cols),
      "human_ratings.json: tracks.{p2, r2, paired}.validity.consensus.invalid_golds.{count, n}")
check("Table 6", "exact one-sided 95% upper limit of the share of invalid golds, in percent", ("2.5", "8.3", "2.8"),
      tuple(_r(100 * _cons[t]["invalid_golds"]["one_sided_95_upper"], 1) for t in _cols),
      "human_ratings.json: tracks.*.validity.consensus.invalid_golds.one_sided_95_upper")
check("Table 6", "rows touched, the share of rows that this upper limit of invalid golds would touch (e), in percent",
      ("4.9", "17.1", "5.6"), tuple(_r(100 * _cons[t]["e_rows"]["from_one_sided_upper"], 1) for t in _cols),
      "human_ratings.json: tracks.*.validity.consensus.e_rows.from_one_sided_upper")
_agreeing = tuple(round(_vv[t]["raw_agreement"]["estimate"] * _ht[t]["n_items"]) for t in _cols)
check("Table 6", "the two raters agree on valid against not", ("120/120", "54/55", "104/104"),
      tuple(f"{a}/{_ht[t]['n_items']}" for a, t in zip(_agreeing, _cols)),
      "human_ratings.json: tracks.*.{n_items, agreement.valid_vs_not.raw_agreement.estimate}",
      ok=tuple(f"{a}/{_ht[t]['n_items']}" for a, t in zip(_agreeing, _cols)) == ("120/120", "54/55", "104/104")
      and all(abs(a / _ht[t]["n_items"] - _vv[t]["raw_agreement"]["estimate"]) < 5e-5 for a, t in zip(_agreeing, _cols)))
check("Table 6", "Gwet's AC1 on valid against not", ("1.00", "0.98", "1.00"),
      tuple(_r(_vv[t]["gwet_ac1"]["estimate"], 2) for t in _cols), "human_ratings.json: tracks.*.agreement.valid_vs_not.gwet_ac1.estimate")
_kap = tuple("n/a" if _vv[t]["cohen_kappa"]["estimate"] is None else _r(_vv[t]["cohen_kappa"]["estimate"], 2) for t in _cols)
check("4.6, Table 6", "Cohen's kappa on valid against not, n/a exactly where neither rater answered not at all",
      ("n/a", "0.66", "n/a"), _kap, "human_ratings.json: tracks.*.{agreement.valid_vs_not.cohen_kappa.estimate, answer_counts}",
      ok=_kap == ("n/a", "0.66", "n/a") and all((_vv[t]["cohen_kappa"]["estimate"] is None)
                                                == (_ht[t]["answer_counts"]["rater_1"]["tidak"] == 0 == _ht[t]["answer_counts"]["rater_2"]["tidak"])
                                                for t in _cols))
_agreeing3 = tuple(round(_v3[t]["raw_agreement"]["estimate"] * _ht[t]["n_items"]) for t in _cols)
check("Table 6", "raw agreement over the 3 answers (full, partial, none), in percent, from the count of agreeing items, since "
      "the stored share has four decimals (52 of 55 is 94.545%, stored as 0.9455)", ("92.5", "94.5", "99.0"),
      tuple(_r(100 * a / _ht[t]["n_items"], 1) for a, t in zip(_agreeing3, _cols)),
      "human_ratings.json: tracks.*.{n_items, agreement.three_classes.raw_agreement.estimate}",
      ok=tuple(_r(100 * a / _ht[t]["n_items"], 1) for a, t in zip(_agreeing3, _cols)) == ("92.5", "94.5", "99.0")
      and all(abs(a / _ht[t]["n_items"] - _v3[t]["raw_agreement"]["estimate"]) < 5e-5 for a, t in zip(_agreeing3, _cols)))
check("Table 6", "Cohen's kappa over the 3 answers", ("0.29", "0.60", "0.00"),
      tuple(_r(_v3[t]["cohen_kappa"]["estimate"], 2) for t in _cols), "human_ratings.json: tracks.*.agreement.three_classes.cohen_kappa.estimate")
_tables = {"p2": _p2, "r2": _r2}
check("Table 6", "the largest slot gain over the six models, as each track reports it, in percentage points", ("4.40", "12.50", "13.71"),
      tuple(_r(_top(t, "consensus", "bound_pp"), 2) for t in _cols),
      "human_ratings.json: tracks.*.sensitivity_bounds.consensus.*.bound_pp; p2_paired.json, r2_broad.json: "
      "contrasts.slot_vs_none.summary.largest_gain_pp_max_over_models",
      ok=tuple(_r(_top(t, "consensus", "bound_pp"), 2) for t in _cols) == ("4.40", "12.50", "13.71")
      and all(_top(t, "consensus", "bound_pp") == _tables[t]["contrasts"]["slot_vs_none"]["summary"]["largest_gain_pp_max_over_models"]
              for t in _tables))
check("Table 6", "every stored rescaled bound equals the formula of 11.1 g applied to its stored bound and e", True,
      all(abs(round(_rescaled(v["bound_pp"], _ht[t]["validity"][rule]["e_rows"]["from_one_sided_upper"], k), 3) - v[k]) < 0.0011
          for t in _cols for rule in ("consensus", "all_raters") for v in _ht[t]["sensitivity_bounds"][rule].values()
          for k in ("no_signal", "reversed", "worst_case") if v[k] is not None),
      "human_ratings.json: tracks.*.sensitivity_bounds.*.*, validity.*.e_rows.from_one_sided_upper")
for _key, _label, _want in (("no_signal", "invalid rows carry no signal, d / (1 - e)", ("4.63", "15.08", "14.52")),
                            ("reversed", "invalid rows carry the reverse signal, d / (1 - 2e)", ("4.88", "19.01", "15.43")),
                            ("worst_case", "the worst case, (U + 100e) / (1 - e)", ("9.76", "35.75", "20.45"))):
    check("Table 6", f"the slot bound rescaled with e when {_label}, the largest of the six models, in percentage points", _want,
          tuple(_r(_top(t, "consensus", _key), 2) for t in _cols), f"human_ratings.json: tracks.*.sensitivity_bounds.consensus.*.{_key}")
_strict = {t: _ht[t]["validity"]["all_raters"]["invalid_golds"] for t in _cols}
check("Table 6", "invalid golds when every rater must accept a gold, outside the protocol", ("21/120", "6/55", "0/104"),
      tuple(_share(_strict[t]) for t in _cols), "human_ratings.json: tracks.*.validity.all_raters.invalid_golds.{count, n}")
for _key, _want in (("no_signal", ("7.66", "20.40", "14.52")), ("worst_case", ("81.76", "83.67", "20.45"))):
    check("Table 6", f"the slot bound rescaled ({_key}) when every rater must accept a gold, outside the protocol", _want,
          tuple(_r(_top(t, "all_raters", _key), 2) for t in _cols), f"human_ratings.json: tracks.*.sensitivity_bounds.all_raters.*.{_key}")
_n3 = _h3["answer_counts"]
check("4.6, 6", "the third rater was stricter, with more answers of no answer than either other non-statistics rater", True,
      _n3["N3"]["tidak"] > max(_n3["N1"]["tidak"], _n3["N2"]["tidak"]), "human_ratings.json: third_rater.answer_counts")
_va = _h3["validity_agreement"]
_kn3 = (_r(_va["N3_vs_N2"]["cohen_kappa"]["estimate"], 2), _r(_va["N3_vs_N1"]["cohen_kappa"]["estimate"], 2))
check("4.6", "on validity the third rater matched each of the others on about 80% of the 159 non-statistics items (kappa 0.28 "
      "and 0.25)", ("about 80%", 159, "0.28", "0.25"),
      (" and ".join(f"{100 * _va[k]['raw_agreement']['estimate']:.1f}%" for k in ("N3_vs_N2", "N3_vs_N1")), _h3["items"], *_kn3),
      "human_ratings.json: third_rater.{items, validity_agreement.N3_vs_N2, validity_agreement.N3_vs_N1}",
      ok=all(abs(100 * _va[k]["raw_agreement"]["estimate"] - 80) <= 0.5 for k in ("N3_vs_N2", "N3_vs_N1")) and _h3["items"] == 159
      and _kn3 == ("0.28", "0.25"))
_bt = _h3["by_track"]
check("4.6", "the third rater alone rejected 21 of the 105 extended golds among the non-statistics items and alone called five "
      "questions unclear", (21, 105, 5),
      (_bt["p2"]["rejected_by_N3_alone"], _bt["p2"]["items"], sum(v["questions_unclear_N3_alone"] for v in _bt.values())),
      "human_ratings.json: third_rater.by_track.*.{rejected_by_N3_alone, items, questions_unclear, questions_unclear_N3_alone}",
      ok=(_bt["p2"]["rejected_by_N3_alone"], _bt["p2"]["items"], sum(v["questions_unclear_N3_alone"] for v in _bt.values())) == (21, 105, 5)
      and all(v["questions_unclear"]["N1"] == v["questions_unclear"]["N2"] == 0 for v in _bt.values()))
_mined = _ht["audit"]["validity"]["consensus"]["by_role"]["mined_partner_gold"]
_partner = _ht["r2"]["validity"]["consensus"]["by_role"]["partner_gold"]
check("4.6", "of 18 mined partners rated, 9 were valid, against 28 of 29 rebuilt partners", (18, 9, 28, 29),
      (_mined["n"], _mined["n"] - _mined["count"], _partner["n"] - _partner["count"], _partner["n"]),
      "human_ratings.json: tracks.{audit, r2}.validity.consensus.by_role.{mined_partner_gold, partner_gold}")
_jf = _ht["r2"]["judge_vs_raters"]["consensus"]["judge_full_answers_rated_valid"]
check("4.6", "the raters accepted 54 of the 55 rebuilt golds the judge rated a full answer", (54, 55), (_jf["count"], _jf["n"]),
      "human_ratings.json: tracks.r2.judge_vs_raters.consensus.judge_full_answers_rated_valid")
_vg = {}
for _rule in ("consensus", "all_raters"):
    _fam = [_ht[t]["recomputed_contrasts"][_rule]["subsets"]["groups_all_golds_valid"] for t in _cols]
    _vg[_rule] = (tuple(len(f["slot_vs_none"]["summary"]["significant_after_holm_mcnemar"]) for f in _fam)
                  + tuple(len(f["both_vs_slot"]["summary"]["significant_after_holm_mcnemar"]) for f in _fam)
                  + (all(len(f[c]["models"]) == 6 for f in _fam for c in ("slot_vs_none", "both_vs_slot")),))
check("4.6", "under both rules, in groups with all golds valid, the slot is significant on no model of any track, and the label on "
      "five, three, and five models (Holm over McNemar)", {r: (0, 0, 0, 5, 3, 5, True) for r in ("consensus", "all_raters")}, _vg,
      "human_ratings.json: tracks.*.recomputed_contrasts.*.subsets.groups_all_golds_valid.{slot_vs_none, both_vs_slot}.summary")
_rb = _hr["rated_books"]
check("6", "about 60% of the non-statistics raters' passages come from business books (accounting, management, marketing, and "
      "business analytics)", "about 60%", f"about {_r(100 * _rb['business_share'], 0)}% ({_rb['business_items']} of "
      f"{_rb['non_statistics_items']})", "human_ratings.json: rated_books", ok=_r(100 * _rb["business_share"], 0) == "60")
check("6", "the statistics raters rejected no passage, even two mined partners the judge called non-answers", (0, 0, 2, 2),
      (_hs["rated_not_at_all"]["S1"], _hs["rated_not_at_all"]["S2"], _hs["mined_partners_judge_not_at_all"],
       _hs["of_those_valid_for_both_raters"]), "human_ratings.json: statistics_raters")


# Report
n_bad = sum(1 for r in rows if r[5] != "match")
with OUT.open("w", encoding="utf-8") as fh:
    fh.write("# The paper's numbers checked against this package\n\n")
    fh.write(f"Package: `{ROOT}`. Checks: {len(rows)}. Match: {len(rows) - n_bad}. Differ: {n_bad}.\n\n")
    fh.write("Each row names the file and key a claim rests on. Derived values (ranges, differences, Holm corrections, sign tests) are recomputed in this script from the keys named.\n\n")
    fh.write("| # | Section | Claim in the paper | Paper value | Package value | Source in this package | Status |\n| --- | --- | --- | --- | --- | --- | --- |\n")
    for i, (b, k, a, p, s, st) in enumerate(rows, 1):
        fh.write(f"| {i} | {b} | {k} | {a.replace('|', '/')} | {p.replace('|', '/')} | `{s}` | {'**' + st + '**' if st != 'match' else st} |\n")
print(f"checks {len(rows)}, match {len(rows) - n_bad}, differ {n_bad}; report: {OUT}")
for r in rows:
    if r[5] != "match":
        print("  DIFFER:", r[0], "|", r[1], "| paper", r[2], "| package", r[3])
sys.exit(1 if n_bad else 0)
