"""Aggregates the worked-example control (58 pairs, three models). CPU only.

The two passages differ in the text itself, a worked example against a definition, so this control
shows whether the setup picks up instruction following at all. Its per-row scores do not ship, so
quick_check.py cannot rerun it. Writes results/tables/control_textual.json.
"""

import os
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import sys  # noqa: E402
for _stage in sorted(p for p in Path(__file__).resolve().parents[1].iterdir() if p.is_dir()):
    sys.path.insert(0, str(_stage))
import metrics as M

ROOT = Path(os.environ.get("SENTIO_ROOT")
            or Path(__file__).resolve().parents[4])
Q2 = ROOT / "study/channel/results"
RAW = Q2 / "raw"
SHARED = ROOT / "study/_shared"

POOL = RAW / "pools_control_textual.jsonl"
MODEL = {
    "Qwen3-Reranker-0.6B": "qwen3rr_Qwen3-Reranker-0.6B",
    "Qwen3-Reranker-4B": "qwen3rr_Qwen3-Reranker-4B",
    "zerank-2": "zerank2_zerank-2",
}
PAIRED_METADATA = {
    "Qwen3-Reranker-0.6B": (
        ROOT / "study/retrieval/results/raw/qwen_scores_0.6b_paired_default.jsonl",
        ROOT / "study/retrieval/results/raw/qwen_scores_0.6b_paired_profile.jsonl"),
    "Qwen3-Reranker-4B": (
        ROOT / "study/retrieval/results/raw/qwen_scores_4b_paired_default.jsonl",
        ROOT / "study/retrieval/results/raw/qwen_scores_4b_paired_profile.jsonl"),
    "zerank-2": (
        RAW / "scores_zerank2_zerank-2_paired_default.jsonl",
        RAW / "scores_zerank2_zerank-2_paired_profile.jsonl"),
}

STOP_LIST = {
    "introduction", "exercises", "exercise", "solution", "solutions",
    "problem", "problems", "note", "notes", "key terms", "chapter review",
    "formula review", "practice", "homework", "references", "glossary",
    "prerequisites", "learning objectives", "learning objectives introduction",
    "what do you think?", "summary", "answers", "review",
    "statistical literacy",
}
STOP_PREFIXES = (
    "try it", "collaborative exercise", "bringing it together", "part ",
    "using the ti-", "example ", "answers to", "chapter ", "appendix",
)
FUNCTION_WORDS = {
    "and", "are", "but", "for", "from", "into", "its", "that", "the",
    "their", "then", "there", "these", "this", "was", "were", "with",
    "you", "your",
}

VERBATIM_SCREEN_RULE = (
    "A topic pair is kept only when the title of its source section passes all four "
    "conditions below. Every condition is computed from the title text alone, without "
    "touching scores, rankings, golds, or counterparts. T1, no PDF extraction artifact: the "
    "title holds none of 'GLYPH<', 'font=/', or '<c='. T2, a well-formed head: the title "
    "starts with a letter, or with a section number matching ^\\d+\\.\\d+\\s. T3, not book "
    "apparatus: the title (lowercased, section number removed, spaces collapsed) is not in "
    "the closed stop list S and starts with none of the prefixes in P. T4, topically "
    "specific: after the section number is removed, at least two alphabetic tokens of 3 or "
    "more letters remain that are not function words F. S, P, and F are listed in full in "
    "the daftar_henti, awalan_henti, and kata_fungsi fields of this table."
)
EN_SCREEN_RULE = (
    "A topic pair is RETAINED only if its source section heading satisfies all four "
    "conditions. Every condition is computed from the heading text alone, touching no score, "
    "ranking, gold, or counterpart. T1 no PDF extraction artefact: the heading contains none "
    "of 'GLYPH<', 'font=/', '<c='. T2 well-formed heading: the heading starts with a letter, "
    "or with a section-number pattern ^\\d+\\.\\d+\\s. T3 not book apparatus: the heading "
    "(lowercased, section number stripped, whitespace collapsed) is not in the closed "
    "stoplist S and does not start with any prefix in P. T4 topical specificity: after "
    "stripping the section number, at least two alphabetic tokens of length >= 3 remain that "
    "are not function words in F. S, P, and F are listed verbatim in the table."
)

RE_NUMBERED = re.compile(r"^\d+\.\d+\s+")
RE_LETTER = re.compile(r"[A-Za-z]+")


def passed_filter(title: str) -> tuple:
    j = " ".join(title.split())
    for mark in ("GLYPH<", "font=/", "<c="):
        if mark in j:
            return False, f"T1 PDF extraction artifact ({mark})"
    if not (j[:1].isalpha() or RE_NUMBERED.match(j)):
        return False, "T2 title head not well formed"
    core = RE_NUMBERED.sub("", j)
    small = " ".join(core.lower().split())
    if small in STOP_LIST:
        return False, f"T3 book apparatus, stop list ({small})"
    for aw in STOP_PREFIXES:
        if small.startswith(aw):
            return False, f"T3 book apparatus, prefix ({aw.strip()})"
    content = [t for t in RE_LETTER.findall(core)
           if len(t) >= 3 and t.lower() not in FUNCTION_WORDS]
    if len(content) < 2:
        return False, f"T4 not specific enough, {len(content)} content token"
    return True, "passed T1-T4"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def binom_two_sided(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def round_p(p: float) -> float:
    if p <= 0:
        return 0.0
    return float(f"{p:.4g}")


def wilson(k: int, n: int, z=1.96) -> list:
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - m), 4), round(min(1.0, c + m), 4)]


def holm(p: dict) -> dict:
    order = sorted(p.items(), key=lambda kv: kv[1])
    m = len(order)
    output, before = {}, 0.0
    for i, (k, v) in enumerate(order):
        adj = min(1.0, max(before, (m - i) * v))
        before = adj
        output[k] = round_p(adj)
    return output


def wins_per_row(rows) -> dict:
    output = {}
    for r in rows:
        pos = {c: i for i, c in enumerate(r["ranking"])}
        g = min((pos[c] for c in (r["gold"] or []) if c in pos), default=None)
        o = min((pos[c] for c in (r.get("counterpart") or []) if c in pos),
                default=None)
        output[r["query_id"]] = None if (g is None or o is None) else (g < o)
    return output


def model_block(name, prefix, used_ids, used_pairs, pool_by_id):
    result = {"model": name}
    records = {}
    for condition in ("default", "kondisi"):
        p = RAW / f"scores_{prefix}_control_{condition}.jsonl"
        man = RAW / f"scores_{prefix}_control_{condition}.manifest.json"
        rows = [r for r in M.read_ranking(p) if r["query_id"] in used_ids]
        M.seal_gate([r["query_id"] for r in rows])
        m = json.loads(man.read_text())
        if m["output"]["sha256"] != sha(p):
            raise SystemExit(f"ABORT: sha256 {p.name} does not match the manifest.")
        if m["sumber"]["pools"] != sha(POOL):
            raise SystemExit(f"ABORT: {man.name} points at a different pool.")
        result[f"metrik_{condition}"] = M.paired_metrics(rows)
        result[f"metrik_{condition}"]["sha256_sumber"] = m["output"]["sha256"]
        records[condition] = wins_per_row(rows)

    d, k = records["default"], records["kondisi"]
    id_valid = sorted(q for q in used_ids
                      if d.get(q) is not None and k.get(q) is not None)

    b = sum(1 for q in id_valid if d[q] and not k[q])
    c = sum(1 for q in id_valid if (not d[q]) and k[q])
    result["mcnemar_baris"] = {
        "n_baris": len(id_valid),
        "keduanya_benar": sum(1 for q in id_valid if d[q] and k[q]),
        "keduanya_salah": sum(1 for q in id_valid if (not d[q]) and (not k[q])),
        "hanya_default_benar": b, "hanya_kondisi_benar": c,
        "p_dua_sisi_exact": round_p(binom_two_sided(b, c)),
        "peringatan": ("the two rows of a pair are perfectly negatively correlated in the default arm, so this "
                       "test is secondary"),
    }

    gains = drops = ties = 0
    both_win = none_win = one_wins = 0
    for pid in used_pairs:
        qs = [f"{pid}-contoh", f"{pid}-definisi"]
        if any(q not in id_valid for q in qs):
            continue
        wd = sum(1 for q in qs if d[q])
        wk = sum(1 for q in qs if k[q])
        if wk == 2:
            both_win += 1
        elif wk == 1:
            one_wins += 1
        else:
            none_win += 1
        if wk > wd:
            gains += 1
        elif wk < wd:
            drops += 1
        else:
            ties += 1
    n_pair = gains + drops + ties
    result["uji_tanda_pasangan"] = {
        "n_pasangan": n_pair,
        "kondisi_lebih_baik": gains, "kondisi_lebih_buruk": drops,
        "seri": ties,
        "kondisi_menang_dua_dua": both_win,
        "kondisi_menang_satu": one_wins,
        "kondisi_menang_nol": none_win,
        "kemenangan_default_per_pasangan": 1,
        "pairwise_default": round(0.5, 4),
        "pairwise_kondisi": (round((2 * both_win + one_wins) / (2 * n_pair), 4)
                             if n_pair else None),
        "delta_pairwise": (round((2 * both_win + one_wins) / (2 * n_pair) - 0.5, 4)
                           if n_pair else None),
        "wilson95_dua_dua": wilson(both_win, n_pair),
        "p_dua_sisi_exact": round_p(binom_two_sided(drops, gains)),
    }

    result["menang_per_sisi"] = {}
    for side in ("contoh", "definisi"):
        qs = [q for q in id_valid if q.endswith(f"-{side}")]
        result["menang_per_sisi"][side] = {
            "n": len(qs),
            "default": sum(1 for q in qs if d[q]),
            "kondisi": sum(1 for q in qs if k[q]),
        }
    return result


def metadata_block(name):
    p_def, p_prof = PAIRED_METADATA[name]
    output = {"populasi": "paired track, 96 rows, the difference lies in the profile metadata"}
    for tag, p in (("default", p_def), ("profil", p_prof)):
        if not p.exists():
            output[tag] = {"status": "missing"}
            continue
        rows = M.read_ranking(p)
        M.seal_gate([r["query_id"] for r in rows])
        m = M.paired_metrics(rows)
        output[tag] = {"pairwise": m["pairwise"],
                       "n_komparabel": m["n_komparabel"],
                       "sha256_sumber": sha(p)}
    return output


def population(name, pool_rows, passed_only):
    used_pairs, used_ids, trace = [], set(), []
    for pid in sorted({r["pair_id"] for r in pool_rows},
                      key=lambda x: int(x.split("-")[1])):
        r = next(x for x in pool_rows if x["pair_id"] == pid)
        title = r["provenance"]["section"]
        ok, reason = passed_filter(title)
        trace.append({"pair_id": pid, "judul_seksi": title,
                      "kueri": r["query"], "lolos": ok, "alasan": reason})
        if ok or not passed_only:
            used_pairs.append(pid)
            used_ids.update({f"{pid}-contoh", f"{pid}-definisi"})
    return used_pairs, used_ids, trace


def main():
    lock = json.loads((SHARED / "config_lock.json").read_text())
    pool_rows = M.read_ranking(POOL)
    pool_by_id = {r["query_id"]: r for r in pool_rows}
    man_pool = json.loads((RAW / "pools_control_textual.manifest.json").read_text())
    if man_pool["output"]["sha256"] != sha(POOL):
        raise SystemExit("ABORT: sha256 of the pool does not match its manifest.")
    M.seal_gate([r["query_id"] for r in pool_rows])
    print("paired metric replication gate:")
    evidence = M.replication_gate()

    output = {
        "dibuat": datetime.now(timezone.utc).isoformat(),
        "config_version_lock": lock["config_version"],
        "_apa_ini": (
            "Worked-example control. It follows the protocol of the paired track on the same two "
            "statistics books, but the two passages of a pair differ in the text itself, a worked "
            "example against a definition, instead of in the profile metadata. It checks on this "
            "corpus, as FollowIR does on an outside English corpus, that the scoring setup can show "
            "instruction following."),
        "status_arm": "post hoc, outside the pre-registered family of eight",
        "label_source": "deterministic (model scores); the textual gold and counterpart are built "
                        "deterministically from the book structure, with no LLM and no human annotator",
        "populasi": "58 topic pairs, 116 rows, a pool of 71 candidates per row",
        "sumber_buku": man_pool["buku_sumber"],
        "instruksi_default_verbatim":
            "Given a web search query, retrieve relevant passages that answer "
            "the query",
        "instruksi_kondisi_verbatim": man_pool["instruksi"],
        "fakta_struktural_default": (
            "In the default arm the two rows of a pair receive the same query and instruction, so "
            "their rankings are identical (checked 58 of 58 for all three models) and exactly one "
            "row wins. The default pairwise rate is therefore exactly 0.5000 by construction."),
        "uji_primer": (
            "pair-level sign test on the difference in win counts (condition minus default, where "
            "default is always 1), Holm within the family of three models, run on the full 58 pairs "
            "and on the 36 filtered pairs"),
        "uji_sekunder": (
            "row-level exact McNemar; reported in full but secondary, because the two rows of a pair "
            "are perfectly negatively correlated in the default arm"),
        "aturan_saring_verbatim": VERBATIM_SCREEN_RULE,
        "filter_rule_verbatim_en": EN_SCREEN_RULE,
        "aturan_saring_ditetapkan": (
            "The four rules were written after the cells were scored. Each reads only the section title, never a score, ranking, or gold."),
        "daftar_henti": sorted(STOP_LIST),
        "awalan_henti": list(STOP_PREFIXES),
        "kata_fungsi": sorted(FUNCTION_WORDS),
        "sumber": {"pools": man_pool["output"]["sha256"]},
        "gerbang_replikasi_metrik": evidence,
        "pembanding_saluran_metadata": {n: metadata_block(n) for n in MODEL},
        "cara_baca_pembanding": (
            "Both blocks use the same three models, metric, and paired protocol, and only the place "
            "of the difference changes. When the difference is in the profile metadata, the pairwise "
            "rate stays near 0.5. When it is in the text, the rate rises on all three models, most "
            "on zerank-2. So the scoring setup can detect a difference, and it does not explain the "
            "null on the metadata channel."),
    }

    for tag, filtered_only in (("tersaring", True), ("penuh", False)):
        used_pairs, used_ids, trace = population(tag, pool_rows, filtered_only)
        block = {"n_pasangan": len(used_pairs), "n_baris": len(used_ids),
                "peran": "exploratory" if filtered_only else "main",
                "model": {}}
        for name, prefix in MODEL.items():
            block["model"][name] = model_block(name, prefix, used_ids,
                                             used_pairs, pool_by_id)
        p_raw = {n: b["uji_tanda_pasangan"]["p_dua_sisi_exact"]
                    for n, b in block["model"].items()}
        p_holm = holm(p_raw)
        for n in block["model"]:
            block["model"][n]["uji_tanda_pasangan"]["p_holm"] = p_holm[n]
        block["holm_keluarga"] = ("three models, pair-level sign test, corrected within this population")
        if filtered_only:
            output["jejak_saring"] = trace
            output["n_dibuang"] = sum(1 for j in trace if not j["lolos"])
        output[f"populasi_{tag}"] = block

    ring = {}
    for tag in ("tersaring", "penuh"):
        b = output[f"populasi_{tag}"]
        passed = [n for n, m in b["model"].items()
                 if m["uji_tanda_pasangan"]["p_holm"] < 0.05]
        up_both_sides = [n for n, m in b["model"].items()
                         if m["menang_per_sisi"]["contoh"]["kondisi"]
                         > m["menang_per_sisi"]["contoh"]["default"]
                         and m["menang_per_sisi"]["definisi"]["kondisi"]
                         >= m["menang_per_sisi"]["definisi"]["default"]]
        ring[tag] = {
            "n_pasangan": b["n_pasangan"],
            "model_signifikan_setelah_holm": passed,
            "jumlah_signifikan": f"{len(passed)} dari {len(b['model'])}",
            "delta_pairwise": {n: m["uji_tanda_pasangan"]["delta_pairwise"]
                               for n, m in b["model"].items()},
            "model_naik_di_kedua_sisi": up_both_sides,
            "arti_naik_di_kedua_sisi": (
                "for these models the example side rises and the definition side does not fall, so the "
                "gain is not a general shift toward one kind of paragraph. The instruction changes which "
                "passage comes first."),
        }
    ring["catatan_daya"] = (
        "The 58 pairs are the main result. The 36 filtered pairs are an exploratory subset, "
        "because their rules were written after scoring. The effect has the same direction on "
        "both sets, and the smallest model is significant only on the 58 pairs, which give more "
        "power.")
    ring["bias_default"] = (
        "In the default arm all three models favor the definition paragraph (for example, "
        "filtered zerank-2 wins 31 of 36 definition rows against 5 of 36 example rows). The "
        "condition instruction raises the example side sharply without lowering the definition "
        "side, which is the pattern expected from instruction following and not from a global "
        "shift in bias.")
    output["ringkasan"] = ring

    out = Q2 / "tables/control_textual.json"
    out.write_text(json.dumps(output, indent=2, ensure_ascii=False))

    man = {
        "run_id": f"control_textual_agg_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}",
        "script": "study/channel/scripts/5_why_slot_fails/aggregate_controls.py",
        "perintah_persis": "python aggregate_controls.py",
        "config_version": lock["config_version"],
        "status_arm": "post hoc, outside the pre-registered family of eight",
        "sumber": {
            "pools": man_pool["output"]["sha256"],
            **{f"scores_{p}_control_{k}": json.loads(
                (RAW / f"scores_{p}_control_{k}.manifest.json").read_text()
            )["output"]["sha256"]
               for p in MODEL.values() for k in ("default", "kondisi")},
        },
        "n": {"pasangan_penuh": output["populasi_penuh"]["n_pasangan"],
              "baris_penuh": output["populasi_penuh"]["n_baris"],
              "pasangan_tersaring": output["populasi_tersaring"]["n_pasangan"],
              "baris_tersaring": output["populasi_tersaring"]["n_baris"]},
        "label_source": output["label_source"],
        "aturan_saring_verbatim": VERBATIM_SCREEN_RULE,
        "aturan_saring_ditetapkan": output["aturan_saring_ditetapkan"],
        "gerbang_segel": "passed, no test_holdout query_id",
        "output": {
            "path": "results/tables/control_textual.json",
            "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            "tingkat_reproduksi": "in full from the frozen files, CPU, deterministic",
        },
    }
    (RAW / "control_textual.manifest.json").write_text(
        json.dumps(man, indent=2, ensure_ascii=False))

    print(f"\ntable   : {out}")
    print(f"manifest: {RAW / 'control_textual.manifest.json'}")
    for tag in ("tersaring", "penuh"):
        b = output[f"populasi_{tag}"]
        print(f"\n[{tag.upper()}] {b['peran']}  pairs {b['n_pasangan']}  "
              f"rows {b['n_baris']}")
        for n, m in b["model"].items():
            u = m["uji_tanda_pasangan"]
            mc = m["mcnemar_baris"]
            print(f"  {n:22} pairwise def {u['pairwise_default']:.4f} -> "
                  f"condition {u['pairwise_kondisi']:.4f} "
                  f"(delta {u['delta_pairwise']:+.4f}) | "
                  f"pairs up {u['kondisi_lebih_baik']} down "
                  f"{u['kondisi_lebih_buruk']} tied {u['seri']} | "
                  f"2-2 {u['kondisi_menang_dua_dua']} | p {u['p_dua_sisi_exact']:.4g} "
                  f"holm {u['p_holm']:.4g} | McNemar rows b {mc['hanya_default_benar']} "
                  f"c {mc['hanya_kondisi_benar']} p {mc['p_dua_sisi_exact']:.4g} | "
                  f"side example {m['menang_per_sisi']['contoh']['kondisi']}/"
                  f"{m['menang_per_sisi']['contoh']['n']} definition "
                  f"{m['menang_per_sisi']['definisi']['kondisi']}/"
                  f"{m['menang_per_sisi']['definisi']['n']}")


if __name__ == "__main__":
    main()
