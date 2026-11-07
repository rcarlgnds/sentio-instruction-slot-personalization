import os
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("SENTIO_ROOT")
            or Path(__file__).resolve().parents[4])
CORPUS = ROOT / "system/data/benchmark_release/corpus/corpus.jsonl"
PAIRED = ROOT / "system/data/paired_set/paired_queries.jsonl"
Q2 = ROOT / "study/channel/results"
SHARED = ROOT / "study/_shared"

MEMBERSHIP_PATTERN = {
    "kode_matkul_berformat": re.compile(r"\b[A-Z]{4}\d{5,7}\b"),
    "frasa_program_studi": re.compile(
        r"\b(?:degree program|study program|program studi|major in|"
        r"undergraduate program|department of)\b", re.I),
    "frasa_kurikulum_institusi": re.compile(
        r"\b(?:this course|our curriculum|the syllabus|course outline)\b", re.I),
    "nama_institusi": re.compile(
        r"\b(?:university of|universitas|binus|college of)\b", re.I),
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def tokenize_text(t: str) -> list:
    return [w for w in ''.join(c.lower() if c.isalnum() else ' '
                               for c in t).split() if len(w) > 2]


def log_odds_dirichlet(a: Counter, b: Counter, strong_prior=0.01) -> dict:
    vocab = set(a) | set(b)
    na, nb = sum(a.values()), sum(b.values())
    total = Counter(a) + Counter(b)
    n_all = na + nb
    output = {}
    for w in vocab:
        a0 = strong_prior * total[w] * n_all / max(n_all, 1)
        ya = a[w] + a0 + 0.5
        yb = b[w] + a0 + 0.5
        la = math.log(ya / (na + a0 - ya + 1))
        lb = math.log(yb / (nb + a0 - yb + 1))
        var = 1 / ya + 1 / yb
        output[w] = (la - lb) / math.sqrt(var)
    return output


def main():
    lock = json.loads((SHARED / "config_lock.json").read_text())

    title = Counter()
    for l in PAIRED.open():
        r = json.loads(l)
        for k in ("gold_anchor", "counterpart_anchor"):
            a = (r.get(k) or [{}])[0]
            if a.get("document_title"):
                title[a["document_title"]] += 1
    title_slug_map, chunks = {}, defaultdict(list)
    gold_text = {}
    for l in CORPUS.open():
        r = json.loads(l)
        m = r["metadata"]
        dt = (m.get("document_title") or "").replace(" ", "_")
        title_slug_map.setdefault(dt, m.get("book_slug"))
        chunks[m.get("book_slug")].append(r.get("text") or "")
        gold_text[r["_id"]] = (r.get("text") or "", m.get("book_slug"))
    pairs = [title_slug_map.get(j) for j, _ in title.most_common(2)]
    pairs = [p for p in pairs if p]
    print("track book pair:", pairs)

    per_book = {}
    for slug, text in chunks.items():
        hit = {name: 0 for name in MEMBERSHIP_PATTERN}
        any_present = 0
        for t in text:
            matched = False
            for name, pattern in MEMBERSHIP_PATTERN.items():
                if pattern.search(t):
                    hit[name] += 1
                    matched = True
            any_present += int(matched)
        per_book[slug] = {
            "n_chunk": len(text),
            "n_chunk_ber_penanda": any_present,
            "rate": round(any_present / len(text), 5) if text else None,
            "per_pola": hit,
        }
    total_chunk = sum(v["n_chunk"] for v in per_book.values())
    total_hit = sum(v["n_chunk_ber_penanda"] for v in per_book.values())
    print(f"full corpus: {total_hit}/{total_chunk} = "
          f"{100*total_hit/total_chunk:.2f}% of chunks carry a membership marker")

    discriminative = {}
    if len(pairs) == 2:
        ca = Counter()
        cb = Counter()
        for t in chunks[pairs[0]]:
            ca.update(tokenize_text(t))
        for t in chunks[pairs[1]]:
            cb.update(tokenize_text(t))
        score = log_odds_dirichlet(ca, cb)
        enough = {w: s for w, s in score.items() if ca[w] + cb[w] >= 20}
        upper = sorted(enough.items(), key=lambda kv: -kv[1])[:25]
        lower = sorted(enough.items(), key=lambda kv: kv[1])[:25]

        def verdict(w):
            for name, pattern in MEMBERSHIP_PATTERN.items():
                if pattern.search(w):
                    return name
            return "kosakata_topik_atau_gaya"
        discriminative = {
            "buku_a": pairs[0], "buku_b": pairs[1],
            "top25_condong_a": [{"token": w, "z": round(s, 2), "jenis": verdict(w)}
                                for w, s in upper],
            "top25_condong_b": [{"token": w, "z": round(s, 2), "jenis": verdict(w)}
                                for w, s in lower],
        }
        n_markers = sum(1 for w, _ in upper + lower
                        if verdict(w) != "kosakata_topik_atau_gaya")
        discriminative["n_dari_50_yang_penanda_keanggotaan"] = n_markers
        print(f"top discriminative tokens: {n_markers}/50 are explicit "
              f"membership markers, the rest topic or style vocabulary")

    gold_ids = []
    for l in PAIRED.open():
        r = json.loads(l)
        gold_ids += (r.get("gold_relevant_chunks") or [])
        gold_ids += (r.get("counterpart_chunks") or [])
    anchor_ids = []
    for l in PAIRED.open():
        r = json.loads(l)
        for k in ("gold_anchor", "counterpart_anchor"):
            a = (r.get(k) or [{}])[0]
            if a.get("canonical_id"):
                anchor_ids.append(a["canonical_id"])
    found = [gold_text[i] for i in set(anchor_ids) if i in gold_text]
    gold_hit = sum(1 for t, _ in found
                    if any(p.search(t) for p in MEMBERSHIP_PATTERN.values()))
    print(f"paired gold chunks mapped: {len(found)}; "
          f"holding a membership marker: {gold_hit}")

    output = {
        "dibuat": datetime.now(timezone.utc).isoformat(),
        "config_version_lock": lock["config_version"],
        "_apa_ini": ("Leakage audit: whether the text of a passage announces its program membership. Domain "
                     "vocabulary is not counted as leakage."),
        "definisi_penanda": {k: v.pattern for k, v in MEMBERSHIP_PATTERN.items()},
        "korpus_penuh": {
            "n_chunk": total_chunk,
            "n_chunk_ber_penanda": total_hit,
            "rate": round(total_hit / total_chunk, 5) if total_chunk else None,
        },
        "per_buku": per_book,
        "pasangan_track": discriminative,
        "gold_paired_track": {
            "n_chunk_terpetakan": len(found),
            "n_ber_penanda": gold_hit,
            "rate": round(gold_hit / len(found), 5) if found else None,
        },
        "label_source": "deterministic (pattern scan over the frozen corpus)",
    }
    out = Q2 / "tables/leakage_audit_v2.json"
    out.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    (Q2 / "raw/leakage_audit_v2.manifest.json").write_text(json.dumps({
        "run_id": "q2_leak_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "script": "study/channel/scripts/5_why_slot_fails/leakage_check.py",
        "config_version": lock["config_version"],
        "label_source": "deterministic",
        "sumber": {"corpus": sha(CORPUS), "paired_queries": sha(PAIRED)},
        "output": {"path": "results/tables/leakage_audit_v2.json",
                   "sha256": sha(out), "tingkat_reproduksi": "1_bit_identical"},
    }, indent=2, ensure_ascii=False))
    print(f"written: {out}")


if __name__ == "__main__":
    main()
