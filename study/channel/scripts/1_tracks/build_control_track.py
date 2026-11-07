import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("SENTIO_ROOT")
            or Path(__file__).resolve().parents[4])
AI = ROOT / "system"
CORPUS = ROOT / "system/data/benchmark_release/corpus/corpus.jsonl"
Q2 = ROOT / "study/channel/results"
SHARED = ROOT / "study/_shared"
SEED = 42
POOL_SIZE = 71

INSTRUCTIONS = {
    "contoh": ("Retrieve a passage that works through a concrete numerical "
               "example with actual numbers and a step-by-step solution"),
    "definisi": ("Retrieve a passage that states the conceptual definition and "
                 "explanation, without numerical worked examples"),
}

RE_DIGIT = re.compile(r"\d")
RE_EXAMPLE_MARKER = re.compile(
    r"\b(example|solution|suppose|calculate|compute|step \d)\b", re.I)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def digit_ratio(t: str) -> float:
    return sum(1 for c in t if c.isdigit()) / max(len(t), 1)


RE_SENTENCE = re.compile(r"[.!?]\s")
RE_GLYPH = re.compile(r"GLYPH<|font=/|\ufffd")


def is_clean_prose(t: str) -> bool:
    if RE_GLYPH.search(t):
        return False
    alpha_words = [w for w in re.findall(r"[A-Za-z]+", t) if len(w) > 2]
    if len(alpha_words) < 60:
        return False
    if len(RE_SENTENCE.findall(t)) < 3:
        return False
    if sum(c.isalpha() or c.isspace() for c in t) / max(len(t), 1) < 0.72:
        return False
    return True


def classify(t: str) -> str:
    if not is_clean_prose(t):
        return "ambigu"
    has_marker = bool(RE_EXAMPLE_MARKER.search(t))
    r = digit_ratio(t)
    if has_marker and 0.008 <= r <= 0.06:
        return "contoh"
    if not has_marker and r <= 0.004:
        return "definisi"
    return "ambigu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-topik", type=int, default=80)
    ap.add_argument("--min-free-mib", type=int, default=3500)
    args = ap.parse_args()

    free = int(subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True).split()[0])
    print(f"free VRAM {free} MiB")
    if free < args.min_free_mib:
        raise SystemExit("ABORT VRAM")

    lock = json.loads((SHARED / "config_lock.json").read_text())

    title = defaultdict(int)
    for l in (ROOT / "system/data/paired_set/paired_queries.jsonl").open():
        r = json.loads(l)
        for k in ("gold_anchor", "counterpart_anchor"):
            a = (r.get(k) or [{}])[0]
            if a.get("document_title"):
                title[a["document_title"]] += 1
    mapping = {}
    corpus = []
    for l in CORPUS.open():
        r = json.loads(l)
        m = r["metadata"]
        dt = (m.get("document_title") or "").replace(" ", "_")
        mapping.setdefault(dt, m.get("book_slug"))
        corpus.append((r["_id"], r.get("text") or "", m))
    book = [mapping[j] for j, _ in sorted(title.items(), key=lambda kv: -kv[1])[:2]
            if j in mapping]
    print("source books:", book)

    candidates = [(i, t, m) for i, t, m in corpus
                if m.get("book_slug") in book and 300 <= len(t) <= 3000]
    print(f"eligible chunks: {len(candidates)}")

    per_section = defaultdict(lambda: {"contoh": [], "definisi": []})
    for i, t, m in candidates:
        k = classify(t)
        if k == "ambigu":
            continue
        key = (m.get("book_slug"), (m.get("section_path") or "").split(">")[0].strip())
        per_section[key][k].append((i, t, m))
    eligible = [(k, v) for k, v in per_section.items() if v["contoh"] and v["definisi"]]
    print(f"sections with both kinds: {len(eligible)}")

    rng = random.Random(SEED)
    rng.shuffle(eligible)
    chosen = eligible[:args.n_topik]
    if len(chosen) < 20:
        raise SystemExit(f"ABORT: only {len(chosen)} eligible topics, too thin")

    sys.path.insert(0, str(AI))
    os.chdir(AI)
    import numpy as np
    from app.ingestion.embedder import HybridEmbedder
    emb = HybridEmbedder()
    emb.load()

    all_text = [t for _, t, _ in candidates]
    all_ids = [i for i, _, _ in candidates]
    print("embedding the candidate corpus ...")
    vecs = []
    B = 64
    for i in range(0, len(all_text), B):
        vecs.extend(emb.embed_text_dense(all_text[i:i + B]))
        if (i // B) % 30 == 0:
            print(f"  {i}/{len(all_text)}", flush=True)
    X = np.asarray(vecs, dtype=np.float32)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    rows = []
    for n, ((slug, section), v) in enumerate(chosen):
        c_example = v["contoh"][0]
        c_def = v["definisi"][0]
        topic = section or (c_def[2].get("document_title") or "")
        query = f"Explain {topic}".strip()
        qv = np.asarray(emb.embed_text_dense([query])[0], dtype=np.float32)
        qv /= np.linalg.norm(qv)
        sim = X @ qv
        order = np.argsort(-sim)
        pool_ids, seen = [], set()
        for j in order:
            cid = all_ids[j]
            if cid in seen:
                continue
            seen.add(cid)
            pool_ids.append(cid)
            if len(pool_ids) >= POOL_SIZE - 2:
                break
        for required in (c_example[0], c_def[0]):
            if required not in seen:
                pool_ids.append(required)
                seen.add(required)
        text_by_id = {i: t for i, t, _ in candidates}
        cands = [{"id": cid, "text": text_by_id[cid][:4000]} for cid in pool_ids]

        for condition, gold, counterpart in (("contoh", c_example[0], c_def[0]),
                                     ("definisi", c_def[0], c_example[0])):
            rows.append({
                "query_id": f"ctrl-{n}-{condition}",
                "pair_id": f"ctrl-{n}",
                "query": query,
                "instruksi_kondisi": INSTRUCTIONS[condition],
                "kondisi": condition,
                "gold": [gold],
                "counterpart": [counterpart],
                "profile": {},
                "cands": cands,
                "provenance": {"book_slug": slug, "section": section},
            })
        if (n + 1) % 20 == 0:
            print(f"  topik {n+1}/{len(chosen)}", flush=True)

    out = Q2 / "raw/pools_control_textual.jsonl"
    with out.open("w") as f:
        for b in rows:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")

    audit = [{"query_id": b["query_id"], "kueri": b["query"],
              "kondisi": b["kondisi"],
              "gold_cuplikan": next(c["text"][:260] for c in b["cands"]
                                    if c["id"] == b["gold"][0]),
              "counterpart_cuplikan": next(c["text"][:260] for c in b["cands"]
                                           if c["id"] == b["counterpart"][0])}
             for b in rows[:40:2]]
    (Q2 / "raw/control_track_audit20.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False))

    meta = {
        "dibuat": datetime.now(timezone.utc).isoformat(),
        "config_version_lock": lock["config_version"],
        "_apa_ini": ("Control track with textual evidence, a mirror of the paired-track protocol with the "
                     "difference in the text."),
        "n_topik": len(chosen), "n_baris": len(rows),
        "pool_size": POOL_SIZE,
        "buku_sumber": book,
        "instruksi": INSTRUCTIONS,
        "klasifikasi_tekstual": {
            "penjaga_prosa": ("reject GLYPH and font debris from the PDF; at least 60 alphabetic words, at least 3 "
                              "sentence endings, and a ratio of letters and spaces of at least 0.72"),
            "contoh": "an example, solution, suppose, or calculate marker and a digit ratio of 0.008 to 0.06",
            "definisi": "none of those markers and a digit ratio of at most 0.004",
            "ambigu": "dibuang",
        },
        "kueri_dibangun": ("deterministic from the section titles, with no LLM, so the provenance gains no "
                           "generation layer"),
        "audit_manual": "raw/control_track_audit20.json holds 20 samples for "
                        "manual inspection before the numbers are used",
        "sumber": {"corpus": sha(CORPUS)},
        "output": {"path": "results/raw/pools_control_textual.jsonl",
                   "sha256": sha(out)},
    }
    (Q2 / "raw/pools_control_textual.manifest.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\n{len(chosen)} topics, {len(rows)} rows -> {out}")
    print(f"audit of 20 samples: {Q2}/raw/control_track_audit20.json")


if __name__ == "__main__":
    main()
