import os
import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("SENTIO_ROOT")
            or Path(__file__).resolve().parents[4])
CORPUS = ROOT / "system/data/benchmark_release/corpus/corpus.jsonl"
Q2 = ROOT / "study/channel/results"
SHARED = ROOT / "study/_shared"
RAW_N1 = ROOT / "study/retrieval/results/raw"
SEED = 42

FALLBACK_PAIRS = ("stat-introduction-to-statistics-lane-publicdomain",
                     "stat-introductory-statistics-2e-ccby-2023")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_corpus() -> dict:
    idx = {}
    for l in CORPUS.open():
        r = json.loads(l)
        m = r["metadata"]
        idx[r["_id"]] = {
            "text": r.get("text") or "",
            "book": m.get("book_slug"), "dept": m.get("department"),
            "section": m.get("section_path") or "", "chunk_id": r["_id"],
        }
    return idx


def pairs_from_gold(corpus: dict) -> tuple:
    title = Counter()
    p = ROOT / "system/data/paired_set/paired_queries.jsonl"
    if p.exists():
        for l in p.open():
            r = json.loads(l)
            for key in ("gold_anchor", "counterpart_anchor"):
                a = (r.get(key) or [{}])[0]
                if a.get("document_title"):
                    title[a["document_title"]] += 1
    if len(title) >= 2:
        mapping = {}
        for l in CORPUS.open():
            m = json.loads(l)["metadata"]
            dt = (m.get("document_title") or "").replace(" ", "_")
            mapping.setdefault(dt, m.get("book_slug"))
        two_slugs = []
        for j, _ in title.most_common(2):
            slug = mapping.get(j) or mapping.get(j.replace("_", " "))
            if slug:
                two_slugs.append(slug)
        if len(two_slugs) == 2:
            return tuple(two_slugs)
    return FALLBACK_PAIRS


def tfidf_logreg_cv(docs: list, label: list, group: list, k=5) -> dict:
    import math

    def tokenize_text(t: str) -> list:
        words = [w for w in ''.join(
            c.lower() if c.isalnum() else ' ' for c in t).split() if len(w) > 2]
        return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]

    docs_tok = [tokenize_text(d) for d in docs]
    rng = random.Random(SEED)
    unique_groups = sorted(set(group))
    rng.shuffle(unique_groups)
    fold_group = {g: i % k for i, g in enumerate(unique_groups)}
    folds = [fold_group[g] for g in group]

    accuracy = []
    for f in range(k):
        tr = [i for i in range(len(docs)) if folds[i] != f]
        te = [i for i in range(len(docs)) if folds[i] == f]
        if not te or len(set(label[i] for i in tr)) < 2:
            continue
        df = Counter()
        for i in tr:
            for w in set(docs_tok[i]):
                df[w] += 1
        vocab = {w: j for j, (w, c) in enumerate(df.most_common(20000)) if c >= 3}
        n_tr = len(tr)
        idf = {w: math.log(n_tr / (1 + df[w])) for w in vocab}

        def tf_vector(i):
            tf = Counter(docs_tok[i])
            v = {}
            norm = 0.0
            for w, c in tf.items():
                if w in vocab:
                    x = (1 + math.log(c)) * idf[w]
                    v[vocab[w]] = x
                    norm += x * x
            norm = math.sqrt(norm) or 1.0
            return {j: x / norm for j, x in v.items()}

        X_tr = [tf_vector(i) for i in tr]
        y_tr = [1.0 if label[i] else 0.0 for i in tr]
        w = [0.0] * len(vocab)
        b = 0.0
        lr, epoch, l2 = 0.5, 60, 1e-4
        order = list(range(len(X_tr)))
        rng2 = random.Random(SEED + f)
        for _ in range(epoch):
            rng2.shuffle(order)
            for i in order:
                z = b + sum(w[j] * x for j, x in X_tr[i].items())
                p = 1 / (1 + math.exp(-max(-30, min(30, z))))
                g = p - y_tr[i]
                b -= lr * g
                for j, x in X_tr[i].items():
                    w[j] -= lr * (g * x + l2 * w[j])
        correct = 0
        for i in te:
            v = tf_vector(i)
            z = b + sum(w[j] * x for j, x in v.items())
            pred = z > 0
            correct += int(pred == bool(label[i]))
        accuracy.append(correct / len(te))
    mean = sum(accuracy) / len(accuracy) if accuracy else None
    return {"akurasi_per_fold": [round(a, 4) for a in accuracy],
            "akurasi_mean": round(mean, 4) if mean is not None else None,
            "n_fold": len(accuracy)}


def main():
    lock = json.loads((SHARED / "config_lock.json").read_text())
    corpus = load_corpus()
    b1, b2 = pairs_from_gold(corpus)
    print(f"book pair: {b1}  versus  {b2}")

    rows = [c for c in corpus.values() if c["book"] in (b1, b2)
            and len(c["text"]) >= 200]
    if not rows:
        raise SystemExit(f"ABORT: no chunk for the pair {b1}/{b2}")
    docs = [c["text"] for c in rows]
    label = [c["book"] == b1 for c in rows]
    section_groups = [f"{c['book']}|{c['section']}" for c in rows]
    chunk_groups = [c["chunk_id"] for c in rows]
    dist = Counter(label)
    baseline = max(dist.values()) / len(label)
    print(f"n chunk {len(rows)}  ({dist[True]} vs {dist[False]})  "
          f"majority class baseline {baseline:.4f}")

    print("CV at chunk level (optimistic) ...")
    cv_chunk = tfidf_logreg_cv(docs, label, chunk_groups)
    print(f"  accuracy {cv_chunk['akurasi_mean']}")
    print("CV grouped by section (conservative) ...")
    cv_group = tfidf_logreg_cv(docs, label, section_groups)
    print(f"  accuracy {cv_group['akurasi_mean']}")

    output = {
        "dibuat": datetime.now(timezone.utc).isoformat(),
        "config_version_lock": lock["config_version"],
        "_apa_ini": ("How much book-membership signal the chunk text actually carries, measured with a simple "
                     "classifier."),
        "pasangan_buku": {"kelas_positif": b1, "kelas_negatif": b2},
        "n_chunk": len(rows),
        "distribusi": {b1: dist[True], b2: dist[False]},
        "baseline_kelas_mayoritas": round(baseline, 4),
        "model": "logistic regression on word 1-2 gram TF-IDF, CPU, seed 42",
        "cv_chunk_level": cv_chunk,
        "cv_grouped_by_section": cv_group,
        "angka_yang_dikutip": "cv_grouped_by_section (conservative)",
        "dua_cabang_interpretasi": {
            "bila_akurasi_mendekati_baseline": (
                "the program-membership signal is not recoverable from the chunk text; the premise is "
                "then a measured result, and the failure of the text rerankers follows from the channel, "
                "not from a weakness of the models"),
            "bila_akurasi_jauh_di_atas_baseline": (
                "the signal is in the text and a trivial classifier recovers it, so the failure of the "
                "instruction-following rerankers is sharper: they fail to use information that is there"),
        },
        "batasan": ("the classifier sees the whole chunk as a standalone document, while a reranker judges "
                    "query-document relevance; this is an upper bound on the available signal, not a "
                    "simulation of the reranker's task"),
        "label_source": "deterministic (CPU classifier, fixed seed)",
    }
    out = Q2 / "tables/separability_probe.json"
    out.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    (Q2 / "raw/separability_probe.manifest.json").write_text(json.dumps({
        "run_id": "q2_sep_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "script": "study/channel/scripts/5_why_slot_fails/separability.py",
        "config_version": lock["config_version"],
        "label_source": "deterministic",
        "sumber": {"corpus": sha(CORPUS)},
        "output": {"path": "results/tables/separability_probe.json",
                   "sha256": sha(out), "tingkat_reproduksi": "1_bit_identical"},
    }, indent=2, ensure_ascii=False))
    print(f"written: {out}")


if __name__ == "__main__":
    main()
