import hashlib
import json
import os
import random
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("SENTIO_ROOT")
            or Path(__file__).resolve().parents[4])
AI = ROOT / "system"
CORPUS = ROOT / "system/data/benchmark_release/corpus/corpus.jsonl"
Q2 = ROOT / "study/channel/results"
SHARED = ROOT / "study/_shared"
SEED = 42


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def track_pairs() -> tuple:
    title = Counter()
    p = ROOT / "system/data/paired_set/paired_queries.jsonl"
    for l in p.open():
        r = json.loads(l)
        for k in ("gold_anchor", "counterpart_anchor"):
            a = (r.get(k) or [{}])[0]
            if a.get("document_title"):
                title[a["document_title"]] += 1
    mapping = {}
    for l in CORPUS.open():
        m = json.loads(l)["metadata"]
        dt = (m.get("document_title") or "").replace(" ", "_")
        mapping.setdefault(dt, m.get("book_slug"))
    return tuple(mapping[j] for j, _ in title.most_common(2) if j in mapping)


def fold_group(group: list, k=5) -> list:
    rng = random.Random(SEED)
    unique = sorted(set(group))
    rng.shuffle(unique)
    mapping = {g: i % k for i, g in enumerate(unique)}
    return [mapping[g] for g in group]


def logreg_cv(X, y, folds, k=5) -> dict:
    import numpy as np
    accs = []
    for f in range(k):
        tr = [i for i in range(len(y)) if folds[i] != f]
        te = [i for i in range(len(y)) if folds[i] == f]
        if not te or len(set(y[i] for i in tr)) < 2:
            continue
        Xtr, ytr = X[tr], np.array([1.0 if y[i] else 0.0 for i in tr])
        w = np.zeros(X.shape[1])
        b = 0.0
        lr, epoch, l2 = 0.5, 200, 1e-3
        n = len(tr)
        for _ in range(epoch):
            z = Xtr @ w + b
            p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
            g = p - ytr
            w -= lr * (Xtr.T @ g / n + l2 * w)
            b -= lr * g.mean()
        pred = (X[te] @ w + b) > 0
        correct = sum(1 for j, i in enumerate(te) if bool(pred[j]) == bool(y[i]))
        accs.append(correct / len(te))
    return {"akurasi_per_fold": [round(a, 4) for a in accs],
            "akurasi_mean": round(sum(accs) / len(accs), 4) if accs else None}


def centroid_cv(X, y, folds, k=5) -> dict:
    import numpy as np
    accs = []
    for f in range(k):
        tr = [i for i in range(len(y)) if folds[i] != f]
        te = [i for i in range(len(y)) if folds[i] == f]
        if not te:
            continue
        a = X[[i for i in tr if y[i]]].mean(axis=0)
        b = X[[i for i in tr if not y[i]]].mean(axis=0)
        a /= np.linalg.norm(a) or 1
        b /= np.linalg.norm(b) or 1
        pred = (X[te] @ a) > (X[te] @ b)
        correct = sum(1 for j, i in enumerate(te) if bool(pred[j]) == bool(y[i]))
        accs.append(correct / len(te))
    return {"akurasi_per_fold": [round(a, 4) for a in accs],
            "akurasi_mean": round(sum(accs) / len(accs), 4) if accs else None}


def main():
    free = int(subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True).split()[0])
    print(f"free VRAM {free} MiB")
    if free < 3500:
        raise SystemExit("ABORT VRAM: need room for BGE-M3 plus 1.8 GB left over")

    lock = json.loads((SHARED / "config_lock.json").read_text())
    b1, b2 = track_pairs()
    print(f"pair: {b1} vs {b2}")

    text, label, section = [], [], []
    for l in CORPUS.open():
        r = json.loads(l)
        m = r["metadata"]
        if m.get("book_slug") in (b1, b2) and len(r.get("text") or "") >= 200:
            text.append(r["text"])
            label.append(m["book_slug"] == b1)
            section.append(f"{m['book_slug']}|{m.get('section_path') or ''}")
    print(f"n chunk {len(text)}")

    sys.path.insert(0, str(AI))
    os.chdir(AI)
    import numpy as np
    from app.ingestion.embedder import HybridEmbedder
    emb = HybridEmbedder()
    emb.load()
    vecs = []
    B = 64
    for i in range(0, len(text), B):
        vecs.extend(emb.embed_text_dense(text[i:i + B]))
        if (i // B) % 20 == 0:
            print(f"  embed {i}/{len(text)}", flush=True)
    X = np.asarray(vecs, dtype=np.float64)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    print("embedding dimension:", X.shape)

    folds = fold_group(section)
    lin = logreg_cv(X, label, folds)
    cen = centroid_cv(X, label, folds)
    dist = Counter(label)
    baseline = max(dist.values()) / len(label)
    print(f"linear probe {lin['akurasi_mean']}  centroid {cen['akurasi_mean']}  "
          f"baseline {baseline:.4f}")

    tfidf = {}
    p_tf = Q2 / "tables/separability_probe.json"
    if p_tf.exists():
        d = json.loads(p_tf.read_text())
        tfidf = {"cv_grouped_by_section": d["cv_grouped_by_section"]["akurasi_mean"],
                 "cv_chunk_level": d["cv_chunk_level"]["akurasi_mean"]}

    output = {
        "dibuat": datetime.now(timezone.utc).isoformat(),
        "config_version_lock": lock["config_version"],
        "_apa_ini": ("Book-membership separability in the production embedding space (BGE-M3 dense), the same "
                     "space the retrieval of the system uses."),
        "pasangan_buku": {"kelas_positif": b1, "kelas_negatif": b2},
        "n_chunk": len(text), "dim": int(X.shape[1]),
        "baseline_kelas_mayoritas": round(baseline, 4),
        "protokol": "grouped 5-fold per section_path, seed 42",
        "probe_linear_logreg": lin,
        "centroid_kosinus_tanpa_pelatihan": cen,
        "pembanding_tfidf": tfidf,
        "arti": ("If separability is high in the embedding space the system uses, the membership "
                 "information is available to neural models; the failure of instruction-following "
                 "rerankers on the paired track is then a failure to use it, not a lack of it."),
        "batasan": ("the probe is supervised and sees the chunk as a standalone document; it measures "
                    "whether the signal is available, not the query-conditioned reranking task"),
        "label_source": "deterministic (frozen embeddings and a probe with a fixed seed)",
    }
    out = Q2 / "tables/separability_probe_embedding.json"
    out.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    (Q2 / "raw/separability_probe_embedding.manifest.json").write_text(json.dumps({
        "run_id": "q2_sepemb_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "script": "study/channel/scripts/5_why_slot_fails/separability_embedding.py",
        "config_version": lock["config_version"],
        "label_source": "deterministic",
        "sumber": {"corpus": sha(CORPUS)},
        "output": {"path": "results/tables/separability_probe_embedding.json",
                   "sha256": sha(out),
                   "tingkat_reproduksi": "2_statistical (embedding GPU)"},
    }, indent=2, ensure_ascii=False))
    print(f"written: {out}")


if __name__ == "__main__":
    main()
