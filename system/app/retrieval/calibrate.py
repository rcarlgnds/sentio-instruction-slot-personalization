import json
import os
from datetime import date

import numpy as np

from app.utils.logging import get_logger

log = get_logger(__name__)

CALIBRATION_QUERIES_PATH = "data/calibration_queries.json"

CORPUS_DRIFT_THRESHOLD = 0.15
MIN_QUERIES_FOR_HEALTHY_CALIBRATION = 10

OOD_CALIBRATION_QUERIES = [
    "bagaimana resep membuat rendang daging sapi yang empuk",
    "siapa pemain sepak bola terbaik di dunia saat ini",
    "bagaimana cara merawat tanaman anggrek agar cepat berbunga",
    "apa lirik lagu kebangsaan Indonesia Raya",
    "bagaimana cuaca di Jakarta besok",
    "rekomendasi tempat wisata pantai di Bali",
]

CALIBRATION_QUERIES = [
    "bagaimana cara kerja algoritma quicksort",
    "apa itu struktur data stack dan queue",
    "jelaskan konsep object oriented programming dan inheritance",
    "apa perbedaan TCP dan UDP",
    "bagaimana routing bekerja di internet",
    "apa itu rekursi dalam pemrograman",
    "apa perbedaan supervised dan unsupervised learning",
    "apa itu overfitting dan cara mencegahnya",
    "bagaimana gradient descent bekerja",
    "what is a neural network and how does it learn",
    "apa itu normalisasi database",
    "jelaskan entity relationship diagram",
    "apa itu sistem informasi manajemen",
    "bagaimana proses bisnis didokumentasikan dengan data flow diagram",
    "apa perbedaan primary key dan foreign key",
    "apa itu elastisitas harga permintaan",
    "jelaskan hukum penawaran dan permintaan",
    "apa saja fungsi manajemen dalam organisasi",
    "apa itu laporan keuangan dan neraca",
    "apa itu marketing mix 4P",
    "jelaskan segmentasi pasar dan target market",
    "apa itu brand positioning",
    "bagaimana consumer behavior memengaruhi keputusan pembelian",
    "apa itu standar deviasi dan varians",
    "jelaskan confidence interval",
    "apa itu hypothesis testing dan p-value",
    "bagaimana cara kerja regresi linear",
    "apa itu distribusi normal",
    "what is the central limit theorem",
]


def load_query_sets() -> tuple[list[str], list[str], str]:
    if os.path.exists(CALIBRATION_QUERIES_PATH):
        try:
            data = json.load(open(CALIBRATION_QUERIES_PATH))
            in_domain = data["in_domain"]
            ood = data["ood_probe"]
            version = data.get("_version", "external")
            log.info(f"[calibrate] query set {version}: "
                     f"{len(in_domain)} in-domain, {len(ood)} OOD probes")
            return in_domain, ood, version
        except (OSError, KeyError, json.JSONDecodeError) as e:
            log.warning(f"[calibrate] {CALIBRATION_QUERIES_PATH} unreadable ({e}), "
                        "falling back to the built-in calib-v1 list")
    return CALIBRATION_QUERIES, OOD_CALIBRATION_QUERIES, "calib-v1-builtin"


def _top1_scores(embedder, qdrant, queries) -> list[float]:
    out = []
    for query in queries:
        emb = embedder.embed_query(query)
        results = qdrant.hybrid_search(
            dense_query=emb["dense"],
            sparse_indices=emb["sparse_indices"],
            sparse_values=emb["sparse_values"],
            limit=10,
        )
        if results:
            out.append(max(r.sim_score for r in results))
    return out


def compute_bounds(embedder, qdrant) -> dict | None:
    corpus_count = qdrant.count()
    if corpus_count == 0:
        log.warning("[calibrate] corpus empty, bounds not computed")
        return None

    in_domain_queries, ood_queries, calib_version = load_query_sets()
    indomain = _top1_scores(embedder, qdrant, in_domain_queries)
    if len(indomain) < MIN_QUERIES_FOR_HEALTHY_CALIBRATION:
        log.warning(
            f"[calibrate] only {len(indomain)} in-domain queries scored, "
            "too few; bounds left unchanged"
        )
        return None
    ood = _top1_scores(embedder, qdrant, ood_queries)

    ind = np.asarray(indomain)
    ind_p5 = float(np.percentile(ind, 5))
    absolute_max = round(float(np.percentile(ind, 99)), 4)

    if ood:
        ood_ceiling = float(np.percentile(np.asarray(ood), 95))
        if ood_ceiling < ind_p5:
            absolute_min = round((ood_ceiling + ind_p5) / 2, 4)
            gate_method = "gap-midpoint"
        else:
            absolute_min = round(ind_p5 - 0.02, 4)
            gate_method = "overlap-fallback"
    else:
        absolute_min = round(ind_p5 - 0.02, 4)
        gate_method = "no-ood-probe"

    return {
        "absolute_min": absolute_min,
        "absolute_max": absolute_max,
        "n_queries": len(indomain),
        "n_ood_queries": len(ood),
        "gate_method": gate_method,
        "calibration_set_version": calib_version,
        "indomain_p5": round(ind_p5, 4),
        "ood_p95": round(float(np.percentile(np.asarray(ood), 95)), 4) if ood else None,
        "corpus_chunk_count": corpus_count,
        "model_version": "bge-m3",
        "date": date.today().isoformat(),
        "provisional": True,
        "median": round(float(np.median(ind)), 4),
    }


def write_bounds(path: str, bounds: dict) -> None:
    with open(path, "w") as f:
        json.dump(bounds, f, indent=2)
    log.info(
        f"[calibrate] bounds written to {path}: "
        f"[{bounds['absolute_min']}, {bounds['absolute_max']}] "
        f"(n={bounds['n_queries']}, corpus={bounds['corpus_chunk_count']})"
    )


def _corpus_drift(path: str, current_count: int) -> float:
    try:
        prev = json.load(open(path)).get("corpus_chunk_count")
    except (OSError, json.JSONDecodeError):
        prev = None
    if not prev:
        return float("inf")
    return abs(current_count - prev) / prev


def maybe_recalibrate(embedder, qdrant, path: str) -> bool:
    current = qdrant.count()
    if current == 0:
        return False
    drift = _corpus_drift(path, current)
    if drift < CORPUS_DRIFT_THRESHOLD:
        log.info(f"[calibrate] corpus drift {drift:.1%} < {CORPUS_DRIFT_THRESHOLD:.0%}, skip")
        return False

    log.info(f"[calibrate] corpus drift {drift:.1%} >= threshold, recalibrate (corpus={current})")
    bounds = compute_bounds(embedder, qdrant)
    if bounds is None:
        return False
    write_bounds(path, bounds)
    return True
