import json
import math
import os
import re
from collections import Counter, defaultdict

import numpy as np

from app.benchmark.sanitizer import ChunkSanitizer
from app.config import settings
from app.domain.models import StudentContext
from app.ingestion.embedder import HybridEmbedder
from app.retrieval import cps
from app.retrieval.hybrid_search import build_payload_filter
from app.retrieval.normalization import CalibrationBounds, normalize_sims
from app.retrieval.prerequisite_graph import PrerequisiteGraph
from app.storage.postgres_client import PostgresAdapter
from app.storage.qdrant_client import QdrantAdapter

DEDUP_COSINE_THRESHOLD = 0.85
OOD_MIN_CPS = 0.55
QC_PROTOCOL = "v2-multiarm"
BM25_TOP_K = 50
BM25_K1 = 1.5
BM25_B = 0.75
BM25_CORPUS_PATH = "data/benchmark_release/corpus/corpus.jsonl"
BM25_UUID_MAP_PATH = "data/benchmark_release/uuid_to_canonical.json"

GOLD_GATE_THRESHOLDS = {"prerequisite_chain_depth": 0.5}
GOLD_GATE_DEFAULT = 1.0

_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")


def _bm25_tokens(text: str) -> list:
    return _TOKEN_RE.findall((text or "").lower())


class Bm25Index:
    def __init__(self, corpus_path: str = BM25_CORPUS_PATH,
                 uuid_map_path: str = BM25_UUID_MAP_PATH):
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(
                f"{corpus_path} does not exist. The BM25 arm reads this corpus snapshot, "
                "which is not in this repository.")
        canonical_to_uuid = {}
        if os.path.exists(uuid_map_path):
            with open(uuid_map_path) as f:
                canonical_to_uuid = {v: k for k, v in json.load(f).items()}
        self.doc_ids = []
        self.doc_len = []
        self.postings = defaultdict(list)
        with open(corpus_path) as f:
            for line in f:
                row = json.loads(line)
                canonical = row["_id"]
                doc_idx = len(self.doc_ids)
                self.doc_ids.append(canonical_to_uuid.get(canonical, canonical))
                tokens = _bm25_tokens(row.get("text") or "")
                self.doc_len.append(len(tokens))
                for term, tf in Counter(tokens).items():
                    self.postings[term].append((doc_idx, tf))
        self.n_docs = len(self.doc_ids)
        self.avg_len = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0

    def top_ids(self, query_text: str, k: int = BM25_TOP_K) -> list:
        scores = defaultdict(float)
        for term in set(_bm25_tokens(query_text)):
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = math.log(1 + (self.n_docs - len(plist) + 0.5) / (len(plist) + 0.5))
            for doc_idx, tf in plist:
                denom = tf + BM25_K1 * (1 - BM25_B + BM25_B * self.doc_len[doc_idx] / (self.avg_len or 1))
                scores[doc_idx] += idf * tf * (BM25_K1 + 1) / denom
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [self.doc_ids[i] for i, _ in ranked]

META_LANGUAGE = re.compile(
    r"\b(chunk|teks di atas|dokumen ini|kutipan di atas|berdasarkan teks|passage)\b",
    re.IGNORECASE)

TERM_COGNATES = {
    "normalization": ["normalisasi"],
    "network": ["jaringan"],
    "regression": ["regresi"],
    "distribution": ["distribusi"],
    "variance": ["varians", "variansi"],
    "sampling": ["penyampelan", "pengambilan sampel"],
    "index": ["indeks"],
    "class": ["kelas"],
    "object": ["objek"],
    "model": [],
    "demand": ["permintaan"],
    "market": ["pasar"],
    "inheritance": ["pewarisan"],
    "cluster": ["klaster"],
    "protocol": ["protokol"],
    "capital": ["kapital", "modal"],
    "function": ["fungsi"],
    "table": ["tabel"],
    "correlation": ["korelasi"],
}


def _loaded_embedder() -> HybridEmbedder:
    embedder = HybridEmbedder()
    if embedder._model is None:
        embedder.load()
    return embedder


def term_in_query(term: str, query_text: str) -> bool:
    variants = [term] + TERM_COGNATES.get(term.lower(), [])
    low = query_text.lower()
    return any(re.search(r"\b" + re.escape(v.lower()) + r"\b", low) for v in variants)


class SemanticDeduper:
    def __init__(self, threshold: float = DEDUP_COSINE_THRESHOLD, embed_fn=None):
        self.threshold = threshold
        self._embed_fn = embed_fn or (lambda texts: _loaded_embedder().embed_text_dense(texts))
        self._accepted: list = []

    def _vectorize(self, text: str) -> np.ndarray:
        vec = np.asarray(self._embed_fn([text])[0], dtype=np.float32)
        return vec / (np.linalg.norm(vec) or 1.0)

    def preload(self, texts: list):
        for text in texts:
            self._accepted.append(self._vectorize(text))

    def matches(self, vec: np.ndarray) -> bool:
        if not self._accepted:
            return False
        sims = np.stack(self._accepted) @ vec
        return float(sims.max()) > self.threshold

    def commit(self, vec: np.ndarray):
        self._accepted.append(vec)


class RetrievalProbe:
    def __init__(self):
        self.embedder = _loaded_embedder()
        self.qdrant = QdrantAdapter()
        self.bounds = CalibrationBounds.load(settings.calibration_path)
        self.graph = PrerequisiteGraph.load(PostgresAdapter())

    def probe(self, query_text: str, student: StudentContext) -> dict:
        embedding = self.embedder.embed_query(query_text)
        payload_filter = build_payload_filter("strict", student)
        results = self.qdrant.hybrid_search(
            dense_query=embedding["dense"],
            sparse_indices=embedding["sparse_indices"],
            sparse_values=embedding["sparse_values"],
            limit=settings.retrieval_pool_size,
            payload_filter=payload_filter,
        )
        if not results:
            return {"passed": False, "reason": "no_results", "top_cps": 0.0}
        in_domain = normalize_sims(results, self.bounds)
        if not in_domain:
            return {"passed": False, "reason": "stage1_out_of_domain",
                    "top_cps": 0.0, "max_raw": max(r.sim_score for r in results)}
        ranked = cps.rerank(results, student, prerequisite_graph=self.graph)
        top_cps = ranked[0].cps_score or 0.0
        gold_pool_ids = [r.payload.get("chunk_id", "") for r in ranked]
        return {"passed": True, "reason": "",
                "top_cps": round(top_cps, 4),
                "retrieved_chunk_ids": gold_pool_ids}


class QcRunner:
    def __init__(self, preload_texts: list = None, embed_fn=None,
                 probe=None, bm25=None):
        self.deduper = SemanticDeduper(embed_fn=embed_fn)
        if preload_texts:
            self.deduper.preload(preload_texts)
        self.probe = probe or RetrievalProbe()
        self.bm25 = bm25 or Bm25Index()
        self.sanitizer = ChunkSanitizer()

    def check(self, query, student: StudentContext) -> dict:
        qc = {"dedup": "ok", "ood": "ok", "class_fit": "ok", "meta": "ok",
              "gold_retrievable": None, "passed": True, "protocol": QC_PROTOCOL}

        if META_LANGUAGE.search(query.query_text):
            qc.update(meta="meta_language", passed=False)
            return qc

        if query.class_label == "multi_disciplinary_ambiguity":
            term = query.class_metadata.get("ambiguous_term", "")
            if term and not term_in_query(term, query.query_text):
                qc.update(class_fit="term_missing", passed=False)
                return qc

        vec = self.deduper._vectorize(query.query_text)
        if self.deduper.matches(vec):
            qc.update(dedup="duplicate", passed=False)
            return qc

        result = self.probe.probe(query.query_text, student)
        qc["top_cps"] = result.get("top_cps", 0.0)
        if not result["passed"]:
            qc.update(ood=result["reason"], passed=False)
            return qc

        gold = set(query.gold_relevant_chunks)
        hybrid_pool = set(result.get("retrieved_chunk_ids", []))
        bm25_pool = set(self.bm25.top_ids(query.query_text))
        ratio_hybrid = round(len(gold & hybrid_pool) / len(gold), 3) if gold else 0.0
        ratio_bm25 = round(len(gold & bm25_pool) / len(gold), 3) if gold else 0.0
        qc["gold_retrievable"] = max(ratio_hybrid, ratio_bm25)
        qc["arms"] = {"hybrid": ratio_hybrid, "bm25": ratio_bm25}
        threshold = GOLD_GATE_THRESHOLDS.get(query.class_label, GOLD_GATE_DEFAULT)
        if qc["gold_retrievable"] < threshold:
            qc.update(gold="not_retrievable", passed=False)
            return qc

        self.deduper.commit(vec)
        return qc
