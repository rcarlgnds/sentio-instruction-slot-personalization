import json
from pathlib import Path

import numpy as np

from app.config import settings
from app.domain.models import RoutingMode
from app.utils.logging import get_logger

log = get_logger(__name__)

STRICT_UTTERANCES = [
    "kapan UTS database systems",
    "kapan jadwal UTS mata kuliah X",
    "kapan deadline tugas akhir",
    "jadwal kuliah minggu ini apa",
    "jam kuliah hari selasa",

    "siapa dosen pengampu ISYS6123",
    "siapa koordinator program studi",
    "siapa dosen wali saya",
    "nomor kontak tata usaha",

    "ruang kelas hari ini dimana",
    "lokasi ujian akhir dimana",

    "bobot SKS mata kuliah X",
    "prasyarat mata kuliah Y",
    "daftar mata kuliah semester 3",
    "kode mata kuliah basis data apa",
    "berapa SKS total semester ini",

    "syarat lulus program studi",
    "aturan cuti akademik",
    "nilai minimum kelulusan",
    "batas maksimal ketidakhadiran",
]

SOCRATIC_UTTERANCES = [
    "jelaskan konsep normalisasi",
    "kenapa kita butuh 3NF",
    "bedanya SQL dan NoSQL apa",
    "apa gunanya index di database",
    "kapan pakai B-tree vs hash index",
    "jelaskan ACID properties",

    "bedanya REST dan SOAP apa",
    "kenapa pakai HTTP/2",
    "jelaskan cara kerja DNS",
    "apa itu load balancer",
    "bagaimana cara kerja TCP handshake",

    "apa itu race condition",
    "jelaskan deadlock dan livelock",
    "bedanya thread dan process",
    "kenapa async lebih cepat dari sync",

    "apa itu dependency injection",
    "jelaskan prinsip SOLID",
    "kenapa microservice lebih kompleks dari monolith",
    "kapan pakai design pattern singleton",

    "analogi untuk memahami recursion apa ya",
    "bedanya array dan linked list",
    "jelaskan Big-O notation",
    "kenapa merge sort lebih stabil",

    "gw bingung soal pointer di C",
    "gw gak paham mutex vs semaphore",
    "bro ajarin gw dong konsep X",
    "gagal paham gw bang sama materi Y",
    "tolong jelasin pakai analogi yang gampang",
    "gimana sih cara kerjanya Z, pusing gw",

    "siapa penemu algoritma X",
    "berapa time complexity dari Y",
    "berapa lama waktu eksekusi struktur data Z",
    "bagaimana sejarah diciptakannya W",

    "how does X work in computer science",
    "explain the difference between A and B",
    "what is the mechanism behind garbage collection",
    "tell me about the underlying architecture of Y",

    "bagaimana cara bikin sistem RAG sendiri",
    "bagaimana cara setup server kafka dari nol",
    "bagaimana cara implementasi authentication",
    "cara bikin REST API di Go",
    "cara deploy microservice ke kubernetes",
    "gimana sih cara integrasi ollama dengan aplikasi",
    "how to build a RAG system from scratch",
    "tutorial singkat bikin chatbot pakai LLM",
    "langkah-langkah bikin search engine dari nol",
    "panduan umum setup development environment untuk ML",
]


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


class SemanticRouter:
    def __init__(self, encoder=None):
        self.encoder = encoder
        self.threshold = settings.routing_threshold
        self.gap_margin = settings.routing_gap_margin
        self.fallback_mode: RoutingMode = settings.routing_fallback_mode
        self._centroid_strict: np.ndarray | None = None
        self._centroid_socratic: np.ndarray | None = None
        self.calibration_meta: dict = {}

    def load(self):
        path = Path(settings.routing_centroids_path)
        if path.exists():
            data = json.loads(path.read_text())
            self._centroid_strict = _normalize(np.asarray(data["factual_centroid"], dtype=np.float32))
            self._centroid_socratic = _normalize(np.asarray(data["socratic_centroid"], dtype=np.float32))
            self.calibration_meta = {k: v for k, v in data.items() if not k.endswith("_centroid")}
            log.info(
                f"Routing centroids loaded: n_factual={data.get('n_factual')} "
                f"n_socratic={data.get('n_socratic')} source={data.get('source')} "
                f"provisional={data.get('provisional')}"
            )
        else:
            log.warning(
                f"{settings.routing_centroids_path} not found. Centroids computed "
                "PROVISIONALLY from the built-in utterances. The study's centroid file is not "
                "in this repository."
            )
            self._centroid_strict = self._centroid_from(STRICT_UTTERANCES)
            self._centroid_socratic = self._centroid_from(SOCRATIC_UTTERANCES)
            self.calibration_meta = {
                "source": "builtin_utterances",
                "n_factual": len(STRICT_UTTERANCES),
                "n_socratic": len(SOCRATIC_UTTERANCES),
                "provisional": True,
            }

    def _centroid_from(self, utterances: list[str]) -> np.ndarray:
        vecs = np.asarray(self.encoder.embed_text_dense(utterances))
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        return _normalize(vecs.mean(axis=0))

    def route_embedded(self, dense_vector) -> tuple[RoutingMode, float, float]:
        q = _normalize(np.asarray(dense_vector, dtype=np.float32))

        cos_strict = float(q @ self._centroid_strict)
        cos_socratic = float(q @ self._centroid_socratic)

        confidence = max(cos_strict, cos_socratic)
        gap = abs(cos_strict - cos_socratic)
        winner: RoutingMode = "strict" if cos_strict >= cos_socratic else "socratic"

        if confidence < self.threshold or gap < self.gap_margin:
            return self.fallback_mode, confidence, gap

        return winner, confidence, gap

    def route_with_scores(self, query: str) -> tuple[RoutingMode, float, float]:
        q = np.asarray(self.encoder.embed_text_dense([query])[0])
        return self.route_embedded(q)

    def route(self, query: str) -> RoutingMode:
        return self.route_with_scores(query)[0]
