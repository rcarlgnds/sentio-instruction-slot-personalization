from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    HnswConfigDiff,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    PointStruct,
    SparseVector,
)

import numpy as np

from app.config import settings
from app.domain.models import Chunk, IngestionJob, RetrievalResult
from app.utils.logging import get_logger

log = get_logger(__name__)

class QdrantAdapter:
    def __init__(self):
        self.client = QdrantClient(
            url=settings.qdrant_url,
            timeout=settings.qdrant_timeout,
        )
        self.collection = settings.qdrant_collection

    def ensure_collection(self):
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection in existing:
            log.info(f"Collection '{self.collection}' already exists")
            return

        log.info(f"Creating collection '{self.collection}'...")
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                "dense": VectorParams(
                    size=1024,
                    distance=Distance.COSINE,
                    on_disk=False,
                ),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False),
                ),
            },
            hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
            quantization_config=ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8,
                    always_ram=True,
                ),
            ),
        )
        log.info(f"Collection '{self.collection}' created")

        self._create_payload_indexes()

    def _create_payload_indexes(self):
        indexes = [
            ("department_id", "keyword"),
            ("course_id", "keyword"),
            ("semester", "integer"),
            ("chunk_level", "keyword"),
            ("topic_tags", "keyword"),
            ("document_id", "keyword"),
        ]

        for field_name, field_type in indexes:
            try:
                self.client.create_payload_index(
                    self.collection,
                    field_name=field_name,
                    field_schema=field_type,
                )
                log.info(f"Created payload index: {field_name} ({field_type})")
            except Exception as e:
                log.debug(f"Payload index {field_name}: {e}")

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        job: IngestionJob,
        course_meta: dict,
        document_title: str = "",
    ):
        points = []
        for idx, chunk in enumerate(chunks):
            point = PointStruct(
                id=chunk.chunk_id,
                vector={
                    "dense": chunk.dense_vector,
                    "sparse": SparseVector(
                        indices=chunk.sparse_indices,
                        values=chunk.sparse_values,
                    )
                },
                payload={
                    "document_id": job.document_id,
                    "document_title": document_title,
                    "department_id": job.department_id,
                    "course_id": job.course_id,
                    "minio_object_key": job.object_key,
                    "uploaded_by": job.uploaded_by,
                    "uploaded_at": job.created_at,

                    "course_name": course_meta.get("course_name", ""),
                    "department_name": course_meta.get("department_name", ""),
                    "semester": course_meta.get("semester", 0),
                    "credits": course_meta.get("credits", 0),

                    "chunk_id": chunk.chunk_id,
                    "chunk_index": idx,
                    "chunk_level": chunk.chunk_level,
                    "parent_chunk_id": chunk.parent_chunk_id or "",
                    "section_path": chunk.section_path,
                    "position": chunk.position or {},
                    "raw_text": chunk.raw_text,
                    "token_count": chunk.token_count,
                    "has_context_prefix": bool(chunk.has_context_prefix),

                    "embedding_model": "bge-m3",
                    "embedding_version": "1.5",
                }
            )
            points.append(point)

        batch_size = 128
        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.collection,
                points=points[i:i + batch_size],
                wait=True,
            )
        log.info(f"Upserted {len(points)} points to Qdrant ({(len(points) + batch_size - 1) // batch_size} batches)")

    def hybrid_search(
            self,
            dense_query: list[float],
            sparse_indices: list[int],
            sparse_values: list[float],
            limit: int = 30,
            payload_filter: qm.Filter | None = None,
    ) -> list[RetrievalResult]:
        sparse_vec = qm.SparseVector(
            indices=sparse_indices,
            values=sparse_values,
        )

        params = qm.SearchParams(exact=settings.qdrant_exact_search)
        common = dict(collection_name=self.collection, limit=50,
                      query_filter=payload_filter, with_payload=True,
                      search_params=params)
        dense_hits = self.client.search(
            query_vector=qm.NamedVector(name="dense", vector=dense_query),
            with_vectors=["dense"], **common)
        sparse_hits = self.client.search(
            query_vector=qm.NamedSparseVector(name="sparse", vector=sparse_vec),
            with_vectors=["dense"], **common)

        k = settings.qdrant_rrf_k
        rrf: dict[str, float] = {}
        points: dict[str, object] = {}
        for hits in (dense_hits, sparse_hits):
            order = sorted(hits, key=lambda h: (-h.score, str(h.id)))
            for rank, h in enumerate(order):
                hid = str(h.id)
                rrf[hid] = rrf.get(hid, 0.0) + 1.0 / (k + rank + 1)
                points.setdefault(hid, h)

        selected = sorted(rrf, key=lambda hid: (-rrf[hid], hid))[:limit]

        q = np.asarray(dense_query, dtype=np.float32)
        q = q / np.linalg.norm(q)

        results = []
        for hid in selected:
            p = points[hid]
            v = np.asarray(p.vector["dense"], dtype=np.float32)
            sim = float((v @ q) / np.linalg.norm(v))
            results.append(RetrievalResult(chunk_id=hid, payload=p.payload, sim_score=sim))

        return results

    def delete_by_document(self, document_id: str):
        self.client.delete(
            collection_name=self.collection,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[qm.FieldCondition(key="document_id", match=qm.MatchValue(value=document_id))]
                )
            ),
        )

    def count(self) -> int:
        info = self.client.get_collection(self.collection)
        return info.points_count or 0
