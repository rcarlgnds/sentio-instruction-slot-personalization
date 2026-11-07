from app.config import settings
from app.domain.models import Chunk

from app.utils.logging import get_logger

log = get_logger(__name__)

class HybridEmbedder:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self):
        from FlagEmbedding import BGEM3FlagModel

        self._model = BGEM3FlagModel(
               settings.embedding_model_name,
               use_fp16=settings.embedding_fp16,
               device=settings.embedding_device,
        )

        log.info(f"Embedding model loaded: {settings.embedding_model_name} on {settings.embedding_device}")

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        text = [c.text_for_embedding for c in chunks]

        output = self._model.encode(
            text,
            batch_size=settings.embedding_batch_size,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
            max_length=512,
        )

        dense = output['dense_vecs']
        sparse_weights = output['lexical_weights']

        for i, chunk in enumerate(chunks):
            chunk.dense_vector = dense[i].tolist()

            sparse = sparse_weights[i]
            chunk.sparse_indices = list(sparse.keys())
            chunk.sparse_values = list(sparse.values())

        return chunks

    def embed_query(self, query: str) -> dict:
        output = self._model.encode(
            [query],
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        dense = output['dense_vecs'][0]
        sparse = output['lexical_weights'][0]

        return {
            "dense": dense.tolist(),
            "sparse_indices": list(sparse.keys()),
            "sparse_values": list(sparse.values()),
        }

    def embed_text_dense(self, texts: list[str]) -> list[list[float]]:
        output = self._model.encode(
            texts,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
            max_length=512,
        )

        return [v.tolist() for v in output['dense_vecs']]
