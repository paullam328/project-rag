from sentence_transformers import CrossEncoder

from . import config


class Reranker:
    """
    Second-stage precision filter over an already-retrieved candidate pool.
    Uses a cross-encoder (query+doc scored jointly) rather than the bi-encoder
    similarity DocumentStore uses for the first-pass retrieval.
    """

    def __init__(self, model_name=config.CROSS_ENCODER_MODEL_NAME):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, docs, top_k=5):
        if not docs:
            return []
        docs = list(docs)
        pairs = [[query, doc] for doc in docs]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(scores, docs), reverse=True)
        return [doc for score, doc in ranked[:top_k]]
