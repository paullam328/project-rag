class RAGPipeline:
    """
    Composes the building blocks into a full pipeline, with three retrieval
    strategies that share the same underlying components:

      naive       : single query -> vector search -> top_k
      multiquery  : query expansion -> vector search per variant -> dedup -> top_k
      reranked    : multiquery -> wider candidate pool -> cross-encoder rerank -> top_k

    Each retrieve_* method has the same signature (query, k) -> List[str] so they
    can be swapped interchangeably, e.g. by the Evaluator.
    """

    def __init__(self, document_store, query_expander, reranker, generator):
        self.document_store = document_store
        self.query_expander = query_expander
        self.reranker = reranker
        self.generator = generator

    def retrieve_naive(self, query, k=5):
        return self.document_store.query(query, n_results=k)

    def retrieve_multiquery(self, query, k=5):
        aug_queries = self.query_expander.expand(query)
        return self.document_store.query(aug_queries, n_results=k)

    def retrieve_reranked(self, query, k=5, candidate_pool_size=10):
        aug_queries = self.query_expander.expand(query)
        candidates = self.document_store.query(aug_queries, n_results=candidate_pool_size)
        unique_candidates = list(set(candidates))
        return self.reranker.rerank(query, unique_candidates, top_k=k)

    def answer(self, query, strategy="reranked", k=5):
        """Full end-to-end: retrieve with the given strategy, then generate an answer."""
        retrieve_fn = {
            "naive": self.retrieve_naive,
            "multiquery": self.retrieve_multiquery,
            "reranked": self.retrieve_reranked,
        }[strategy]

        chunks = retrieve_fn(query, k=k)
        return self.generator.answer(query, context_chunks=chunks), chunks
