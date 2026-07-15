from .document_store import DocumentStore
from .llm import LLMClient, QueryExpander, Generator
from .reranker import Reranker
from .pipeline import RAGPipeline
from .evaluator import Evaluator


def build_pipeline():
    document_store = DocumentStore().build_index()
    llm_client = LLMClient()
    query_expander = QueryExpander(llm_client)
    reranker = Reranker()
    generator = Generator(llm_client)

    pipeline = RAGPipeline(document_store, query_expander, reranker, generator)
    return pipeline, document_store, generator


def run_eval(pipeline, document_store, generator, n=15, k=5):
    print(f"\nBuilding synthetic eval set (n={n})...")
    evaluator = Evaluator(generator, document_store.chunks)
    evaluator.build_eval_set(n=n)

    print(f"\nComparing retrieval strategies (Recall@{k}):")
    evaluator.compare(
        {
            "naive": pipeline.retrieve_naive,
            "multiquery": pipeline.retrieve_multiquery,
            "reranked": pipeline.retrieve_reranked,
        },
        k=k,
    )


def run_interactive(pipeline):
    print("\nAsk a question about the annual report (or type 'quit' to exit).")
    print("Using strategy: multiquery + reranking\n")

    while True:
        query = input("Question: ").strip()
        if query.lower() in ("quit", "exit"):
            break
        if not query:
            continue

        answer, chunks = pipeline.answer(query, strategy="reranked", k=5)
        print("\n======Answer======")
        print(answer)
        print()


if __name__ == "__main__":
    pipeline, document_store, generator = build_pipeline()

    run_eval(pipeline, document_store, generator, n=15, k=5)
    run_interactive(pipeline)
