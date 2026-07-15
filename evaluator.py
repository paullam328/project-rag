import random


class Evaluator:
    """
    Builds a synthetic eval set (chunk -> LLM-generated question) and measures
    Recall at k (the fraction where test questions' chunk appeared somewhere in the top-k result) 
    for any retrieval function with signature (query, k) -> List[str].

    This is deliberately decoupled from RAGPipeline: it only needs a list of
    chunks (to sample from), a generator (to invert chunk->question), and
    whatever retrieval callables you want to compare.
    """

    def __init__(self, generator, chunks):
        self.generator = generator
        self.chunks = chunks
        self.eval_set = []

    def build_eval_set(self, n=15, seed=None):
        if seed is not None:
            random.seed(seed)
        sample_chunks = random.sample(self.chunks, n) # Takes all the in all the chunks, and choose n of them
        self.eval_set = [
            (self.generator.generate_question_from_chunk(chunk), chunk)
            for chunk in sample_chunks
        ]
        return self.eval_set

    def recall_at_k(self, retrieve_fn, k=5):
        if not self.eval_set:
            raise ValueError("Call build_eval_set() before evaluating.")

        hits = 0
        for question, ground_truth_chunk in self.eval_set:
            retrieved = retrieve_fn(question, k)
            if ground_truth_chunk in retrieved: # If the retrieval fn is capable of fetching the same chunk, then yep +1 accurancy point
                hits += 1
        return hits / len(self.eval_set)

    def compare(self, strategies: dict, k=5):
        """
        strategies: {"naive": fn, "multiquery": fn, "reranked": fn}
        Returns {"naive": 0.87, "multiquery": 0.93, ...}
        """
        results = {}
        for name, fn in strategies.items():
            score = self.recall_at_k(fn, k=k)
            results[name] = score
            print(f"{name}: {score:.2f} ({int(score * len(self.eval_set))}/{len(self.eval_set)})")
        return results
