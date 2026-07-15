from openai import OpenAI

from . import config


class LLMClient:
    """Thin wrapper around the Groq-hosted OpenAI-compatible chat client."""

    def __init__(self, model=config.LLM_MODEL_NAME):
        self.model = model
        self.client = OpenAI(base_url=config.GROQ_BASE_URL, api_key=config.GROQ_API_KEY)

    def chat(self, system_prompt, user_prompt):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        resp = self.client.chat.completions.create(model=self.model, messages=messages)
        return resp.choices[0].message.content


class QueryExpander:
    """Generates related questions from an original query (multi-query expansion)."""

    SYSTEM_PROMPT = """
    You are a knowledgeable financial research assistant.
    Your users are inquiring about an annual report.
    For the given question, propose up to five related questions to assist them in finding the information they need.
    Provide concise, single-topic questions (without compounding sentences) that cover various aspects of the topic.
    Ensure each question is complete and directly related to the original inquiry.
    List each question on a separate line without numbering.
    """

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def expand(self, query):
        """Returns [original_query, generated_variant_1, ..., generated_variant_5]."""
        raw = self.llm_client.chat(self.SYSTEM_PROMPT, query)
        variants = [line.strip() for line in raw.split("\n") if line.strip()]
        return [query] + variants


class Generator:
    """Generates the final answer to a query, optionally grounded in retrieved context."""

    BASE_PROMPT = """You are a helpful expert financial research assistant.
        Provide an example answer to the given question, that might be found in a document like an annual report."""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def answer(self, query, context_chunks=None):
        prompt = self.BASE_PROMPT
        if context_chunks:
            context = "\n\n".join(context_chunks)
            prompt += "\n\nContext:\n" + context + "\n\nQuestion:\n" + query
        return self.llm_client.chat(prompt, query)

    def generate_question_from_chunk(self, chunk):
        """Reverse-direction generation used for synthetic eval sets: chunk -> question."""
        system_prompt = """Generate ONE specific factual question that is answered by this text passage.
        Return only the question, nothing else."""
        return self.llm_client.chat(system_prompt, chunk).strip()
