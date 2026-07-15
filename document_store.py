import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

from . import config


def word_wrap(text, width=87):
    """Wraps the given text to the specified width (used for readable printing)."""
    return "\n".join([text[i: i + width] for i in range(0, len(text), width)])


class DocumentStore:
    """
    Owns the full ingestion pipeline: PDF -> parsed text -> chunks -> vector DB.
    Also owns raw querying against the vector DB (bi-encoder similarity search).
    Reranking and generation are deliberately NOT this class's job (see Reranker/Generator).
    """

    def __init__(
        self,
        pdf_path=config.PDF_PATH,
        persist_path=config.CHROMA_PERSIST_PATH,
        collection_name=config.COLLECTION_NAME,
    ):
        self.pdf_path = pdf_path
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.EMBEDDING_MODEL_NAME
        )
        self.chroma_client = chromadb.PersistentClient(path=persist_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name, embedding_function=self.embedding_fn
        )
        self.chunks = []

    def _load_pdf_text(self):
        reader = PdfReader(self.pdf_path)
        pdf_texts = [p.extract_text().strip() for p in reader.pages]
        return [text for text in pdf_texts if text]

    def _chunk_text(self, pdf_texts):
        char_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        )
        char_split_texts = char_splitter.split_text("\n\n".join(pdf_texts))

        token_splitter = SentenceTransformersTokenTextSplitter(
            chunk_overlap=0, tokens_per_chunk=config.TOKENS_PER_CHUNK
        )
        token_split_texts = []
        for text in char_split_texts:
            token_split_texts += token_splitter.split_text(text)
        return token_split_texts

    def build_index(self):
        """Parses, chunks, and indexes the PDF -- but only if not already indexed."""
        pdf_texts = self._load_pdf_text()
        self.chunks = self._chunk_text(pdf_texts)

        if self.collection.count() == 0:
            ids = [str(i) for i in range(len(self.chunks))]
            self.collection.add(ids=ids, documents=self.chunks)

        return self

    def query(self, questions, n_results=5):
        """
        Runs vector similarity search for one or more query strings.
        Always returns a flat list of chunk texts across all input questions
        (fixes the original bug of only reading res["documents"][0]).
        """
        if isinstance(questions, str):
            questions = [questions]

        res = self.collection.query(query_texts=questions, n_results=n_results)
        return [doc for sublist in res["documents"] for doc in sublist]

    def count(self):
        return self.collection.count()
