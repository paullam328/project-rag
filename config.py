import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

PDF_PATH = "pdfs/microsoft-annual-report.pdf"
CHROMA_PERSIST_PATH = "chroma_persistent_expansion_storage"
COLLECTION_NAME = "my-collection"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL_NAME = "llama-3.3-70b-versatile"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 0
TOKENS_PER_CHUNK = 256
