import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME", "documents")
EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001")
LLM_MODEL = os.getenv("GOOGLE_LLM_MODEL", "gemini-2.5-flash-lite")
PDF_PATH = os.getenv("PDF_PATH", "document.pdf")

if not DATABASE_URL:
    raise EnvironmentError("DATABASE_URL não definida — verifique o arquivo .env")
