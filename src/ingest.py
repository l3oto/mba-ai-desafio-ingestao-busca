import os
import time
import random
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH", "document.pdf")
DATABASE_URL = os.getenv("DATABASE_URL")
COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME", "documents")
EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001")
BATCH_SIZE = 5
BATCH_DELAY = 2  # segundos entre lotes (respeita rate limit do tier gratuito)


def ingest_pdf():
    print(f"Carregando PDF: {PDF_PATH}")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    print(f"  {len(documents)} página(s) carregada(s)")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    print(f"  {len(chunks)} chunks gerados")

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    # Cria a store vazia (apaga coleção anterior)
    store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        pre_delete_collection=True,
    )

    total = len(chunks)
    print(f"Indexando {total} chunks em lotes de {BATCH_SIZE}...")
    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        _add_with_retry(store, batch)
        print(f"  {min(i + BATCH_SIZE, total)}/{total} chunks indexados")
        if i + BATCH_SIZE < total:
            time.sleep(BATCH_DELAY)


def _add_with_retry(store, batch, max_retries=5):
    for attempt in range(max_retries):
        try:
            store.add_documents(batch)
            return
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = (2 ** attempt) + random.uniform(0, 1)
                print(f"  Rate limit — aguardando {wait:.1f}s (tentativa {attempt + 1}/{max_retries})")
                time.sleep(wait)
            else:
                raise

    print("Ingestão concluída.")


if __name__ == "__main__":
    ingest_pdf()