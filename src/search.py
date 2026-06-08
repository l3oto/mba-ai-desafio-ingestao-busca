import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_postgres import PGVector

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME", "documents")
EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001")
LLM_MODEL = os.getenv("GOOGLE_LLM_MODEL", "gemini-2.5-flash-lite")

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def search_prompt():
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        store = PGVector(
            embeddings=embeddings,
            collection_name=COLLECTION_NAME,
            connection=DATABASE_URL,
        )
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL)
    except Exception as e:
        print(f"Erro ao inicializar: {e}")
        return None

    def chain(question: str) -> str:
        results = store.similarity_search_with_score(question, k=10)
        context = "\n\n".join(doc.page_content for doc, _ in results)
        prompt = PROMPT_TEMPLATE.format(contexto=context, pergunta=question)
        response = llm.invoke(prompt)
        return response.content

    return chain