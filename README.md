# Desafio MBA Engenharia de Software com IA - Full Cycle

Ingestão e Busca Semântica com LangChain e PostgreSQL + pgVector.

## Tecnologias

- Python 3 + LangChain
- PostgreSQL com pgVector (via Docker)
- Google Gemini (embeddings + LLM)

## Configuração

1. Copie o arquivo de variáveis de ambiente:
```bash
cp .env.example .env
```

2. Preencha `GOOGLE_API_KEY` no `.env` com sua chave do [Google AI Studio](https://aistudio.google.com/apikey).

3. Crie e ative o ambiente virtual:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Execução

1. Suba o banco de dados:
```bash
docker compose up -d
```

2. Coloque o PDF desejado na raiz do projeto como `document.pdf` e execute a ingestão:
```bash
python src/ingest.py
```

3. Inicie o chat:
```bash
python src/chat.py
```

### Exemplo
```
PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: O faturamento foi de 10 milhões de reais.

PERGUNTA: Qual a capital da França?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```