
# Desafio MBA Engenharia de Software com IA - Full Cycle

Este projeto implementa **ingestão e busca semântica** usando **LangChain** + **PostgreSQL (pgVector)** para responder perguntas com base no conteúdo de um PDF.

---

## Tecnologias

- **Python 3.10+**
- **LangChain**
- **PostgreSQL 17 + pgVector**
- **Docker + Docker Compose**
- **Embeddings**:
  - Local (HuggingFace - *sentence-transformers/all-MiniLM-L6-v2*)
  - OpenAI (opcional)
  - Google Gemini (opcional)

---

## Estrutura do Projeto

```
├── docker-compose.yml      # Banco Postgres + pgVector
├── requirements.txt        # Dependências Python
├── .env.example            # Template de variáveis de ambiente
├── document.pdf            # PDF para ingestão
└── src/
    ├── ingest.py           # Ingestão do PDF → Postgres
    ├── search.py           # Busca semântica e montagem do prompt
    └── chat.py             # CLI interativo (perguntas/respostas)
```

---

## 1. Clonar o projeto e entrar na pasta
```bash
git clone https://github.com/<SEU_USUARIO>/mba-ia-desafio-ingestao-busca.git
cd mba-ia-desafio-ingestao-busca
```

---

## 2. Criar e ativar ambiente virtual
```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Instalar dependências
```bash
pip install -r requirements.txt
```

---

## 4. Configurar o `.env`

Copie o `.env.example` para `.env`:
```bash
cp .env.example .env
```

Preencha as variáveis necessárias:

| Variável                    | Descrição                                        | Exemplo                                            |
|-----------------------------|--------------------------------------------------|--------------------------------------------------|
| `DATABASE_URL`               | URL do Postgres com pgVector                     | `postgresql+psycopg://postgres:postgres@localhost:5432/rag` |
| `PG_VECTOR_COLLECTION_NAME`  | Nome da coleção no pgVector                      | `pdf_chunks_local`                                 |
| `PDF_PATH`                   | Caminho do PDF a ser ingerido                    | `./document.pdf`                                   |
| `EMBEDDING_PROVIDER`         | `local`, `openai` ou `gemini`                    | `local`                                            |
| `LOCAL_EMBEDDING_MODEL`       | Modelo local HuggingFace (se usar local)          | `sentence-transformers/all-MiniLM-L6-v2`          |
| `OPENAI_API_KEY` (opcional)   | Chave da OpenAI (se usar OpenAI)                  | `sk-...`                                           |
| `GOOGLE_API_KEY` (opcional)   | Chave do Gemini (se usar Gemini)                  | `AIza...`                                          |

---

## 5. Subir o banco de dados (Docker)

```bash
docker compose up -d
```

Crie a extensão pgVector:
```bash
docker exec -it <NOME_DO_CONTAINER> psql -U postgres -d rag -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

---

## 6. Ingestão do PDF → Vetores no Postgres
```bash
python src/ingest.py
```

Saída esperada:
```
[ingest] Páginas carregadas: 34
[ingest] Chunks gerados: 67
[ingest] Gravando embeddings no Postgres...
[ingest] Concluído! ✅
```

---

## 7. Busca semântica isolada (debug)
```bash
PYTHONPATH=src python - << 'PY'
from search import search_prompt
p = search_prompt("Qual o faturamento da Empresa SuperTechIABrazil?", k=3, debug=True)
print(p)
PY
```

---

## 8. CLI interativo (chat)
```bash
python src/chat.py
```

Exemplo:
```
Faça sua pergunta:
> Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: O faturamento foi de 10 milhões de reais.
```

---

## 9. Estrutura esperada no Postgres

Após ingestão:
```sql
\d langchain_pg_collection;
\d langchain_pg_embedding;
```

Para contar chunks:
```sql
SELECT COUNT(*) 
FROM langchain_pg_embedding e
JOIN langchain_pg_collection c ON c.uuid = e.collection_id
WHERE c.name = 'pdf_chunks_local';
```

---

## 10. Trocar de provedor de embeddings

Para usar OpenAI ou Gemini:
1. Adicione sua chave no `.env`
2. Altere `EMBEDDING_PROVIDER=openai` ou `gemini`
3. Re-ingira o PDF:
   ```bash
   python src/ingest.py
   ```

---

## Entregável Final

- Código no GitHub
- README com passo a passo (este arquivo)
- Scripts:
  - `ingest.py` → ingestão
  - `search.py` → busca semântica
  - `chat.py` → CLI perguntas/respostas

---
