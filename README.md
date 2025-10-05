# 🧩 Desafio MBA Engenharia de Software com IA — Full Cycle

Este projeto implementa **ingestão e busca semântica (RAG)** usando **LangChain** + **PostgreSQL (pgVector)** para responder perguntas com base no conteúdo de um **PDF**.

---

## 🚀 Tecnologias Utilizadas

- **Python 3.10+**
- **LangChain**
- **PostgreSQL 17 + pgVector**
- **Docker + Docker Compose**
- **Embeddings**:
  - Local (HuggingFace - *sentence-transformers/all-MiniLM-L6-v2*)
  - OpenAI (opcional)
  - Google Gemini (opcional)

---

## 🧱 Estrutura do Projeto

```bash
├── docker-compose.yml       # Banco Postgres + pgVector
├── requirements.txt         # Dependências Python
├── .env.example             # Template das variáveis de ambiente
├── document.pdf             # Documento de entrada (PDF)
└── src/
    ├── ingest.py            # Ingestão do PDF → Postgres (embeddings)
    ├── search.py            # Busca semântica e montagem de prompt
    └── chat.py              # CLI interativo para perguntas/respostas
```

---

## ⚙️ 1. Clonar o projeto e acessar a pasta

```bash
git clone https://github.com/ViniAguiar1/mba-ia-desafio-ingestao-busca.git
cd mba-ia-desafio-ingestao-busca
```

---

## 🧩 2. Criar e ativar ambiente virtual

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 📦 3. Instalar dependências

```bash
pip install -r requirements.txt
```

---

## ⚙️ 4. Configurar o `.env`

Copie o exemplo e edite:

```bash
cp .env.example .env
```

Preencha com suas variáveis:

| Variável | Descrição | Exemplo |
|-----------|------------|----------|
| `DATABASE_URL` | URL do banco com pgVector | `postgresql+psycopg://postgres:postgres@localhost:5432/rag` |
| `PG_VECTOR_COLLECTION_NAME` | Nome da coleção de vetores | `pdf_chunks_local` |
| `PDF_PATH` | Caminho do PDF | `./document.pdf` |
| `EMBEDDING_PROVIDER` | `local`, `openai` ou `gemini` | `local` |
| `LOCAL_EMBEDDING_MODEL` | Modelo local HuggingFace | `sentence-transformers/all-MiniLM-L6-v2` |
| `OPENAI_API_KEY` | Chave da OpenAI | `sk-...` |
| `GOOGLE_API_KEY` | Chave do Gemini | `AIza...` |

> 💡 **Dica:**  
> Use `EMBEDDING_PROVIDER=openai` para usar o OpenAI,  
> ou `EMBEDDING_PROVIDER=local` para rodar 100% offline.

---

## 🐘 5. Subir o banco PostgreSQL com pgVector

```bash
docker compose up -d
```

Crie a extensão pgVector:

```bash
docker exec -it <NOME_DO_CONTAINER> psql -U postgres -d rag -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

---

## 📥 6. Ingestão do PDF (criação dos embeddings)

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

## 🔍 7. Busca semântica isolada (debug/teste)

```bash
PYTHONPATH=src python - << 'PY'
from search import search_prompt
p = search_prompt("Qual o faturamento da Empresa SuperTechIABrazil?", k=3, debug=True)
print(p)
PY
```

Exemplo de saída:
```
[debug] top-k resultados:
#1 score=0.47 preview='SuperTechIABrazil R$ 10.000.000,00 2025'
...
```

---

## 💬 8. CLI interativo (chat de perguntas)

```bash
python src/chat.py
```

### Exemplo:

```
Digite sua pergunta (ou 'sair' para encerrar).

PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: O faturamento foi de R$ 10.000.000,00.
```

---

## 🧾 9. Estrutura esperada no Postgres

Verifique as tabelas criadas:

```sql
\d langchain_pg_collection;
\d langchain_pg_embedding;
```

E conte quantos chunks foram armazenados:

```sql
SELECT COUNT(*)
FROM langchain_pg_embedding e
JOIN langchain_pg_collection c ON c.uuid = e.collection_id
WHERE c.name = 'pdf_chunks_local';
```

---

## 🧠 10. Perguntas de exemplo

✅ **Dentro do contexto (responde com base no PDF):**
- Qual o faturamento da Empresa SuperTechIABrazil?
- Quando foi fundada a Empresa Beta Energia Indústria?
- Quais empresas têm faturamento acima de R$ 1.000.000,00?
- Quais empresas foram criadas depois de 2000?

🚫 **Fora do contexto (resposta padrão):**
- Qual é a capital da França?
- Você acha que o faturamento está bom?
- Quantos clientes temos em 2024?

---

## 🔄 11. Trocar de provedor de embeddings

Para mudar para **OpenAI** ou **Gemini**:
1. Adicione sua chave no `.env`
2. Altere a variável:
   ```bash
   EMBEDDING_PROVIDER=openai
   ```
   ou  
   ```bash
   EMBEDDING_PROVIDER=gemini
   ```
3. Reexecute a ingestão:
   ```bash
   python src/ingest.py
   ```

---

## 📦 Entregável Final

✅ **Repositório GitHub** contendo:
- Código completo e comentado
- `README.md` atualizado (este arquivo)
- Scripts:
  - `ingest.py` → ingestão de PDF
  - `search.py` → busca vetorial + prompt
  - `chat.py` → interface de perguntas e respostas
- Banco PostgreSQL com vetores armazenados
- PDF de teste (`document.pdf`)

---

## 🧠 Autor

**Vinicius Aguiar**  
MBA em Engenharia de Software com IA — Full Cycle  
💻 [github.com/ViniAguiar1](https://github.com/ViniAguiar1)

---
