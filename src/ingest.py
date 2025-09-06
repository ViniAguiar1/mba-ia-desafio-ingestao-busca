# src/ingest.py
import os
from dotenv import load_dotenv

# 1) Carregadores/split/embeddings/vectorstore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_postgres import PGVector

def get_embeddings():
    """
    Escolhe o provedor de embeddings a partir do .env:
    - Se houver OPENAI_API_KEY, usa OpenAI.
    - Senão, se houver GOOGLE_API_KEY, usa Gemini.
    - (Você pode ajustar essa lógica depois. Para o desafio, OpenAI já atende.)
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")

    if openai_key:
        model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        return OpenAIEmbeddings(model=model, api_key=openai_key)
    elif google_key:
        model = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001")
        return GoogleGenerativeAIEmbeddings(model=model, google_api_key=google_key)
    else:
        raise RuntimeError("Nenhuma API key encontrada. Defina OPENAI_API_KEY ou GOOGLE_API_KEY no .env")

def main():
    load_dotenv()

    # 2) Ler configurações essenciais
    pdf_path = os.getenv("PDF_PATH", "./document.pdf")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF não encontrado em: {pdf_path}")

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL não definido no .env")

    collection = os.getenv("PG_VECTOR_COLLECTION_NAME", "pdf_chunks")

    # 3) Carregar PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()  # lista de Document (um por página por padrão)
    print(f"[ingest] Páginas carregadas: {len(docs)}")

    # 4) Split em chunks 1000/150 (requisito do desafio)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    print(f"[ingest] Chunks gerados: {len(chunks)}")

    # 5) Embeddings (OpenAI ou Gemini)
    embeddings = get_embeddings()

    # 6) Vector store apontando para Postgres/pgvector
    #    - connection = DATABASE_URL (estilo SQLAlchemy, ex.: postgresql+psycopg://...)
    #    - collection_name = nome lógico da “tabela”/coleção
    #    Obs: PGVector cria a tabela e colunas necessárias se não existirem.
    vs = PGVector(
        connection_string=database_url,
        collection_name=collection,
        embedding_function=embeddings,
        use_jsonb=True,  # metadados em jsonb (mais prático para consulta/depuração)
    )

    # 7) Persistir
    print("[ingest] Gravando embeddings no Postgres...")
    vs.add_documents(chunks)
    print("[ingest] Concluído! ✅")

if __name__ == "__main__":
    main()
