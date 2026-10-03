"""
RAG Engine.

Ingests contracts and policy documents into a local ChromaDB vector store,
using a local (free, no API key) embedding model. At query time, retrieves
the most relevant chunks and asks Groq to answer using ONLY those chunks,
with citations.

Why local embeddings instead of an API?
  - No extra API key, no rate limits, works offline after the model downloads.
  - sentence-transformers' all-MiniLM-L6-v2 is small (~80MB) and good enough
    for this scale of document set.

Why ChromaDB instead of Qdrant?
  - Runs embedded in Python, no Docker/server needed — one less thing that
    can go wrong during setup.
"""

import os
import glob
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = "openai/gpt-oss-120b"
TOP_K = 4  # how many chunks to retrieve per query
CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 150    # overlap between consecutive chunks

DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "data", "docs")
CHROMA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                            "data", "chroma_db")

client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

_embedder = None
_chroma_client = None
_collection = None


def get_embedder():
    global _embedder
    if _embedder is None:
        print("Loading embedding model (first time only, downloads ~80MB)...")
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def get_collection():
    global _chroma_client, _collection
    if _collection is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = _chroma_client.get_or_create_collection("documents")
    return _collection


def chunk_text(text: str, source: str) -> list[dict]:
    """Split text into overlapping chunks, keeping track of source filename."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        if chunk.strip():
            chunks.append({"text": chunk, "source": source})
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def ingest_documents():
    """Read all .txt files under data/docs, chunk them, embed them, and
    store them in ChromaDB. Safe to re-run — clears and rebuilds each time."""
    collection = get_collection()
    embedder = get_embedder()

    # clear existing collection for a clean rebuild
    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    all_files = glob.glob(f"{DOCS_DIR}/**/*.txt", recursive=True)
    if not all_files:
        print(f"No .txt files found under {DOCS_DIR}. Run generate_docs.py first.")
        return

    all_chunks = []
    for filepath in all_files:
        source = os.path.basename(filepath)
        with open(filepath, encoding="utf-8") as f:
            text = f.read()
        all_chunks.extend(chunk_text(text, source))

    print(f"Embedding {len(all_chunks)} chunks from {len(all_files)} documents...")
    texts = [c["text"] for c in all_chunks]
    embeddings = embedder.encode(texts, show_progress_bar=False).tolist()

    ids = [f"chunk_{i}" for i in range(len(all_chunks))]
    metadatas = [{"source": c["source"]} for c in all_chunks]

    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    print(f"Ingested {len(all_chunks)} chunks into ChromaDB at {CHROMA_PATH}")


def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    """Return the top_k most relevant chunks for the question."""
    collection = get_collection()
    embedder = get_embedder()

    query_embedding = embedder.encode([question]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({"text": doc, "source": meta["source"]})
    return chunks


ANSWER_SYSTEM_PROMPT = """You are a helpful assistant answering questions using ONLY the
provided document excerpts. Follow these rules strictly:
- Answer using only information in the excerpts below. Do not use outside knowledge.
- If the excerpts don't contain the answer, say so clearly — do not guess.
- Always cite which document(s) you used, by filename.
- Be concise and direct.
"""


def answer_question(question: str) -> dict:
    """Full RAG pipeline: retrieve relevant chunks, then generate an answer with citations."""
    chunks = retrieve(question)

    if not chunks:
        return {"question": question, "answer": "No documents found.", "sources": []}

    context = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in chunks
    )

    user_msg = f"Document excerpts:\n\n{context}\n\nQuestion: {question}"

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=500,
        temperature=0,
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    answer = response.choices[0].message.content.strip()
    sources = list(set(c["source"] for c in chunks))
    raw_context = "\n".join(c["text"] for c in chunks)

    return {"question": question, "answer": answer, "sources": sources, "raw_context": raw_context}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        ingest_documents()
    else:
        q = "What is the termination notice period for Bera and Sons?"
        result = answer_question(q)
        print(f"Question: {result['question']}")
        print(f"Answer: {result['answer']}")
        print(f"Sources: {result['sources']}")
