# =========================
# embed_store
# =========================
import faiss
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
from typing import List, Dict
from dotenv import load_dotenv
import os
load_dotenv()

vs = None

# Initialize the embeddings model once
embeddings_model = HuggingFaceEmbeddings(
    model_name='intfloat/multilingual-e5-small'
)

def init_vectorstore(persist_dir: str = "faiss_db", collection_name: str = "pdf_chunks") -> FAISS:
    """
    Initialize and return a FAISS vectorstore instance.
    If an index exists on disk, it will load it.
    Otherwise, create a new empty FAISS index safely.
    """
    global vs

    index_path = os.path.join(persist_dir, f"{collection_name}.faiss")
    os.makedirs(persist_dir, exist_ok=True)

    # Load existing FAISS index if available
    if os.path.exists(index_path):
        vectorstore = FAISS.load_local(index_path, embeddings_model)
    else:
        # Create empty FAISS index manually
        embedding_dim = len(embeddings_model.embed_query("dummy"))
        index = faiss.IndexFlatL2(embedding_dim)
        vectorstore = FAISS(
            embedding_function=embeddings_model,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )

    vs = vectorstore
    return vectorstore

def add_chunks_to_vectorstore(vectorstore: FAISS, chunks: List[Dict]):
    """
    Add text chunks to a FAISS vectorstore.
    """
    # Filter out empty chunks
    chunks = [c for c in chunks if c['text'].strip()]
    if not chunks:
        return

    # Convert to Documents
    docs = [Document(page_content=c['text'], metadata={'chunk_id': c['chunk_id']}) for c in chunks]

    # Add to FAISS vectorstore
    vectorstore.add_documents(docs)


def get_vectorstore() -> FAISS:
    """
    Return the FAISS vectorstore instance.
    """
    if vs is None:
        raise ValueError("Vectorstore not initialized. Call init_vectorstore() first.")
    return vs
