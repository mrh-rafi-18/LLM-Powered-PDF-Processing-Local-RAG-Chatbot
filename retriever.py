# =========================
# retriever.py
# =========================
from langchain.vectorstores import FAISS

class Retriever:
    def __init__(self, vectorstore: FAISS):
        if vectorstore is None:
            raise ValueError("Vectorstore cannot be None")
        self.vectorstore = vectorstore

    def retrieve(self, query: str, k: int = 5):
        """
        Retrieve top-k relevant documents from FAISS vectorstore.
        """
        if self.vectorstore is None:
            raise ValueError("Vectorstore is not initialized")
        
        # Enable MMR in the retriever
        retriever = self.vectorstore.as_retriever(
        search_type="mmr",                   
        search_kwargs={"k": k, "lambda_mult": 0.5}
        )
        
        return retriever.invoke(query)
