# =========================
# app.py 
# =========================
import streamlit as st
import tempfile
from pathlib import Path
from dotenv import load_dotenv
import os
from streamlit_pdf_viewer import pdf_viewer


load_dotenv()

# ---------------------------
# Imports
# ---------------------------
try:
    from extractor import PDFExtractor
    from chunker import chunk_texts
    from embed_store import init_vectorstore, add_chunks_to_vectorstore, get_vectorstore
    from retriever import Retriever
    from rag_chain import RAGChain
except ModuleNotFoundError as e:
    st.error(f"Module import failed: {e}")
    st.stop()



# ---------------------------
# Streamlit page setup
# ---------------------------
st.set_page_config(page_title='PDF RAG Chatbot', layout='wide')
st.title('📄 LLM-Powered PDF RAG Chatbot')

# ---------------------------
# Sidebar: PDF Upload
# ---------------------------
with st.sidebar:
    
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])
    
    if uploaded_file:
        pdf_name_extension = uploaded_file.name
        pdf_name = os.path.splitext(pdf_name_extension)[0].replace(" ", "")

        # --- Restart pipeline if a new PDF is uploaded ---
        if 'uploaded_pdf_name' in st.session_state:
            if st.session_state.uploaded_pdf_name != pdf_name_extension:
                for key in ['rag_chain', 'chat_history', 'vectorstore']:
                    if key in st.session_state:
                        del st.session_state[key]
        st.session_state.uploaded_pdf_name = pdf_name_extension
        # --------------------------------------------------

        # Save uploaded file to a temporary path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = Path(tmp_file.name)

        st.success("PDF uploaded successfully!")
        st.write(f"**File size:** {len(uploaded_file.getvalue()) / 1024:.2f} KB")

        # PDF Preview (sidebar)
        st.subheader("PDF Preview")
        pdf_viewer(tmp_path, height=400)

# ---------------------------
# Main: Chat Interface
# ---------------------------
st.subheader("Chat with your PDF")

if uploaded_file:
    if 'rag_chain' not in st.session_state:
        st.info("Processing PDF...")

        try:
            # Extract PDF content
            pdf_extractor = PDFExtractor()
            pdf_data = pdf_extractor.extract(tmp_path)
            st.success("Extraction completed!")

            # Chunking
            chunks = chunk_texts(pdf_data)
            st.info(f"Created {len(chunks)} chunks")

            # Embedding store (Chroma)
            vector_store = init_vectorstore(persist_dir="faiss", collection_name=pdf_name)
            add_chunks_to_vectorstore(vector_store, chunks)

            # Debug: check if vectorstore exists
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Vectorstore failed to initialize!")
                st.stop()
            
            st.success("Embeddings created successfully!")

            # Retriever + RAGChain
            retriever = Retriever(vectorstore)
            st.session_state.rag_chain = RAGChain(retriever)
            st.session_state.chat_history = []
            
            st.success("RAG system ready!")

        except Exception as e:
            st.error(f"Error during setup: {e}")
            st.stop()

    # ---------------------------
    # Chat input
    # ---------------------------
    user_input = st.text_input("Your question:", key="user_input")
    if user_input:
        try:
            rag_chain = st.session_state.rag_chain
            answer = rag_chain.query(user_input)

            st.session_state.chat_history.append({"user": user_input, "bot": answer})
        except Exception as e:
            st.error(f"Error during query: {e}")

    # ---------------------------
    # Display chat history
    # ---------------------------
    for chat in st.session_state.chat_history:
        st.markdown(f"**You:** {chat['user']}")
        st.markdown(f"**Bot:** {chat['bot']}")
