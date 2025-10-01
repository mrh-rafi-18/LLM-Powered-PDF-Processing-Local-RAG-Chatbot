# =========================
# chunker.py 
# =========================
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
id=1
def chunk_texts(doc):
    """
    Chunk a Docling-loaded document using RecursiveCharacterTextSplitter.
    Returns a list of chunks in the format:
    [{'chunk_id': str, 'text': str}, ...]
    
    Logic:
    - Skip empty text
    - Keep texts <= chunk_size*2 as a single chunk
    - Split texts >= chunk_size*2 into chunks of size chunk_size with overlap
    """
    global id
    chunks = []
    chunker = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    for idx, page in enumerate(doc,start=1):
        text = page["structured_content"]
        if not text:
            continue

      
        if len(text) < CHUNK_SIZE :
            # Keep as single moderate chunk
            chunks.append({'chunk_id': f"{id}","page_no": f"{idx}", 'text': text})
            id=id+1
        else:
            # Split large text into smaller chunks
            split_texts = chunker.split_text(text)
            for i, t in enumerate(split_texts):
                chunks.append({'chunk_id':f"{id}" ,"page_no": f"{idx}", 'text': t})
                id=id+1

    return chunks
