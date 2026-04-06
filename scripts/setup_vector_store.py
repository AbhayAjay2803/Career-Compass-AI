import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.vector_store import create_vector_store

if __name__ == "__main__":
    print("Creating FAISS vector store...")
    create_vector_store()
    print("Done. Index saved to 'faiss_index' folder.")