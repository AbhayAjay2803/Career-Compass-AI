import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings   # new import
from langchain_core.documents import Document
import json

# Initialize the sentence-transformers model (no API key needed!)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def create_vector_store():
    with open('data/role_requirements.json', 'r') as f:
        role_data = json.load(f)
    
    documents = []
    for role, details in role_data.items():
        content = f"""
        Role: {role}
        Required Skills: {', '.join(details['skills'])}
        Recommended Certifications: {', '.join(details['certifications'])}
        Learning Path:
        - Beginner: {', '.join(details['learning_path']['beginner'])}
        - Intermediate: {', '.join(details['learning_path']['intermediate'])}
        - Advanced: {', '.join(details['learning_path']['advanced'])}
        """
        metadata = {"role": role}
        documents.append(Document(page_content=content, metadata=metadata))
    
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_vector_store():
    if os.path.exists("faiss_index"):
        print("Loading existing FAISS index...")
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    else:
        print("Creating new FAISS index...")
        return create_vector_store()