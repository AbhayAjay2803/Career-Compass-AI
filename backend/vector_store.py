import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from backend.config import GOOGLE_API_KEY
import json

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

def create_vector_store():
    with open('data/role_requirements.json', 'r') as f:
        role_data = json.load(f)
    documents = []
    for role, details in role_data.items():
        content = f"""
        Role: {role}
        Required Skills: {', '.join(details['skills'])}
        Certifications: {', '.join(details['certifications'])}
        Learning Path Beginner: {', '.join(details['learning_path']['beginner'])}
        Learning Path Intermediate: {', '.join(details['learning_path']['intermediate'])}
        Learning Path Advanced: {', '.join(details['learning_path']['advanced'])}
        """
        documents.append(Document(page_content=content, metadata={"role": role}))
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_vector_store():
    if os.path.exists("faiss_index"):
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    else:
        return create_vector_store()