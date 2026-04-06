import streamlit as st
import json
import os
from pathlib import Path
from PyPDF2 import PdfReader
import io
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# ------------------------------
# Load API key from .env file
# ------------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Google API key not found. Please set GOOGLE_API_KEY in a .env file.")
    st.stop()

# ------------------------------
# Page config & CSS
# ------------------------------
st.set_page_config(page_title="CareerCompass AI", page_icon="🎯", layout="wide")

def load_css():
    css_file = Path(__file__).parent / "styles.css"
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css()

# ------------------------------
# Initialize embedding model (cached)
# ------------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def load_vector_store():
    embeddings = load_embeddings()
    index_path = Path(__file__).parent / "faiss_index"
    if index_path.exists():
        return FAISS.load_local(str(index_path), embeddings, allow_dangerous_deserialization=True)
    else:
        st.error("FAISS index not found. Run scripts/setup_vector_store.py first.")
        st.stop()

# ------------------------------
# Helper functions
# ------------------------------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text if text.strip() else "No text extracted."
    except Exception as e:
        return f"Error: {str(e)}"

def extract_skills_from_resume(resume_text: str) -> dict:
    # Use a valid Gemini model name
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-001",   # changed from gemini-1.5-flash
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3
    )
    prompt = f"""
    Analyze the following resume and extract key information.
    Return ONLY a JSON object with these exact keys: "skills", "experience", "education".
    "skills" should be a list of technical and soft skills.
    "experience" should be a list of previous roles/positions.
    "education" should be a list of degrees and institutions.
    
    Resume Text:
    {resume_text[:4000]}
    """
    try:
        response = llm.invoke(prompt)
        content = response.content
        # Clean up markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        return json.loads(content)
    except Exception as e:
        st.warning(f"Skill extraction failed: {e}")
        return {"skills": [], "experience": [], "education": []}

def analyze_skill_gap(user_skills: list, target_role: str, vector_store) -> dict:
    docs = vector_store.similarity_search(target_role, k=1)
    if not docs:
        return {"match_score": 0, "missing_skills": [], "analysis": "Role not found."}
    role_info = docs[0].page_content
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-001",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3
    )
    prompt = f"""
    Compare the user's skills with the target role requirements.
    
    User Skills: {', '.join(user_skills)}
    
    Role Requirements: {role_info}
    
    Return ONLY a JSON object with:
    - "match_score": a number from 0 to 100
    - "missing_skills": list of important missing skills
    - "analysis": a brief explanation
    """
    try:
        response = llm.invoke(prompt)
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        return json.loads(content)
    except Exception as e:
        st.warning(f"Gap analysis failed: {e}")
        return {"match_score": 50, "missing_skills": [], "analysis": "Analysis completed."}

def generate_roadmap(user_skills: list, missing_skills: list, target_role: str, vector_store) -> str:
    docs = vector_store.similarity_search(target_role, k=1)
    role_context = docs[0].page_content if docs else ""
    resources_path = Path(__file__).parent / "data" / "learning_resources.json"
    try:
        with open(resources_path, 'r') as f:
            resources = json.load(f)
    except:
        resources = {"courses": {}}
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-001",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.5
    )
    prompt = f"""
    You are a career advisor. Generate a personalized learning roadmap.
    
    User's Current Skills: {', '.join(user_skills)}
    Skills to Acquire: {', '.join(missing_skills)}
    Target Role: {target_role}
    
    Role Requirements Context: {role_context}
    
    Available Learning Resources: {json.dumps(resources, indent=2)}
    
    Provide a roadmap with:
    1. Brief summary
    2. Recommended learning path (specific courses/resources)
    3. Suggested certifications
    4. Timeline (3, 6, 12 months)
    
    Format as Markdown.
    """
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Roadmap generation failed: {e}"

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🎯 CareerCompass AI")
st.markdown("*Bridge the gap between your skills and your dream job*")
st.markdown("---")

with st.sidebar:
    st.header("📌 How it works")
    st.markdown("""
    1. Upload your resume (PDF)
    2. Select your target job role
    3. Get your **match score**, **skill gaps**, and a **personalized roadmap**
    """)
    st.markdown("---")
    st.caption("Powered by Google Gemini AI & RAG")

# Load roles from dataset
roles_path = Path(__file__).parent / "data" / "role_requirements.json"
if not roles_path.exists():
    st.error("role_requirements.json not found in data/ folder.")
    st.stop()
with open(roles_path, 'r') as f:
    role_data = json.load(f)
    roles = list(role_data.keys())

col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Upload Resume")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    selected_role = st.selectbox("Choose your target job role", roles)

with col2:
    st.subheader("📊 Analysis Results")
    analyze_button = st.button("🚀 Analyze My Career Path", type="primary", use_container_width=True)

if analyze_button:
    if not uploaded_file:
        st.error("Please upload your resume first.")
    else:
        with st.spinner("Analyzing your resume and generating roadmap..."):
            # Extract text
            resume_text = extract_text_from_pdf(uploaded_file.getvalue())
            # Extract skills
            extracted = extract_skills_from_resume(resume_text)
            user_skills = extracted.get("skills", [])
            # Load vector store
            vector_store = load_vector_store()
            # Analyze gap
            gap = analyze_skill_gap(user_skills, selected_role, vector_store)
            # Generate roadmap
            roadmap = generate_roadmap(user_skills, gap.get("missing_skills", []), selected_role, vector_store)
            # Display results
            st.markdown(f"### 🎯 Match Score: {gap.get('match_score', 0)}%")
            st.progress(gap.get('match_score', 0) / 100)
            st.markdown("### 📉 Skill Gaps Identified")
            missing = gap.get("missing_skills", [])
            if missing:
                for skill in missing:
                    st.warning(f"• {skill}")
            else:
                st.success("No major skill gaps! You're well-aligned for this role.")
            st.markdown("### 🗺️ Your Personalized Roadmap")
            st.markdown(roadmap)
            st.caption("⚠️ Disclaimer: AI-generated guidance only. Always cross-check with official sources.")