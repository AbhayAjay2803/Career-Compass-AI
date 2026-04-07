import streamlit as st
import json
import os
import re
from pathlib import Path
from PyPDF2 import PdfReader
import io
import warnings
warnings.filterwarnings('ignore')

# ------------------------------
# Load API key from secrets or .env
# ------------------------------
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except Exception:
    from dotenv import load_dotenv
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

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
# Cached resources: embeddings, vector store
# ------------------------------
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

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
        # Create index from role_requirements.json
        with open(Path(__file__).parent / "data" / "role_requirements.json", 'r') as f:
            role_data = json.load(f)
        documents = []
        for role, details in role_data.items():
            content = f"Role: {role}\nRequired Skills: {', '.join(details['skills'])}\nCertifications: {', '.join(details['certifications'])}\nLearning Path: {details['learning_path']}"
            documents.append(Document(page_content=content, metadata={"role": role}))
        vector_store = FAISS.from_documents(documents, embeddings)
        vector_store.save_local("faiss_index")
        return vector_store

# ------------------------------
# Gemini LLM (with error handling)
# ------------------------------
def get_gemini_llm():
    if not GOOGLE_API_KEY:
        return None
    try:
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3
        )
    except:
        return None

# ------------------------------
# Rule‑based skill extraction (fallback)
# ------------------------------
def extract_skills_rule_based(text: str) -> dict:
    """Extract skills using keyword matching."""
    common_skills = [
        "python", "java", "c++", "javascript", "html", "css", "react", "angular", "vue",
        "sql", "mongodb", "postgresql", "mysql", "tensorflow", "pytorch", "scikit-learn",
        "pandas", "numpy", "data analysis", "machine learning", "deep learning", "nlp",
        "aws", "azure", "gcp", "docker", "kubernetes", "git", "agile", "scrum",
        "project management", "leadership", "communication", "teamwork", "problem solving"
    ]
    text_lower = text.lower()
    found = [skill for skill in common_skills if skill in text_lower]
    # Also try to extract from "skills" section if present
    skills_section = re.search(r'skills?:?\s*(.*?)(?:\n\n|\n[A-Z]|$)', text, re.IGNORECASE | re.DOTALL)
    if skills_section:
        section_text = skills_section.group(1)
        words = re.findall(r'\b[a-zA-Z\+#][a-zA-Z0-9\+\.#]{1,}\b', section_text)
        for w in words:
            if len(w) > 2 and w.lower() not in found:
                found.append(w.lower())
    return {"skills": list(set(found)), "experience": [], "education": []}

# ------------------------------
# Helper functions with fallback
# ------------------------------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text if text.strip() else ""
    except:
        return ""

def extract_skills_with_fallback(resume_text: str):
    """Try Gemini first, then rule‑based."""
    llm = get_gemini_llm()
    if llm and GOOGLE_API_KEY:
        prompt = f"""
        Analyze the resume. Return ONLY JSON: {{"skills": [...], "experience": [...], "education": [...]}}.
        Resume: {resume_text[:4000]}
        """
        try:
            response = llm.invoke(prompt)
            content = response.content
            # Clean markdown
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            data = json.loads(content)
            return data, "Gemini"
        except Exception as e:
            st.warning(f"Gemini failed: {e}. Using rule‑based extraction.")
    # Fallback to rule‑based
    return extract_skills_rule_based(resume_text), "Rule‑based"

def analyze_skill_gap_fallback(user_skills, target_role, vector_store):
    """Rule‑based skill gap analysis (same as before)."""
    docs = vector_store.similarity_search(target_role, k=1)
    if not docs:
        return {"match_score": 0, "missing_skills": [], "analysis": "Role not found."}
    role_info = docs[0].page_content
    required_skills_match = re.search(r'Required Skills: (.*?)(?:\n|$)', role_info)
    if required_skills_match:
        required_skills = [s.strip() for s in required_skills_match.group(1).split(',')]
    else:
        required_skills = []
    user_skills_lower = [s.lower() for s in user_skills]
    missing = []
    for skill in required_skills:
        skill_lower = skill.lower()
        if not any(us in skill_lower or skill_lower in us for us in user_skills_lower):
            missing.append(skill)
    if required_skills:
        matched = sum(1 for s in required_skills if any(us in s.lower() or s.lower() in us for us in user_skills_lower))
        score = int((matched / len(required_skills)) * 100)
    else:
        score = 50
    return {"match_score": score, "missing_skills": missing, "analysis": f"Matched {score}% of required skills."}

def generate_roadmap_fallback(missing_skills, target_role):
    """Static roadmap generator."""
    if not missing_skills:
        return f"""
## Your Career Path to {target_role}

Great news! You already have the key skills needed for this role. 

### Recommended Next Steps:
1. Build a portfolio project showcasing your existing skills
2. Network with professionals in the {target_role} field
3. Prepare for interviews by practicing common questions
4. Apply for entry-level positions or internships

### Estimated Timeline: 1-3 months
"""
    roadmap = f"""
## Your Personalized Roadmap to {target_role}

### Skills to Focus On:
"""
    for skill in missing_skills[:5]:
        roadmap += f"- **{skill}**\n"
    roadmap += f"""
### Learning Path:

1. **Master the Fundamentals** – focus on {', '.join(missing_skills[:2])}
2. **Build Practical Experience** – work on projects using these skills
3. **Get Certified** – look for certifications in {target_role}

### Suggested Timeline:
- **3 months**: Complete foundational courses
- **6 months**: Build 2-3 portfolio projects
- **12 months**: Apply for roles and continue advanced learning

### Recommended Resources:
- Coursera, Udemy, NPTEL for online courses
- GitHub for open‑source projects
- LinkedIn Learning for professional skills
"""
    return roadmap

# ------------------------------
# Main Streamlit UI
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
    st.caption("Powered by Gemini AI (fallback to rule‑based when quota exceeded)")

# Load roles
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
        with st.spinner("Processing your resume..."):
            # Extract text
            resume_text = extract_text_from_pdf(uploaded_file.getvalue())
            if not resume_text:
                st.error("Could not extract text from PDF. Please ensure it contains selectable text.")
                st.stop()
            
            # Step 1: Extract skills
            skills_data, skills_model = extract_skills_with_fallback(resume_text)
            user_skills = skills_data.get("skills", [])
            
            # Step 2: Load vector store and analyze gap
            vector_store = load_vector_store()
            gap = analyze_skill_gap_fallback(user_skills, selected_role, vector_store)
            
            # Step 3: Generate roadmap
            roadmap = generate_roadmap_fallback(gap.get("missing_skills", []), selected_role)
            
            # Show which model was used
            st.info(f"🧠 Skill extraction: **{skills_model}** | Gap analysis & roadmap: **Rule‑based**")
            
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
            
            st.markdown("### 🗺️ Your Personalized Career Roadmap")
            st.markdown(roadmap)
            
            st.caption("⚠️ Disclaimer: AI-generated guidance only. Always cross-check with official sources.")