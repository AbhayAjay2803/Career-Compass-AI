import streamlit as st
import requests
import json
import time
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="CareerCompass AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load custom CSS
def load_css():
    css_file = Path(__file__).parent / "styles.css"
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Backend API URL (change if deployed elsewhere)
API_BASE_URL = "http://localhost:8000"

# Title and description
st.title("🎯 CareerCompass AI")
st.markdown("*Bridge the gap between your skills and your dream job*")
st.markdown("---")

# Sidebar for instructions
with st.sidebar:
    st.header("📌 How it works")
    st.markdown("""
    1. Upload your resume (PDF)
    2. Select your target job role
    3. Get your **match score**, **skill gaps**, and a **personalized roadmap**
    4. Track your progress over time
    """)
    st.markdown("---")
    st.caption("Powered by Google Gemini AI & RAG")

# Main layout - two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📄 Upload Resume")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload your latest resume in PDF format"
    )
    
    st.subheader("🎯 Select Target Role")
    
    # Fetch available roles from backend
    @st.cache_data(ttl=3600)
    def fetch_roles():
        try:
            response = requests.get(f"{API_BASE_URL}/roles", timeout=5)
            if response.status_code == 200:
                return response.json().get("roles", [])
            else:
                return ["Data Scientist", "Software Engineer", "Product Manager", "UX Designer"]
        except:
            # Fallback roles if backend not reachable
            return ["Data Scientist", "Software Engineer", "Product Manager", "UX Designer"]
    
    roles = fetch_roles()
    selected_role = st.selectbox("Choose your target job role", roles)
    
    # Optional: user's current year/semester
    st.subheader("📚 Your Context (Optional)")
    current_level = st.selectbox("Current academic level", ["1st Year", "2nd Year", "3rd Year", "4th Year", "Graduate", "Other"])

with col2:
    st.subheader("📊 Analysis Results")
    
    # Button to trigger analysis
    analyze_button = st.button("🚀 Analyze My Career Path", type="primary", use_container_width=True)
    
    # Placeholder for results
    results_placeholder = st.empty()

# Handle analysis
if analyze_button:
    if not uploaded_file:
        st.error("Please upload your resume first.")
    else:
        # Show loading spinner
        with st.spinner("Analyzing your resume and generating roadmap... This may take 15-30 seconds."):
            try:
                # Prepare files and data for backend
                files = {"resume": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                data = {"role": selected_role, "level": current_level}
                
                # Call backend analyze endpoint
                response = requests.post(
                    f"{API_BASE_URL}/analyze",
                    files=files,
                    data=data,
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display results in a nice format
                    with results_placeholder.container():
                        # Match score with progress bar
                        match_score = result.get("match_score", 0)
                        st.markdown(f"### 🎯 Match Score: {match_score}%")
                        st.progress(match_score / 100)
                        
                        # Skill gaps
                        st.markdown("### 📉 Skill Gaps Identified")
                        skill_gaps = result.get("skill_gaps", [])
                        if skill_gaps:
                            for gap in skill_gaps:
                                st.warning(f"• {gap}")
                        else:
                            st.success("No major skill gaps! You're well-aligned for this role.")
                        
                        # Personalized roadmap
                        st.markdown("### 🗺️ Your Personalized Roadmap")
                        roadmap = result.get("roadmap", "")
                        st.markdown(roadmap)
                        
                        # Optional: learning resources
                        st.markdown("### 📚 Recommended Resources")
                        resources = result.get("resources", [])
                        if resources:
                            for res in resources:
                                st.markdown(f"- [{res['name']}]({res['url']})")
                        else:
                            st.info("Check online platforms like Coursera, Udemy, or NPTEL for courses on the above skills.")
                        
                        # Disclaimer
                        st.markdown("---")
                        st.caption("⚠️ Disclaimer: This is an AI-generated guidance tool. Always cross-check with official job descriptions and career counselors.")
                        
                else:
                    st.error(f"Backend error: {response.status_code} - {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to backend. Make sure the backend server is running at http://localhost:8000")
            except requests.exceptions.Timeout:
                st.error("Request timed out. Please try again.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>CareerCompass AI © 2025 | Next Gen Hackathon</div>",
    unsafe_allow_html=True
)