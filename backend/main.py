from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.pdf_parser import extract_text_from_pdf
from backend.skill_extractor import extract_skills_from_resume
from backend.skill_gap_analyzer import analyze_skill_gap
from backend.roadmap_generator import generate_roadmap
from backend.vector_store import get_vector_store

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

vector_store = get_vector_store()

# Load roles from the dataset
def get_available_roles():
    with open('data/role_requirements.json', 'r') as f:
        data = json.load(f)
        return list(data.keys())

@app.get("/roles")
async def get_roles():
    return {"roles": get_available_roles()}

@app.post("/analyze")
async def analyze_resume(
    resume: UploadFile = File(...),
    role: str = Form(...),
    level: str = Form("")
):
    # Extract text from PDF
    contents = await resume.read()
    resume_text = extract_text_from_pdf(contents)
    
    # Extract skills
    extracted = extract_skills_from_resume(resume_text)
    user_skills = extracted.get("skills", [])
    
    # Analyze skill gap
    gap = analyze_skill_gap(user_skills, role, vector_store)
    match_score = gap.get("match_score", 0)
    missing_skills = gap.get("missing_skills", [])
    
    # Generate roadmap
    roadmap_md = generate_roadmap(user_skills, missing_skills, role, vector_store)
    
    # Load learning resources for recommended courses
    with open('data/learning_resources.json', 'r') as f:
        all_resources = json.load(f)
    # Simple mapping: if a missing skill matches a course keyword, suggest it
    suggested_resources = []
    for skill in missing_skills:
        for course_name, info in all_resources.get("courses", {}).items():
            if any(keyword.lower() in course_name.lower() or keyword.lower() in skill.lower() 
                   for keyword in skill.split()):
                suggested_resources.append({"name": course_name, "url": info["url"]})
                break
    # Deduplicate
    seen = set()
    unique_resources = []
    for r in suggested_resources:
        if r["name"] not in seen:
            seen.add(r["name"])
            unique_resources.append(r)
    
    return {
        "match_score": match_score,
        "skill_gaps": missing_skills,
        "roadmap": roadmap_md,
        "resources": unique_resources[:5]  # limit to 5
    }