from langchain_google_genai import ChatGoogleGenerativeAI
from backend.config import GOOGLE_API_KEY
import json

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.3)

def extract_skills_from_resume(resume_text: str) -> dict:
    prompt = f"""
    Analyze resume. Return JSON with keys: "skills", "experience", "education".
    Skills: list of technical/soft skills.
    Experience: list of roles.
    Education: list of degrees/institutions.
    Resume: {resume_text[:4000]}
    """
    response = llm.invoke(prompt)
    content = response.content
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]
    try:
        return json.loads(content)
    except:
        return {"skills": [], "experience": [], "education": []}