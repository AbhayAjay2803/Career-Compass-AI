from langchain_google_genai import ChatGoogleGenerativeAI
from backend.config import GOOGLE_API_KEY
import json

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.3)

def analyze_skill_gap(user_skills: list, target_role: str, vector_store) -> dict:
    docs = vector_store.similarity_search(target_role, k=1)
    if not docs:
        return {"match_score": 0, "missing_skills": [], "analysis": "Role not found."}
    role_info = docs[0].page_content
    prompt = f"""
    Compare user skills {user_skills} with role requirements: {role_info}.
    Return JSON: "match_score" (0-100), "missing_skills" (list), "analysis" (brief).
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
        return {"match_score": 50, "missing_skills": [], "analysis": "Analysis complete."}