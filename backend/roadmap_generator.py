import json
from langchain_google_genai import ChatGoogleGenerativeAI
from backend.config import GOOGLE_API_KEY

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.5)

def generate_roadmap(user_skills: list, missing_skills: list, target_role: str, vector_store) -> str:
    docs = vector_store.similarity_search(target_role, k=1)
    role_context = docs[0].page_content if docs else ""
    with open('data/learning_resources.json', 'r') as f:
        resources = json.load(f)
    prompt = f"""
    Generate a personalized learning roadmap for {target_role}.
    User skills: {user_skills}. Missing: {missing_skills}.
    Role context: {role_context}
    Resources: {json.dumps(resources)}
    Output as Markdown with: Summary, Learning path (courses), Certifications, Timeline (3/6/12 months).
    """
    response = llm.invoke(prompt)
    return response.content