# CareerCompass AI

AI-powered career advisor that analyzes your resume and generates a personalized learning roadmap.

## Setup

1. Create virtual environment: `python -m venv venv`
2. Activate: `.\venv\Scripts\activate` (Windows)
3. Install: `pip install -r requirements.txt`
4. Add your Google Gemini API key to `.env`
5. Run setup: `python scripts/setup_vector_store.py`
6. Start backend: `uvicorn backend.main:app --reload`
7. Start frontend: `streamlit run frontend/app.py`
8. Live Link: `https://career-compass-ai-sstj3dbkmk9nk6ot8yprz7.streamlit.app/`