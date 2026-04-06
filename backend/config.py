import os
from pathlib import Path
from dotenv import load_dotenv

# Construct the absolute path to the .env file in the project root
project_root = Path(__file__).parent.parent
env_path = project_root / ".env"

# Load the .env file from that exact location
load_dotenv(dotenv_path=env_path)

# Get the API key from the environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Simple check to help with debugging
if GOOGLE_API_KEY:
    print(f"✅ API key loaded: {GOOGLE_API_KEY[:10]}...")
else:
    print(f"❌ API key not found. Please ensure .env exists at {env_path}")