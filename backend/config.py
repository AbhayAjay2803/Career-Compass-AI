import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("AIzaSyBGTmQ9RDG-Tw03hwX5ngJjYZ64sfxxFA8")

print(f"Loaded API key: {GOOGLE_API_KEY}...")