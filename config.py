# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Access environment variables
GEM_API_KEY = os.getenv("GEM_API_KEY")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
