# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Access environment variables
GEM_API_KEY = os.getenv("GEM_API_KEY")
GOOGLE_APPLICATION_CREDENTIALS = "/etc/secrets/gen-lang-client-0621264914-16562d035a74.json"
