import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Groq API key rotation
api_keys = os.getenv("GROQ_API_KEYS").split(",")
current_key_index = 0

def get_next_api_key():
    global current_key_index
    api_key = api_keys[current_key_index]
    current_key_index = (current_key_index + 1) % len(api_keys)
    logger.debug(f"Using API key index: {current_key_index}")
    return api_key

def setup_directories():
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

def cleanup_files(file_paths):
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Removed temporary file: {file_path}")