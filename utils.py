import os
import logging
import random
from dotenv import load_dotenv
from threading import Lock
from collections import deque
import time
from functools import wraps
import random
load_dotenv()
import re
import json
import logging
from typing import Dict, Any
logger = logging.getLogger(__name__)

def parse_groq_response(content: str) -> Dict[str, Any]:
    """
    Parse the Groq API response and extract the structured content.
    
    Args:
    content (str): The raw content returned by the Groq API.
    
    Returns:
    dict: A dictionary containing the parsed content.
    """
    try:
        # First, try to parse the entire content as JSON
        return json.loads(content)
    except json.JSONDecodeError:
        # If that fails, try to find a JSON-like structure in the content
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON structure: {e}")
        
        # If JSON parsing fails, extract information manually
        topics = re.findall(r'"topics":\s*\[(.*?)\]', content, re.DOTALL)
        topics = [topic.strip(' "') for topic in re.findall(r'"([^"]*)"', topics[0])] if topics else []
        
        clips = re.findall(r'\{[^{}]*"start_time":[^{}]*"duration":[^{}]*"description":[^{}]*\}', content)
        clips = [json.loads(clip) for clip in clips]
        
        hashtags = re.findall(r'#\w+', content)
        
        return {
            "topics": topics,
            "clips": clips,
            "hashtags": list(set(hashtags))  # Remove duplicates
        }


def exponential_backoff(max_retries=5, base_delay=1, max_delay=60):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if "429" not in str(e) and retries == max_retries - 1:
                        raise
                    
                    retries += 1
                    delay = min(base_delay * (2 ** retries) + random.uniform(0, 1), max_delay)
                    logger.warning(f"Rate limit exceeded. Retrying in {delay:.2f} seconds. Retry {retries}/{max_retries}")
                    time.sleep(delay)
            raise Exception(f"Max retries ({max_retries}) exceeded")
        return wrapper
    return decorator

class APIKeyManager:
    def __init__(self):
        self.api_keys = os.getenv("GROQ_API_KEYS", "").split(",")
        self.key_queue = deque(self.api_keys)
        self.key_usage = {key: {"last_used": 0, "count": 0} for key in self.api_keys}
        self.cooldown_period = 60  # 60 seconds cooldown

    def get_next_api_key(self):
        current_time = time.time()
        for _ in range(len(self.key_queue)):
            key = self.key_queue[0]
            if current_time - self.key_usage[key]["last_used"] > self.cooldown_period:
                self.key_queue.rotate(-1)
                self.key_usage[key]["last_used"] = current_time
                self.key_usage[key]["count"] += 1
                logger.debug(f"Using API key: {key[:5]}... (Used {self.key_usage[key]['count']} times)")
                return key
            self.key_queue.rotate(-1)
        
        # If all keys are on cooldown, wait and retry
        time.sleep(self.cooldown_period)
        return self.get_next_api_key()
api_key_manager = APIKeyManager()

def get_next_api_key():
    return api_key_manager.get_next_api_key()
class APIKeyManager:
    def __init__(self):
        self.api_keys = os.getenv("GROQ_API_KEYS", "").split(",")
        self.key_queue = deque()
        self.lock = Lock()
        self.keys_shuffled = False
        self.key_usage_count = {key: 0 for key in self.api_keys}

    def get_next_api_key(self):
        with self.lock:
            if not self.keys_shuffled:
                self.shuffle_api_keys()
            if not self.key_queue:
                self.key_queue.extend(self.api_keys)
            api_key = self.key_queue.popleft()
            self.key_queue.append(api_key)
            self.key_usage_count[api_key] += 1
            logger.debug(f"Using API key: {api_key[:5]}... (Used {self.key_usage_count[api_key]} times)")
        return api_key
def setup_directories():
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

def cleanup_files(file_paths):
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Removed temporary file: {file_path}")

def initialize_app():
    api_key_manager.shuffle_api_keys()
    setup_directories()
    logger.info("Application initialized: API keys shuffled and directories set up")



if __name__ == "__main__":
    sample_content = '''
    Here is the analysis of the provided video content:

    **Analysis**

    **1. Key topics discussed:**
    The key topics discussed in this video are:
    * Tata Consumer's growth path and digital transformation
    * The company's email management challenges and the solution provided by Fresh Service
    * The benefits of using Fresh Service, including reduced MTTR (Mean Time To Resolve) and improved incident resolution

    **2. Potential clip ideas:**
    Here are some potential clip ideas:

    * Clip 1: (0:00 - 0:15) - "Exciting growth path to become a premium FSCG company"
        + Brief description: Introduction to Tata Consumer's growth plans
    * Clip 2: (9:93 - 10:28) - "We decided that this is not going to work out in this current digital transformation"
        + Brief description: The company's email management challenges

    **3. Relevant hashtags:**
    Here are some relevant hashtags:

    * #TataConsumer
    * #DigitalTransformation
    * #CustomerService

    **JSON Output:**

    {
    "topics": [
    "Tata Consumer's growth path and digital transformation",
    "Email management challenges and Fresh Service solution",
    "Benefits of using Fresh Service"
    ],
    "clips": [
    {
    "start_time": 0.00,
    "duration": 15,
    "description": "Introduction to Tata Consumer's growth plans"
    },
    {
    "start_time": 9.93,
    "duration": 15,
    "description": "The company's email management challenges"
    }
    ],
    "hashtags": [
    "#TataConsumer",
    "#DigitalTransformation",
    "#CustomerService"
    ]
    }
    '''
    
    result = parse_groq_response(sample_content)
    print(json.dumps(result, indent=2))