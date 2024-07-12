import logging
import json
import signal
from typing import List, Union
from fastapi import HTTPException
from groq import Groq
from utils import get_next_api_key, exponential_backoff, parse_groq_response

logger = logging.getLogger(__name__)

# Global flag for interrupt handling
interrupt_requested = False

def signal_handler(signum, frame):
    global interrupt_requested
    interrupt_requested = True
    logger.info("Interrupt received. Stopping gracefully...")

# Set up the signal handler
signal.signal(signal.SIGINT, signal_handler)

def chunk_content(content: Union[str, List[str]], max_length: int = 1000) -> List[str]:
    if isinstance(content, str):
        return [content[i:i+max_length] for i in range(0, len(content), max_length)]
    elif isinstance(content, list):
        chunks = []
        current_chunk = ""
        for item in content:
            if len(current_chunk) + len(item) + 2 <= max_length:  # +2 for ', '
                current_chunk += (", " if current_chunk else "") + item
            else:
                chunks.append(current_chunk)
                current_chunk = item
        if current_chunk:
            chunks.append(current_chunk)
        return chunks
    else:
        raise ValueError("Content must be either a string or a list of strings")

@exponential_backoff(max_retries=5, base_delay=1, max_delay=60)
def analyze_content(transcript: Union[str, dict], frame_texts: List[str], timestamps: List[float]) -> dict:
    global interrupt_requested
    if interrupt_requested:
        raise KeyboardInterrupt("Operation cancelled by user")

    logger.info("Starting content analysis")
    api_key = get_next_api_key()
    client = Groq(api_key=api_key)
    
    # Handle different transcript formats
    if isinstance(transcript, dict):
        transcript_text = transcript.get('text', '')
    elif isinstance(transcript, str):
        transcript_text = transcript
    else:
        raise ValueError("Transcript must be either a string or a dictionary with a 'text' key")
    
    # Chunk the content
    transcript_chunks = chunk_content(transcript_text)
    frame_text_chunks = chunk_content(frame_texts)
    timestamp_chunks = chunk_content([f"{t:.2f}" for t in timestamps])
    
    all_results = []
    
    for i in range(max(len(transcript_chunks), len(frame_text_chunks), len(timestamp_chunks))):
        t_chunk = transcript_chunks[i] if i < len(transcript_chunks) else ""
        f_chunk = frame_text_chunks[i] if i < len(frame_text_chunks) else ""
        ts_chunk = timestamp_chunks[i] if i < len(timestamp_chunks) else ""
        
        prompt = f"""Analyze the following video content (Part {i+1}):
Transcript: {t_chunk}
Key Frame Texts: {f_chunk}
Timestamps: {ts_chunk}

Provide the following information:
1. Identify 1-3 main topics discussed in this chunk.
2. Suggest 1-2 clips for creating short videos from this chunk. For each clip, provide:
   - Start time (in seconds, use the provided timestamps as reference)
   - Duration (between 15 and 30 seconds)
   - A brief description of the clip content
3. Recommend 2-4 hashtags relevant to this chunk for social media posts.

Format your response as a JSON object with the following structure:
{{
  "topics": ["topic1", "topic2", ...],
  "clips": [
    {{"start_time": float, "duration": int, "description": "string"}},
    ...
  ],
  "hashtags": ["#hashtag1", "#hashtag2", ...]
}}

Ensure all fields are present in your response, even if some are empty lists.
"""

        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "model": "llama3-70b-8192",
        }

        masked_api_key = f"{api_key[:4]}...{api_key[-4:]}"
        logger.info(f"API call to Groq for chunk {i+1}. Key: {masked_api_key}")

        try:
            response = client.chat.completions.create(**payload)
            content = response.choices[0].message.content.strip()
            chunk_result = parse_groq_response(content)
            all_results.append(chunk_result)
        except Exception as e:
            logger.error(f"Error processing chunk {i+1}: {str(e)}")
    
    # Combine results
    combined_result = {
        "topics": [],
        "clips": [],
        "hashtags": []
    }
    
    for result in all_results:
        combined_result["topics"].extend(result.get("topics", []))
        combined_result["clips"].extend(result.get("clips", []))
        combined_result["hashtags"].extend(result.get("hashtags", []))
    
    # Remove duplicates and limit the number of items
    combined_result["topics"] = list(set(combined_result["topics"]))[:5]
    combined_result["clips"] = combined_result["clips"][:5]
    combined_result["hashtags"] = list(set(combined_result["hashtags"]))[:10]
    
    # Generate titles based on the combined analysis
    titles_prompt = f"""Based on the following analysis, suggest 2 catchy titles for short-form content:
Topics: {', '.join(combined_result['topics'])}
Hashtags: {', '.join(combined_result['hashtags'])}

Provide 2 titles in a JSON format: {{"titles":[]}}"""

    titles_payload = {
        "messages": [{"role": "user", "content": titles_prompt}],
        "model": "llama3-8b-8192",
    }

    try:
        titles_response = client.chat.completions.create(**titles_payload)
        titles_content = titles_response.choices[0].message.content.strip()
        titles_result = json.loads(titles_content)
        combined_result["titles"] = titles_result.get("titles", [])[:2]
    except Exception as e:
        logger.error(f"Error generating titles: {str(e)}")
        combined_result["titles"] = []

    return combined_result