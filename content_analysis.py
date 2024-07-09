import logging
import json
import re
from typing import List
from fastapi import HTTPException
from groq import Groq
from utils import get_next_api_key

logger = logging.getLogger(__name__)

def analyze_content(transcript: str, frame_texts: List[str], timestamps: List[float]) -> dict:
    logger.info("Starting content analysis")
    client = Groq(api_key=get_next_api_key())
    prompt = f"""
    Analyze the following video transcript and key frame texts:
    Transcript: {transcript[:500]}... (truncated)
    Key Frame Texts: {', '.join(frame_texts[:5])}... (truncated)
    Timestamps: {', '.join([f"{t:.2f}" for t in timestamps[:5]])}... (truncated)
    
    1. Identify the main topics discussed.
    2. Suggest 3 catchy titles for short-form content.
    3. Propose 5 clips for creating short videos. For each clip, provide:
       - Start time (in seconds)
       - Duration (15, 30, or 60 seconds)
       - A brief description of the clip content
    4. Recommend hashtags for social media posts.

    Format your response as a JSON object with keys: "topics", "titles", "clips", and "hashtags".
    """
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
        )
        content = response.choices[0].message.content.strip()
        logger.info(f"Groq API response received. Length: {len(content)} characters")
        
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            logger.info("JSON extracted from code block in response")
        else:
            json_match = re.search(r'(\{[^{]*"topics":[^{]*"titles":[^{]*"clips":[^{]*"hashtags":[^}]*\})', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                logger.info("JSON extracted from text in response")
            else:
                logger.error("No valid JSON found in the response")
                raise ValueError("No valid JSON found in the response")
        
        try:
            analysis = json.loads(json_str)
            logger.info("JSON parsed successfully")
            return analysis
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from Groq API response: {e}")
            logger.error(f"Extracted JSON string: {json_str}")
            raise ValueError(f"Failed to parse JSON: {e}")
    except Exception as e:
        logger.error(f"Error during content analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during content analysis: {str(e)}")