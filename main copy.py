import os
import json
import os
import json
import requests
import logging
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from moviepy.editor import VideoFileClip, concatenate_videoclips
import cv2
import numpy as np
from PIL import Image
import pytesseract
from groq import Groq
import subprocess
from pydantic import BaseModel
from dotenv import load_dotenv


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

# Groq API key rotation
api_keys = os.getenv("GROQ_API_KEYS").split(",")
current_key_index = 0

def get_next_api_key():
    global current_key_index
    api_key = api_keys[current_key_index]
    current_key_index = (current_key_index + 1) % len(api_keys)
    return api_key

def extract_audio(video_path: str) -> str:
    video = VideoFileClip(video_path)
    audio = video.audio
    audio_path = video_path.replace('.mp4', '.wav')
    audio.write_audiofile(audio_path)
    return audio_path

def preprocess_audio(input_path: str, output_path: str):
    command = [
        "ffmpeg",
        "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        "-map", "0:a",
        output_path
    ]
    subprocess.run(command, check=True)

def transcribe_audio(audio_path: str) -> str:
    preprocessed_audio = audio_path.replace('.wav', '_preprocessed.wav')
    preprocess_audio(audio_path, preprocessed_audio)

    api_key = get_next_api_key()
    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    with open(preprocessed_audio, "rb") as audio_file:
        files = {"file": audio_file}
        data = {"model": "whisper-large-v3"}
        response = requests.post(url, headers=headers, files=files, data=data)
    
    os.remove(preprocessed_audio)  # Clean up preprocessed file
    
    if response.status_code == 200:
        return response.json()["text"]
    else:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {response.text}")

def extract_key_frames(video_path: str, interval: int = 5) -> tuple:
    video = cv2.VideoCapture(video_path)
    frames = []
    timestamps = []
    count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        if count % (30 * interval) == 0:  # Assuming 30 fps, extract every 'interval' seconds
            frames.append(frame)
            timestamps.append(count / 30)  # Convert frame count to seconds
        count += 1
    video.release()
    return frames, timestamps

def detect_text_in_frame(frame: np.ndarray) -> str:
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    text = pytesseract.image_to_string(img)
    return text

def analyze_content(transcript: str, frame_texts: List[str], timestamps: List[float]) -> dict:
    client = Groq(api_key=get_next_api_key())
    prompt = f"""
    Analyze the following video transcript and key frame texts:
    Transcript: {transcript}
    Key Frame Texts: {', '.join(frame_texts)}
    Timestamps: {', '.join([f"{t:.2f}" for t in timestamps])}
    
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
        logger.info(f"Groq API response: {content}")
        
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from Groq API response: {e}")
            logger.error(f"Raw content: {content}")
            raise HTTPException(status_code=500, detail="Failed to parse content analysis results")
    except Exception as e:
        logger.error(f"Error during content analysis: {str(e)}")
        raise HTTPException(status_code=500, detail="Error during content analysis")

def create_short_clip(video_path: str, start_time: float, duration: int, output_path: str):
    try:
        with VideoFileClip(video_path) as video:
            # Ensure the clip doesn't extend beyond the video duration
            end_time = min(start_time + duration, video.duration)
            clip = video.subclip(start_time, end_time)
            
            # Write the clip with both video and audio
            clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
            
        logger.info(f"Created clip: {output_path}")
    except Exception as e:
        logger.error(f"Error creating clip: {str(e)}")
        raise


# def create_short_clip(video_path: str, start_time: float, duration: int, output_path: str):
#     with VideoFileClip(video_path) as video:
#         clip = video.subclip(start_time, start_time + duration)
#         clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

class VideoAnalysis(BaseModel):
    analysis: dict
    transcript: str
    clips: List[dict]


@app.post("/process-video/", response_model=VideoAnalysis)
async def process_video(file: UploadFile = File(...)):
    # Create directories if they don't exist
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # Save uploaded file
    video_path = f"uploads/{file.filename}"
    with open(video_path, "wb") as buffer:
        buffer.write(await file.read())
    
    try:
        # Process video
        audio_path = extract_audio(video_path)
        transcript = transcribe_audio(audio_path)
        frames, timestamps = extract_key_frames(video_path)
        frame_texts = [detect_text_in_frame(frame) for frame in frames]
        
        analysis = analyze_content(transcript, frame_texts, timestamps)
        
        # Create short clips
        clips = []
        for i, clip_info in enumerate(analysis['clips']):
            output_path = f"outputs/clip_{i+1}.mp4"
            create_short_clip(video_path, clip_info['start_time'], clip_info['duration'], output_path)
            clips.append({
                "path": output_path,
                "description": clip_info['description']
            })
        
        return VideoAnalysis(analysis=analysis, transcript=transcript, clips=clips)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup
        if os.path.exists(video_path):
            os.remove(video_path)
        if 'audio_path' in locals() and os.path.exists(audio_path):
            os.remove(audio_path)

@app.get("/clip/{clip_id}")
async def get_clip(clip_id: int):
    clip_path = f"outputs/clip_{clip_id}.mp4"
    if not os.path.exists(clip_path):
        raise HTTPException(status_code=404, detail="Clip not found")
    return FileResponse(clip_path, media_type="video/mp4")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)