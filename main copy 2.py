import os
import json
import requests
import logging
import re
import traceback
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
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips, CompositeVideoClip, vfx, TextClip
from moviepy.video.VideoClip import ImageClip
from typing import List, Tuple
import logging
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
from PIL import Image, ImageDraw, ImageFont

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

# Groq API key rotation
api_keys = os.getenv("GROQ_API_KEYS").split(",")
current_key_index = 0

def get_word_timings(audio_path: str) -> List[Tuple[str, float, float]]:
    r = sr.Recognizer()
    word_timings = []

    # Load audio file
    audio = AudioSegment.from_wav(audio_path)
    
    # Split audio into chunks
    chunks = split_on_silence(audio, min_silence_len=500, silence_thresh=audio.dBFS-14)

    for i, chunk in enumerate(chunks):
        # Export chunk to a temporary file
        chunk_path = f"temp_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")

        with sr.AudioFile(chunk_path) as source:
            audio = r.record(source)
            try:
                result = r.recognize_google(audio, show_all=True)
                if result and 'alternative' in result:
                    words = result['alternative'][0]['transcript'].split()
                    word_count = len(words)
                    chunk_duration = len(chunk) / 1000  # Convert to seconds
                    word_duration = chunk_duration / word_count
                    start_time = sum([len(c) for c in chunks[:i]]) / 1000
                    for j, word in enumerate(words):
                        word_start = start_time + j * word_duration
                        word_end = word_start + word_duration
                        word_timings.append((word, word_start, word_end))
            except sr.UnknownValueError:
                logger.warning(f"Could not understand audio in chunk {i}")
            except sr.RequestError as e:
                logger.error(f"Could not request results from Google Speech Recognition service; {e}")

        # Clean up temporary file
        os.remove(chunk_path)

    return word_timings

def create_caption_image(text: str, highlight_word: str, size: Tuple[int, int]) -> np.ndarray:
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 40)
    image = Image.new('RGBA', size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    
    words = text.split()
    x, y = 10, 10
    for word in words:
        word_width, word_height = draw.textsize(word + " ", font=font)
        if x + word_width > size[0] - 10:
            x = 10
            y += word_height + 5
        if word == highlight_word:
            draw.rectangle([x, y, x + word_width, y + word_height], fill=(255, 255, 0, 128))
        draw.text((x, y), word, font=font, fill=(255, 255, 255, 255))
        x += word_width

    return np.array(image)

def autoframe_and_resize_clip(clip: VideoFileClip, target_aspect_ratio: float = 9/16) -> VideoFileClip:
    def make_frame(t):
        frame = clip.get_frame(t)
        face = detect_face(frame)
        
        if face is not None:
            x, y, w, h = face
            center_x, center_y = x + w//2, y + h//2
        else:
            center_x, center_y = frame.shape[1]//2, frame.shape[0]//2
        
        frame_h, frame_w = frame.shape[:2]
        target_w = int(frame_h * target_aspect_ratio)
        
        x1 = max(0, center_x - target_w//2)
        x2 = min(frame_w, x1 + target_w)
        
        cropped_frame = frame[:, x1:x2]
        return cv2.resize(cropped_frame, (int(clip.h * target_aspect_ratio), clip.h))

    return clip.fl(make_frame)

def create_content_from_shorts(output_folder: str, final_output_path: str):
    logger.info(f"Creating content from shorts in folder: {output_folder}")
    try:
        clip_files = [f for f in os.listdir(output_folder) if f.startswith("clip_") and f.endswith(".mp4")]
        clip_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))

        final_clips = []
        for clip_file in clip_files:
            clip_path = os.path.join(output_folder, clip_file)
            logger.info(f"Processing clip: {clip_path}")
            
            clip = VideoFileClip(clip_path)
            
            # Autoframe and resize the clip to shorts aspect ratio
            processed_clip = autoframe_and_resize_clip(clip)
            
            # Extract audio and get word timings
            audio_path = clip_path.replace('.mp4', '.wav')
            clip.audio.write_audiofile(audio_path)
            word_timings = get_word_timings(audio_path)
            
            # Create caption clips
            caption_clips = []
            for word, start, end in word_timings:
                caption = create_caption_image(" ".join(word for word, _, _ in word_timings), word, (processed_clip.w, 200))
                caption_clip = ImageClip(caption).set_duration(end - start).set_start(start)
                caption_clips.append(caption_clip)
            
            # Composite the processed clip with captions
            captioned_clip = CompositeVideoClip([processed_clip] + caption_clips)
            
            # Add transitions
            final_clip = captioned_clip.fx(vfx.fadeout, duration=0.5).fx(vfx.fadein, duration=0.5)
            
            final_clips.append(final_clip)
            
            # Clean up
            os.remove(audio_path)

        # Concatenate all clips
        final_video = concatenate_videoclips(final_clips)

        # Add a title to the video
        txt_clip = TextClip("Video Highlights", fontsize=70, color='white', size=(final_video.w, final_video.h))
        txt_clip = txt_clip.set_pos('center').set_duration(4)

        # Composite the title and the main video
        final_video = CompositeVideoClip([final_video, txt_clip.set_start(0)])

        # Write the result to a file
        final_video.write_videofile(final_output_path, codec="libx264", audio_codec="aac")

        logger.info(f"Final video created: {final_output_path}")

    except Exception as e:
        logger.error(f"Error creating content from shorts: {str(e)}")
        raise



def get_next_api_key():
    global current_key_index
    api_key = api_keys[current_key_index]
    current_key_index = (current_key_index + 1) % len(api_keys)
    logger.debug(f"Using API key index: {current_key_index}")
    return api_key

def extract_audio(video_path: str) -> str:
    logger.info(f"Extracting audio from {video_path}")
    try:
        video = VideoFileClip(video_path)
        audio = video.audio
        audio_path = video_path.replace('.mp4', '.wav')
        audio.write_audiofile(audio_path)
        logger.info(f"Audio extracted successfully: {audio_path}")
        return audio_path
    except Exception as e:
        logger.error(f"Error extracting audio: {str(e)}")
        raise

def preprocess_audio(input_path: str, output_path: str):
    logger.info(f"Preprocessing audio: {input_path} -> {output_path}")
    command = [
        "ffmpeg",
        "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        "-map", "0:a",
        output_path
    ]
    try:
        subprocess.run(command, check=True)
        logger.info("Audio preprocessing completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error preprocessing audio: {str(e)}")
        raise

def transcribe_audio(audio_path: str) -> str:
    logger.info(f"Transcribing audio: {audio_path}")
    preprocessed_audio = audio_path.replace('.wav', '_preprocessed.wav')
    try:
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
            transcript = response.json()["text"]
            logger.info(f"Transcription successful. Length: {len(transcript)} characters")
            return transcript
        else:
            logger.error(f"Transcription failed. Status code: {response.status_code}, Response: {response.text}")
            raise HTTPException(status_code=500, detail=f"Transcription failed: {response.text}")
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        raise

def extract_key_frames(video_path: str, interval: int = 5) -> tuple:
    logger.info(f"Extracting key frames from {video_path} at {interval} second intervals")
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
    logger.info(f"Extracted {len(frames)} key frames")
    return frames, timestamps

def detect_text_in_frame(frame: np.ndarray) -> str:
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    text = pytesseract.image_to_string(img)
    return text

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
        
        # Extract JSON from the response
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            logger.info("JSON extracted from code block in response")
        else:
            # If no code block is found, try to find a JSON object in the text
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


def create_short_clip(video_path: str, start_time: float, duration: int, output_path: str):
    logger.info(f"Creating short clip: start_time={start_time}, duration={duration}, output_path={output_path}")
    try:
        with VideoFileClip(video_path) as video:
            # Ensure the clip doesn't extend beyond the video duration
            end_time = min(start_time + duration, video.duration)
            
            # Extract the clip
            clip = video.subclip(start_time, end_time)
            
            # Ensure the audio is included
            if clip.audio is None:
                logger.warning("No audio found in the original video")
            else:
                logger.info("Audio track found and included in the clip")
            
            # Write the clip with both video and audio
            clip.write_videofile(output_path, 
                                 codec="libx264", 
                                 audio_codec="aac", 
                                 temp_audiofile=output_path.replace(".mp4", "-temp-audio.m4a"),
                                 remove_temp=True,
                                 logger=None)  # Suppress moviepy's console output
            
        logger.info(f"Clip created successfully: {output_path}")
        
        # Verify the output file
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            logger.info(f"Output file size: {file_size} bytes")
            if file_size == 0:
                logger.error("Output file is empty")
                raise ValueError("Generated clip file is empty")
        else:
            logger.error("Output file was not created")
            raise FileNotFoundError("Output clip file not found")
        
    except Exception as e:
        logger.error(f"Error creating clip: {str(e)}")
        raise

class VideoAnalysis(BaseModel):
    analysis: dict
    transcript: str
    clips: List[dict]

@app.post("/process-video/", response_model=VideoAnalysis)
async def process_video(file: UploadFile = File(...)):
    logger.info(f"Received video for processing: {file.filename}")
    
    # Create directories if they don't exist
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # Save uploaded file
    video_path = f"uploads/{file.filename}"
    with open(video_path, "wb") as buffer:
        buffer.write(await file.read())
    logger.info(f"Video saved to {video_path}")
    
    try:
        # Process video
        logger.info("Starting video processing")
        audio_path = extract_audio(video_path)
        logger.info("Audio extraction completed")
        
        transcript = transcribe_audio(audio_path)
        logger.info("Audio transcription completed")
        
        logger.info("Extracting key frames")
        frames, timestamps = extract_key_frames(video_path)
        logger.info(f"Extracted {len(frames)} key frames")
        
        logger.info("Detecting text in frames")
        frame_texts = [detect_text_in_frame(frame) for frame in frames]
        logger.info("Text detection completed")
        
        logger.info("Analyzing content")
        analysis = analyze_content(transcript, frame_texts, timestamps)
        logger.info("Content analysis completed")
        
        # Create short clips
        clips = []
        logger.info("Creating short clips")
        for i, clip_info in enumerate(analysis.get('clips', [])):
            output_path = f"outputs/clip_{i+1}.mp4"
            create_short_clip(video_path, clip_info['start_time'], clip_info['duration'], output_path)
            clips.append({
                "path": output_path,
                "description": clip_info['description']
            })
        logger.info(f"Created {len(clips)} short clips")
        
        return VideoAnalysis(analysis=analysis, transcript=transcript, clips=clips)
    
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
    
    finally:
        # Cleanup
        logger.info("Cleaning up temporary files")
        if os.path.exists(video_path):
            os.remove(video_path)
        if 'audio_path' in locals() and os.path.exists(audio_path):
            os.remove(audio_path)

@app.post("/create-final-video/")
async def create_final_video():
    try:
        output_folder = "outputs"
        final_output_path = "final_video.mp4"
        create_content_from_shorts(output_folder, final_output_path)
        return {"message": "Final video created successfully", "path": final_output_path}
    except Exception as e:
        logger.error(f"Error in create_final_video endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/clip/{clip_id}")
async def get_clip(clip_id: int):
    logger.info(f"Request for clip {clip_id}")
    clip_path = f"outputs/clip_{clip_id}.mp4"
    if not os.path.exists(clip_path):
        logger.warning(f"Clip not found: {clip_path}")
        raise HTTPException(status_code=404, detail="Clip not found")
    logger.info(f"Returning clip: {clip_path}")
    return FileResponse(clip_path, media_type="video/mp4", filename=f"clip_{clip_id}.mp4")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server")
    uvicorn.run(app, host="0.0.0.0", port=8000)