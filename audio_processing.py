import os
import logging
import subprocess
import requests
from typing import Dict
from fastapi import HTTPException
from moviepy.editor import VideoFileClip
import os
import logging
import subprocess
import requests
from typing import Dict, List, Tuple
from fastapi import HTTPException
from moviepy.editor import VideoFileClip
from utils import exponential_backoff, get_next_api_key
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence


import cv2
import numpy as np
from moviepy.editor import VideoFileClip, CompositeVideoClip, TextClip
from typing import List, Tuple
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence


logger = logging.getLogger(__name__)

def extract_audio(video_path: str) -> str:
    video = VideoFileClip(video_path)
    audio = video.audio
    audio_path = video_path.replace('.mp4', '.wav')
    audio.write_audiofile(audio_path)
    return audio_path


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
        subprocess.run(command, check=True, capture_output=True, text=True)
        logger.info("Audio preprocessing completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error preprocessing audio: {e.stderr}")
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

@exponential_backoff(max_retries=5, base_delay=1, max_delay=60)
def transcribe_audio_whisper(audio_path: str) -> Dict:
    logger.info(f"Transcribing audio with Whisper: {audio_path}")
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
            data = {"model": "whisper-large-v3", "response_format": "verbose_json"}
            
            logger.debug(f"Making API call to Groq for audio transcription. API Key: {api_key[:5]}...")
            response = requests.post(url, headers=headers, files=files, data=data)
            logger.debug(f"Received response from Groq API. Status code: {response.status_code}")
        
        os.remove(preprocessed_audio)  # Clean up preprocessed file
        
        response.raise_for_status()
        transcript = response.json()
        logger.info(f"Transcription successful. Number of segments: {len(transcript['segments'])}")
        return transcript
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        raise
def get_word_timings(audio_path: str) -> List[Tuple[str, float, float]]:
    r = sr.Recognizer()
    word_timings = []

    audio = AudioSegment.from_wav(audio_path)
    chunks = split_on_silence(audio, min_silence_len=500, silence_thresh=audio.dBFS-14)

    accumulated_time = 0
    for chunk in chunks:
        chunk_path = "temp_chunk.wav"
        chunk.export(chunk_path, format="wav")

        with sr.AudioFile(chunk_path) as source:
            audio = r.record(source)
            try:
                result = r.recognize_google(audio, show_all=True)
                if result and 'alternative' in result:
                    words = result['alternative'][0]['transcript'].split()
                    chunk_duration = len(chunk) / 1000  # Convert to seconds
                    word_duration = chunk_duration / len(words)
                    for word in words:
                        word_start = accumulated_time
                        word_end = word_start + word_duration
                        word_timings.append((word, word_start, word_end))
                        accumulated_time += word_duration
            except sr.UnknownValueError:
                pass

    return word_timings

def detect_mouth_movement(video_path: str) -> List[Tuple[float, float]]:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    mouth_movements = []

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            mouths = mouth_cascade.detectMultiScale(roi_gray, 1.7, 11)

            if len(mouths) > 0:
                mouth_movements.append((frame_count / fps, (frame_count + 1) / fps))

        frame_count += 1

    cap.release()
    return mouth_movements

def create_subtitle_clips(word_timings: List[Tuple[str, float, float]], 
                          mouth_movements: List[Tuple[float, float]], 
                          video_duration: float, 
                          video_size: Tuple[int, int]) -> List[TextClip]:
    subtitle_clips = []
    current_sentence = []
    sentence_start = 0

    for word, start, end in word_timings:
        current_sentence.append(word)
        
        # Check if the word aligns with mouth movement
        word_has_mouth_movement = any(start <= m[0] <= end or start <= m[1] <= end for m in mouth_movements)

        if word_has_mouth_movement or len(current_sentence) > 10 or end - sentence_start > 5:
            sentence = " ".join(current_sentence)
            txt_clip = TextClip(sentence, fontsize=24, color='white', bg_color='black',
                                size=(video_size[0], None), method='caption').set_position(('center', 'bottom'))
            txt_clip = txt_clip.set_start(sentence_start).set_end(end)
            subtitle_clips.append(txt_clip)
            
            current_sentence = []
            sentence_start = end

    # Add any remaining words
    if current_sentence:
        sentence = " ".join(current_sentence)
        txt_clip = TextClip(sentence, fontsize=24, color='white', bg_color='black',
                            size=(video_size[0], None), method='caption').set_position(('center', 'bottom'))
        txt_clip = txt_clip.set_start(sentence_start).set_end(video_duration)
        subtitle_clips.append(txt_clip)

    return subtitle_clips

def add_subtitles_to_video(video_path: str, output_path: str):
    # Extract audio and get word timings
    audio_path = extract_audio(video_path)
    word_timings = get_word_timings(audio_path)

    # Detect mouth movements
    mouth_movements = detect_mouth_movement(video_path)

    # Create video clip and get its properties
    video = VideoFileClip(video_path)
    video_duration = video.duration
    video_size = video.size

    # Create subtitle clips
    subtitle_clips = create_subtitle_clips(word_timings, mouth_movements, video_duration, video_size)

    # Add subtitles to the video
    final_clip = CompositeVideoClip([video] + subtitle_clips)

    # Write the result to a file
    final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

    # Clean up
    video.close()
    for clip in subtitle_clips:
        clip.close()