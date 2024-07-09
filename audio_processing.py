import os
import logging
import subprocess
import requests
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
from moviepy.editor import VideoFileClip
from typing import List, Tuple
from fastapi import FastAPI, File, UploadFile, HTTPException

from utils import get_next_api_key

logger = logging.getLogger(__name__)
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

def transcribe_audio_whisper(audio_path: str) -> dict:
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
            response = requests.post(url, headers=headers, files=files, data=data)
        
        os.remove(preprocessed_audio)  # Clean up preprocessed file
        
        if response.status_code == 200:
            transcript = response.json()
            logger.info(f"Transcription successful. Number of segments: {len(transcript['segments'])}")
            return transcript
        else:
            logger.error(f"Transcription failed. Status code: {response.status_code}, Response: {response.text}")
            raise Exception(f"Transcription failed: {response.text}")
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        raise


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
