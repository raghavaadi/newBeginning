import os
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips, CompositeVideoClip, vfx, TextClip
from moviepy.video.VideoClip import ImageClip
from typing import List, Tuple
import logging
from models import VideoAnalysis
from audio_processing import extract_audio, transcribe_audio, get_word_timings, transcribe_audio_whisper
from content_analysis import analyze_content
from PIL import Image, ImageDraw, ImageFont
import pytesseract
from fastapi import HTTPException
from models import VideoAnalysis
from audio_processing import extract_audio, transcribe_audio_whisper
from content_analysis import analyze_content
from typing import Dict
logger = logging.getLogger(__name__)

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

def create_short_clip(video_path: str, start_time: float, duration: int, output_path: str):
    logger.info(f"Creating short clip: start_time={start_time}, duration={duration}, output_path={output_path}")
    try:
        with VideoFileClip(video_path) as video:
            end_time = min(start_time + duration, video.duration)
            clip = video.subclip(start_time, end_time)
            
            if clip.audio is None:
                logger.warning("No audio found in the original video")
            else:
                logger.info("Audio track found and included in the clip")
            
            clip.write_videofile(output_path, 
                                 codec="libx264", 
                                 audio_codec="aac", 
                                 temp_audiofile=output_path.replace(".mp4", "-temp-audio.m4a"),
                                 remove_temp=True,
                                 logger=None)
            
        logger.info(f"Clip created successfully: {output_path}")
        
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            logger.info(f"Output file size: {file_size} bytes")
            if file_size == 0:
                raise ValueError("Generated clip file is empty")
        else:
            raise FileNotFoundError("Output clip file not found")
        
    except Exception as e:
        logger.error(f"Error creating clip: {str(e)}")
        raise

async def process_video(video_path: str) -> Dict:
    logger.info(f"Starting video processing for: {video_path}")
    temp_files = []
    
    try:
        # Step 1: Extract audio
        audio_path = extract_audio(video_path)
        temp_files.append(audio_path)
        logger.info(f"Audio extracted: {audio_path}")

        # Step 2: Transcribe audio
        transcript_result = transcribe_audio_whisper(audio_path)
        logger.info(f"Transcription completed")

        # Extract full text from transcript result
        full_transcript = transcript_result.get('text', '')
        
        # Step 3: Extract key frames
        frames, timestamps = extract_key_frames(video_path)
        logger.info(f"Extracted {len(frames)} key frames")

        # Step 4: Detect text in frames
        frame_texts = [detect_text_in_frame(frame) for frame in frames]
        logger.info("Text detection in frames completed")

        # Step 5: Analyze content
        analysis = analyze_content(transcript_result, frame_texts, timestamps)
        logger.info("Content analysis completed")

        # Step 6: Create short clips
        clips = []
        for i, clip_info in enumerate(analysis.get('clips', [])):
            if not isinstance(clip_info, dict):
                logger.warning(f"Unexpected clip info format: {clip_info}")
                continue

            start_time = clip_info.get('start_time')
            duration = clip_info.get('duration')
            description = clip_info.get('description')

            if start_time is None or duration is None:
                logger.warning(f"Missing start_time or duration for clip {i+1}")
                continue

            try:
                start_time = float(start_time)
                duration = int(duration)
            except ValueError:
                logger.warning(f"Invalid start_time or duration for clip {i+1}")
                continue

            output_path = f"outputs/clip_{i+1}.mp4"
            temp_files.append(output_path)
            try:
                create_short_clip(video_path, start_time, duration, output_path, fade_duration=0.5)
                clips.append({
                    "path": output_path,
                    "description": description
                })
            except Exception as e:
                logger.error(f"Error creating clip {i+1}: {str(e)}")
                continue

        logger.info(f"Created {len(clips)} short clips")

        # Step 7: Autoframe and stabilize clips
        stabilized_clips = []
        for i, clip_info in enumerate(clips):
            try:
                with VideoFileClip(clip_info['path']) as clip:
                    stabilized_clip = autoframe_and_stabilize_clip(clip)
                    stabilized_path = f"outputs/stabilized_clip_{i+1}.mp4"
                    temp_files.append(stabilized_path)
                    stabilized_clip.write_videofile(stabilized_path,
                                                    codec="libx264",
                                                    audio_codec="aac",
                                                    temp_audiofile=stabilized_path.replace(".mp4", "-temp-audio.m4a"),
                                                    remove_temp=True,
                                                    logger=None)
                    stabilized_clips.append({
                        "path": stabilized_path,
                        "description": clip_info['description']
                    })
            except Exception as e:
                logger.error(f"Error stabilizing clip {i+1}: {str(e)}")
                continue

        logger.info("Clips autoframed and stabilized")

        # Create the VideoAnalysis object
        result = VideoAnalysis(
            analysis=analysis,
            transcript=full_transcript,
            clips=stabilized_clips
        )

        return result.dict()

    except Exception as e:
        logger.error(f"Error in video processing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error in video processing: {str(e)}")

    finally:
        # Clean up temporary files
        for file in temp_files:
            if os.path.exists(file):
                os.remove(file)
                logger.info(f"Removed temporary file: {file}")

def create_content_from_shorts(output_folder: str, final_output_folder: str):
    logger.info(f"Creating content from shorts in folder: {output_folder}")
    try:
        clip_files = [f for f in os.listdir(output_folder) if f.startswith("clip_") and f.endswith(".mp4")]
        clip_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))

        os.makedirs(final_output_folder, exist_ok=True)

        for i, clip_file in enumerate(clip_files, 1):
            clip_path = os.path.join(output_folder, clip_file)
            logger.info(f"Processing clip: {clip_path}")
            
            try:
                clip = VideoFileClip(clip_path)
                
                # Autoframe, stabilize, and resize the clip to shorts aspect ratio
                processed_clip = autoframe_and_stabilize_clip(clip)
                
                # Extract audio and get word timings
                audio_path = clip_path.replace('.mp4', '.wav')
                clip.audio.write_audiofile(audio_path)
                transcript = transcribe_audio_whisper(audio_path)
                
                # Create caption clips
                caption_clips = []
                for segment in transcript['segments']:
                    text = segment['text']
                    start = segment['start']
                    end = segment['end']
                    words = text.split()
                    for word in words:
                        word_duration = (end - start) / len(words)
                        caption = create_caption_image(text, word, (processed_clip.w, 200))
                        caption_clip = ImageClip(caption).set_duration(word_duration).set_start(start)
                        caption_clips.append(caption_clip)
                        start += word_duration
                
                # Composite the processed clip with captions
                captioned_clip = CompositeVideoClip([processed_clip] + caption_clips)
                
                # Add transitions
                final_clip = captioned_clip.fx(vfx.fadeout, duration=0.5).fx(vfx.fadein, duration=0.5)
                
                # Add a title to the video
                # txt_clip = TextClip(f"Clip {i}", fontsize=70, color='white', size=(final_clip.w, final_clip.h))
                # txt_clip = txt_clip.set_pos('center').set_duration(2)
                
                # Composite the title and the main video
                # final_video = CompositeVideoClip([final_clip, txt_clip.set_start(0)])
                final_video = CompositeVideoClip([final_clip])
                
                # Write the result to a file
                final_output_path = os.path.join(final_output_folder, f"final_clip_{i}.mp4")
                final_video.write_videofile(final_output_path, codec="libx264", audio_codec="aac")
                
                logger.info(f"Final video created: {final_output_path}")
            
            except Exception as e:
                logger.error(f"Error processing clip {clip_file}: {str(e)}")
                continue
            
            finally:
                # Clean up
                if 'audio_path' in locals() and os.path.exists(audio_path):
                    os.remove(audio_path)

    except Exception as e:
        logger.error(f"Error creating content from shorts: {str(e)}")
        raise
def detect_face(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        return faces[0]  # Returns (x, y, w, h) of the first detected face
    return None

def stabilize_frame(prev_frame, curr_frame, prev_pts):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    if prev_pts is None or len(prev_pts) < 10:
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.01, minDistance=30, blockSize=3)
        if prev_pts is None:
            return curr_frame, None
    
    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
    
    if curr_pts is None:
        return curr_frame, None
    
    idx = np.where(status == 1)[0]
    if len(idx) < 10:
        return curr_frame, None
    
    prev_pts = prev_pts[idx]
    curr_pts = curr_pts[idx]
    
    m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
    
    if m is None:
        return curr_frame, curr_pts
    
    stabilized_frame = cv2.warpAffine(curr_frame, m, (curr_frame.shape[1], curr_frame.shape[0]))
    
    return stabilized_frame, curr_pts

def autoframe_and_stabilize_clip(clip: VideoFileClip, target_aspect_ratio: float = 9/16) -> VideoFileClip:
    prev_frame = None
    prev_pts = None
    stabilization_enabled = True

    def process_frame(frame):
        nonlocal prev_frame, prev_pts, stabilization_enabled
        
        if stabilization_enabled:
            if prev_frame is None:
                prev_frame = frame
                stabilized_frame = frame
            else:
                try:
                    stabilized_frame, prev_pts = stabilize_frame(prev_frame, frame, prev_pts)
                    prev_frame = stabilized_frame
                except Exception as e:
                    logger.warning(f"Stabilization failed: {str(e)}. Disabling stabilization.")
                    stabilization_enabled = False
                    stabilized_frame = frame
        else:
            stabilized_frame = frame
        
        face = detect_face(stabilized_frame)
        
        if face is not None:
            x, y, w, h = face
            center_x, center_y = x + w//2, y + h//2
        else:
            center_x, center_y = stabilized_frame.shape[1]//2, stabilized_frame.shape[0]//2
        
        frame_h, frame_w = stabilized_frame.shape[:2]
        target_w = int(frame_h * target_aspect_ratio)
        
        x1 = max(0, center_x - target_w//2)
        x2 = min(frame_w, x1 + target_w)
        
        cropped_frame = stabilized_frame[:, x1:x2]
        return cv2.resize(cropped_frame, (int(frame_h * target_aspect_ratio), frame_h))

    return clip.fl_image(process_frame)

def create_caption_image(text: str, highlight_word: str, size: Tuple[int, int]) -> np.ndarray:
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 10)
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
