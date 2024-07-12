import os
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, CompositeVideoClip, vfx
from moviepy.video.fx.all import fadeout, fadein
from moviepy.video.VideoClip import ImageClip
from typing import List, Tuple
import logging
from models import VideoAnalysis
from audio_processing import extract_audio, transcribe_audio, transcribe_audio_whisper
from content_analysis import analyze_content
from PIL import Image, ImageDraw, ImageFont
import pytesseract
from collections import deque
from fastapi import HTTPException
from typing import Dict
from moviepy.editor import VideoFileClip, concatenate_videoclips, CompositeVideoClip, vfx
from utils import get_next_api_key, exponential_backoff
import time
logger = logging.getLogger(__name__)

def extract_key_frames(video_path: str, interval: int = 5) -> Tuple[List[np.ndarray], List[float]]:
    logger.info(f"Extracting key frames from {video_path} at {interval} second intervals")
    video = cv2.VideoCapture(video_path)
    frames = []
    timestamps = []
    count = 0
    fps = video.get(cv2.CAP_PROP_FPS)
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
        if count % int(fps * interval) == 0:
            frames.append(frame)
            timestamps.append(count / fps)
        count += 1
    
    video.release()
    logger.info(f"Extracted {len(frames)} key frames")
    return frames, timestamps

def detect_text_in_frame(frame: np.ndarray) -> str:
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    text = pytesseract.image_to_string(img)
    return text.strip()


def create_short_clip(video_path: str, start_time: float, duration: int, output_path: str, fade_duration: float = 0.5):
    logger.info(f"Creating short clip: start_time={start_time}, duration={duration}, output_path={output_path}")
    try:
        with VideoFileClip(video_path) as video:
            clip = video.subclip(start_time, start_time + duration)
            clip = clip.fx(fadein, duration=fade_duration)
            clip = clip.fx(fadeout, duration=fade_duration)
            
            clip.write_videofile(output_path, 
                                 codec="libx264", 
                                 audio_codec="aac", 
                                 temp_audiofile=output_path.replace(".mp4", "-temp-audio.m4a"),
                                 remove_temp=True,
                                 logger=None)
        
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise FileNotFoundError("Output clip file not found or is empty")
        
    except Exception as e:
        logger.error(f"Error creating clip: {str(e)}")
        raise

def create_final_video(clips: List[dict], output_path: str, transition_duration: float = 0.5):
    logger.info(f"Creating final video with {len(clips)} clips")
    try:
        video_clips = []
        for i, clip_info in enumerate(clips):
            clip = VideoFileClip(clip_info['path'])
            if i > 0:  # Add fade in to all clips except the first one
                clip = clip.fx(fadein, duration=transition_duration)
            if i < len(clips) - 1:  # Add fade out to all clips except the last one
                clip = clip.fx(fadeout, duration=transition_duration)
            video_clips.append(clip)
        
        # Overlap the clips slightly to create a smooth transition
        final_clips = []
        for i, clip in enumerate(video_clips):
            if i > 0:
                start_time = sum(c.duration for c in final_clips) - transition_duration
                final_clips.append(clip.set_start(start_time))
            else:
                final_clips.append(clip)
        
        # Create the final composite video
        final_clip = CompositeVideoClip(final_clips)
        
        # Write the final video
        final_clip.write_videofile(output_path,
                                   codec="libx264",
                                   audio_codec="aac",
                                   temp_audiofile=output_path.replace(".mp4", "-temp-audio.m4a"),
                                   remove_temp=True,
                                   logger=None)
        
        logger.info(f"Final video created successfully: {output_path}")
    except Exception as e:
        logger.error(f"Error creating final video: {str(e)}")
        raise
    finally:
        # Close all clip objects to free up resources
        for clip in video_clips:
            clip.close()


@exponential_backoff(max_retries=5, base_delay=1, max_delay=60)
async def process_video(video_path: str) -> VideoAnalysis:
    logger.info(f"Starting video processing for: {video_path}")
    temp_files = []
    api_request_count = 0
    
    try:
        # Step 1: Extract audio
        audio_path = extract_audio(video_path)
        temp_files.append(audio_path)
        logger.info(f"Audio extracted: {audio_path}")

        # Step 2: Transcribe audio
        transcript = transcribe_audio_whisper(audio_path)
        api_request_count += 1
        logger.info(f"Transcription completed. Length: {len(transcript)} characters. API requests so far: {api_request_count}")

        # Add a small delay between API calls
        time.sleep(1)

        # Step 3: Extract key frames
        frames, timestamps = extract_key_frames(video_path)
        logger.info(f"Extracted {len(frames)} key frames")

        # Step 4: Detect text in frames
        frame_texts = [detect_text_in_frame(frame) for frame in frames]
        logger.info("Text detection in frames completed")

        # Step 5: Analyze content
        analysis = analyze_content(transcript, frame_texts, timestamps)
        api_request_count += 1
        logger.info(f"Content analysis completed. API requests so far: {api_request_count}")

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
                create_short_clip(
                    video_path, 
                    start_time, 
                    duration, 
                    output_path,
                    fade_duration=0.5
                )
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

        # Step 8: Add captions to clips
        captioned_clips = []
        for i, clip_info in enumerate(stabilized_clips):
            try:
                clip = VideoFileClip(clip_info['path'])
                captioned_clip = sync_captions_with_lip_movement(clip, transcript)
                captioned_path = f"outputs/captioned_clip_{i+1}.mp4"
                temp_files.append(captioned_path)
                captioned_clip.write_videofile(captioned_path,
                                               codec="libx264",
                                               audio_codec="aac",
                                               temp_audiofile=captioned_path.replace(".mp4", "-temp-audio.m4a"),
                                               remove_temp=True,
                                               logger=None)
                captioned_clips.append({
                    "path": captioned_path,
                    "description": clip_info['description']
                })
                clip.close()
                captioned_clip.close()
            except Exception as e:
                logger.error(f"Error adding captions to clip {i+1}: {str(e)}")
                continue

        logger.info("Captions added to clips")

        # Prepare the result
        result = VideoAnalysis(
            analysis=analysis,
            transcript=transcript,
            clips=captioned_clips
        )

        logger.info(f"Video processing completed. Total API requests: {api_request_count}")
        return result

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
                # Step 1: Load the clip
                clip = VideoFileClip(clip_path)
                
                # Step 2: Autoframe and stabilize the clip
                logger.info("Autoframing and stabilizing clip")
                processed_clip = autoframe_and_stabilize_clip(clip)
                processed_clip_path = os.path.join(output_folder, f"processed_clip_{i}.mp4")
                processed_clip.write_videofile(processed_clip_path, codec="libx264", audio_codec="aac")
                
                # Step 3: Extract audio and get transcript
                logger.info("Extracting audio and transcribing")
                audio_path = clip_path.replace('.mp4', '.wav')
                clip.audio.write_audiofile(audio_path)
                transcript = transcribe_audio_whisper(audio_path)
                
                # Step 4: Process clip with synchronized captions
                logger.info("Generating synchronized captions")
                captioned_clip = sync_captions_with_lip_movement(processed_clip, transcript)
                
                # Step 5: Write the final captioned video
                captioned_output_path = os.path.join(final_output_folder, f"final_clip_{i}.mp4")
                captioned_clip.write_videofile(captioned_output_path, codec="libx264", audio_codec="aac")
                
                logger.info(f"Final captioned video created: {captioned_output_path}")
            
            except Exception as e:
                logger.error(f"Error processing clip {clip_file}: {str(e)}")
                continue
            
            finally:
                # Clean up
                if 'clip' in locals():
                    clip.close()
                if 'processed_clip' in locals():
                    processed_clip.close()
                if 'captioned_clip' in locals():
                    captioned_clip.close()
                if 'audio_path' in locals() and os.path.exists(audio_path):
                    os.remove(audio_path)
                if 'processed_clip_path' in locals() and os.path.exists(processed_clip_path):
                    os.remove(processed_clip_path)

    except Exception as e:
        logger.error(f"Error creating content from shorts: {str(e)}")
        raise

    logger.info("Content creation from shorts completed")
def detect_face(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        return faces[0]  # Returns (x, y, w, h) of the first detected face
    return None

def movingAverage(curve, radius):
    window_size = 2 * radius + 1
    f = np.ones(window_size) / window_size
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    curve_smoothed = curve_smoothed[radius:-radius]
    return curve_smoothed

def smooth(trajectory):
    smoothed_trajectory = np.copy(trajectory)
    for i in range(3):
        smoothed_trajectory[:, i] = movingAverage(trajectory[:, i], radius=50)
    return smoothed_trajectory

def fix_border(frame):
    s = frame.shape
    T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.04)
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame

def stabilize_video(clip: VideoFileClip) -> VideoFileClip:
    logger.info("Starting video stabilization")
    
    def get_frames(clip):
        for t in np.arange(0, clip.duration, 1/clip.fps):
            yield clip.get_frame(t)

    frames = list(get_frames(clip))
    frame_count = len(frames)
    
    transforms = np.zeros((frame_count - 1, 3), np.float32)
    prev = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)

    for i in range(1, frame_count):
        try:
            curr = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            prev_pts = cv2.goodFeaturesToTrack(prev, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
            
            if prev_pts is None or len(prev_pts) < 2:
                transforms[i-1] = transforms[i-2] if i > 1 else np.array([0, 0, 0])
                continue

            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev, curr, prev_pts, None)

            idx = np.where(status == 1)[0]
            if len(idx) < 2:
                transforms[i-1] = transforms[i-2] if i > 1 else np.array([0, 0, 0])
                continue

            prev_pts = prev_pts[idx]
            curr_pts = curr_pts[idx]

            m = cv2.estimateAffinePartial2D(prev_pts, curr_pts)[0]
            
            if m is None or not np.all(np.isfinite(m)):
                transforms[i-1] = transforms[i-2] if i > 1 else np.array([0, 0, 0])
            else:
                dx = m[0, 2]
                dy = m[1, 2]
                da = np.arctan2(m[1, 0], m[0, 0])
                transforms[i-1] = np.array([dx, dy, da])

            prev = curr
        except Exception as e:
            logger.error(f"Error processing frame {i}: {str(e)}")
            transforms[i-1] = transforms[i-2] if i > 1 else np.array([0, 0, 0])

    trajectory = np.cumsum(transforms, axis=0)
    smoothed_trajectory = smooth(trajectory)
    difference = smoothed_trajectory - trajectory
    transforms_smooth = transforms + difference

    def process_frame(frame, i):
        if i == 0:
            return frame
        
        try:
            dx, dy, da = transforms_smooth[i-1]
            m = np.zeros((2, 3), np.float32)
            m[0, 0] = np.cos(da)
            m[0, 1] = -np.sin(da)
            m[1, 0] = np.sin(da)
            m[1, 1] = np.cos(da)
            m[0, 2] = dx
            m[1, 2] = dy

            stabilized_frame = cv2.warpAffine(frame, m, (frame.shape[1], frame.shape[0]))
            stabilized_frame = fix_border(stabilized_frame)
            return stabilized_frame
        except Exception as e:
            logger.error(f"Error stabilizing frame {i}: {str(e)}")
            return frame

    stabilized_frames = [process_frame(frame, i) for i, frame in enumerate(frames)]
    
    def make_frame(t):
        frame_index = int(t * clip.fps)
        if frame_index >= len(stabilized_frames):
            frame_index = len(stabilized_frames) - 1
        return stabilized_frames[frame_index]

    logger.info("Video stabilization completed")
    return clip.fl_image(make_frame)


def autoframe_and_stabilize_clip(clip: VideoFileClip, target_aspect_ratio: float = 9/16, smoothing_window: int = 30) -> VideoFileClip:
    try:
        stabilized_clip = stabilize_video(clip)
    except Exception as e:
        logger.error(f"Error during video stabilization: {str(e)}. Proceeding without stabilization.")
        stabilized_clip = clip

    center_x_queue = deque(maxlen=smoothing_window)
    center_y_queue = deque(maxlen=smoothing_window)

    # Get the original video dimensions
    orig_height, orig_width = clip.h, clip.w

    # Calculate the target width based on the original height and desired aspect ratio
    target_width = int(orig_height * target_aspect_ratio)

    def process_frame(frame):
        face = detect_face(frame)
        
        frame_h, frame_w = frame.shape[:2]
        
        if face is not None:
            x, y, w, h = face
            center_x, center_y = x + w//2, y + h//2
        else:
            center_x, center_y = frame_w//2, frame_h//2

        center_x_queue.append(center_x)
        center_y_queue.append(center_y)

        smooth_center_x = int(sum(center_x_queue) / len(center_x_queue))
        smooth_center_y = int(sum(center_y_queue) / len(center_y_queue))

        # Calculate crop boundaries
        x1 = max(0, smooth_center_x - target_width//2)
        x2 = min(frame_w, x1 + target_width)

        # Ensure we don't go out of bounds
        if x2 == frame_w:
            x1 = max(0, x2 - target_width)
        
        # Crop the frame
        cropped_frame = frame[:, x1:x2]
        
        # Resize only if necessary, using high-quality interpolation
        if cropped_frame.shape[:2] != (orig_height, target_width):
            cropped_frame = cv2.resize(cropped_frame, (target_width, orig_height), 
                                       interpolation=cv2.INTER_LANCZOS4)
        
        return cropped_frame

    return stabilized_clip.fl_image(process_frame)


def detect_lip_movement(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mouth.xml')
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        mouths = mouth_cascade.detectMultiScale(roi_gray, 1.7, 11)
        
        if len(mouths) > 0:
            return True
    
    return False

def create_caption_image(text: str, highlight_word: str, size: Tuple[int, int]) -> np.ndarray:
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 24)
    image = Image.new('RGBA', size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    
    words = text.split()
    x, y = 10, size[1] - 40  # Position at the bottom
    for word in words:
        word_width, word_height = draw.textsize(word + " ", font=font)
        if x + word_width > size[0] - 10:
            x = 10
            y -= word_height + 5
        if word.lower() == highlight_word.lower():
            draw.rectangle([x, y, x + word_width, y + word_height], fill=(255, 255, 0, 128))
        draw.text((x, y), word, font=font, fill=(255, 255, 255, 255))
        x += word_width
    return np.array(image)

def sync_captions_with_lip_movement(clip: VideoFileClip, transcript: dict) -> CompositeVideoClip:
    def make_frame(t):
        current_frame = clip.get_frame(t)
        
        # Find the current segment and word
        current_segment = next((seg for seg in transcript['segments'] if seg['start'] <= t < seg['end']), None)
        if current_segment:
            words = current_segment['text'].split()
            word_duration = (current_segment['end'] - current_segment['start']) / len(words)
            current_word_index = int((t - current_segment['start']) / word_duration)
            current_word = words[min(current_word_index, len(words) - 1)]
            
            # Check for lip movement
            lip_moving = detect_lip_movement(current_frame)
            
            if lip_moving:
                caption = create_caption_image(current_segment['text'], current_word, (clip.w, clip.h))
                return np.maximum(current_frame, caption)
        
        return current_frame

    return clip.fl(make_frame)

def process_clip_with_captions(clip_path: str, transcript: dict, output_path: str):
    clip = VideoFileClip(clip_path)
    captioned_clip = sync_captions_with_lip_movement(clip, transcript)
    captioned_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
    clip.close()
    captioned_clip.close()

