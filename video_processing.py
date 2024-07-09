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
from collections import deque
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

async def process_video(video_path: str) -> VideoAnalysis:
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
                
                # Composite the main video
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

    def process_frame(frame):
        face = detect_face(frame)
        
        frame_h, frame_w = frame.shape[:2]
        target_w = int(frame_h * target_aspect_ratio)
        
        if face is not None:
            x, y, w, h = face
            center_x, center_y = x + w//2, y + h//2
        else:
            center_x, center_y = frame_w//2, frame_h//2

        center_x_queue.append(center_x)
        center_y_queue.append(center_y)

        smooth_center_x = int(sum(center_x_queue) / len(center_x_queue))
        smooth_center_y = int(sum(center_y_queue) / len(center_y_queue))

        x1 = max(0, smooth_center_x - target_w//2)
        x2 = min(frame_w, x1 + target_w)

        # Ensure we don't go out of bounds
        if x2 == frame_w:
            x1 = max(0, x2 - target_w)
        
        cropped_frame = frame[:, x1:x2]
        
        # Resize the frame to maintain consistent output size
        return cv2.resize(cropped_frame, (int(frame_h * target_aspect_ratio), frame_h))

    return stabilized_clip.fl_image(process_frame)

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
