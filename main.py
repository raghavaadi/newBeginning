import os
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from models import VideoAnalysis
from video_processing import process_video, create_content_from_shorts
from utils import initialize_app, setup_directories, cleanup_files
import signal
import sys
# Set up logging
interrupt_requested = False

def signal_handler(signum, frame):
    global interrupt_requested
    interrupt_requested = True
    print("Interrupt received. Stopping gracefully...")

# Set up the signal handler
signal.signal(signal.SIGINT, signal_handler)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# @app.on_event("startup")
# async def startup_event():
#     # initialize_app()


@app.post("/process-video/", response_model=VideoAnalysis)
async def process_video_endpoint(file: UploadFile = File(...)):
    global interrupt_requested
    if interrupt_requested:
        raise HTTPException(status_code=400, detail="Operation cancelled by user")
    
    logger.info(f"Received video for processing: {file.filename}")
    
    setup_directories()

    video_path = f"uploads/{file.filename}"
    with open(video_path, "wb") as buffer:
        buffer.write(await file.read())
    logger.info(f"Video saved to {video_path}")
    
    try:
        result = await process_video(video_path)
        if interrupt_requested:
            raise KeyboardInterrupt("Operation cancelled by user")
        return result
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
    finally:
        cleanup_files([video_path])

@app.post("/create-final-videos/")
async def create_final_videos():
    try:
        output_folder = "outputs"
        final_output_folder = "final_outputs"
        create_content_from_shorts(output_folder, final_output_folder)
        
        final_clips = [f for f in os.listdir(final_output_folder) if f.startswith("final_clip_") and f.endswith(".mp4")]
        return {"message": f"Final videos created successfully", "clips": final_clips}
    except Exception as e:
        logger.error(f"Error in create_final_videos endpoint: {str(e)}")
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

@app.get("/final-clip/{clip_id}")
async def get_final_clip(clip_id: int):
    logger.info(f"Request for final clip {clip_id}")
    clip_path = f"final_outputs/final_clip_{clip_id}.mp4"
    if not os.path.exists(clip_path):
        logger.warning(f"Final clip not found: {clip_path}")
        raise HTTPException(status_code=404, detail="Final clip not found")
    logger.info(f"Returning final clip: {clip_path}")
    return FileResponse(clip_path, media_type="video/mp4", filename=f"final_clip_{clip_id}.mp4")


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server")
    uvicorn.run(app, host="0.0.0.0", port=8000)