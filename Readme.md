# Video Processing Microservice

This microservice processes videos to create short-form content suitable for platforms like LinkedIn, YouTube, and Instagram.

## Features

- Video transcription using Groq's Whisper API
- Content analysis and suggestion of short clips
- Automatic generation of video shorts with prescribed lengths
- Key frame extraction and text detection

## Setup

1. Ensure you have Python 3.8+ and Poetry installed on your system.

2. Clone this repository:
   ```
   git clone https://github.com/yourusername/video-processing-microservice.git
   cd video-processing-microservice
   ```

3. Install dependencies:
   ```
   poetry install
   ```

4. Set up environment variables:
   Create a `.env` file in the project root and add your Groq API keys:
   ```
   GROQ_API_KEYS=key1,key2,key3
   ```

5. Install system dependencies:
   - FFmpeg (for audio processing)
   - Tesseract (for OCR)

   On Ubuntu:
   ```
   sudo apt-get update
   sudo apt-get install ffmpeg tesseract-ocr
   ```

## Running the Service

1. Start the FastAPI server:
   ```
   poetry run uvicorn main:app --reload
   ```

2. The service will be available at `http://localhost:8000`

3. Use the `/process-video/` endpoint to upload and process videos.

## API Usage

POST `/process-video/`
- Upload a video file
- Returns analysis, transcript, and information about generated short clips

Example using curl:
```
curl -X POST "http://localhost:8000/process-video/" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@path/to/your/video.mp4"
```

## Development

- Run tests: `poetry run pytest`
- Format code: `poetry run black .`
- Check linting: `poetry run flake8`

## License

[MIT License](LICENSE)