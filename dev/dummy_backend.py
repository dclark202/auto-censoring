# dummy_backend.py
import uvicorn
import time
import uuid
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pydub import AudioSegment

# --- Configuration ---
# This dummy backend will create a small, silent mp3 file on startup to serve for downloads.
DUMMY_MP3_PATH = "dummy_edited_file.mp3"

# --- App Initialization ---
app = FastAPI(title="Dummy FSP Finder Backend")

# --- Helper Function to create a dummy audio file ---
def create_dummy_mp3():
    """Generates a 1-second silent MP3 file if it doesn't exist."""
    if not os.path.exists(DUMMY_MP3_PATH):
        print(f"Creating a dummy MP3 file at: {DUMMY_MP3_PATH}")
        one_second_of_silence = AudioSegment.silent(duration=1000) # duration in milliseconds
        one_second_of_silence.export(DUMMY_MP3_PATH, format="mp3")
        print("Dummy file created successfully.")

# --- Pydantic Model for the /finalize endpoint ---
class FinalizeRequest(BaseModel):
    job_id: str

# --- Dummy API Endpoints ---

@app.post("/analyze/")
async def dummy_analyze_file(file: UploadFile = File(...)):
    """
    Simulates the analysis process by waiting for a few seconds and returning
    a hardcoded, fake analysis result that matches the frontend's expected structure.
    """
    print(f"Dummy Backend: Received file '{file.filename}' for analysis.")
    print("Simulating ML model processing for 3 seconds...")
    time.sleep(3)

    # This is the fake data structure. You can modify the values here to test
    # different scenarios in your frontend (e.g., more explicit words, LLM errors, etc.)
    dummy_results = {
        "job_id": f"dummy-job-{uuid.uuid4()}",
        "filename": file.filename,
        "metadata": {
            "title": "Greatest Hit",
            "artist": "The Big Dummies",
            "album": "Dinguses and Dongles",
            "year": "2025",
            "genius_url": "https://github.com/dclark202/auto-censoring",
            "wer_score": "0.123",
            "genius_lyrics": "This is a fake song for testing the frontend."
        },
        "transcript": [
            {
                "line_words": [
                    {'id': [0, 0], 'text': 'This', 'start': 1.0, 'end': 1.2},
                    {'id': [0, 1], 'text': 'is', 'start': 1.3, 'end': 1.4},
                    {'id': [0, 2], 'text': 'a', 'start': 1.4, 'end': 1.5},
                    {'id': [0, 3], 'text': 'clean', 'start': 1.6, 'end': 2.0},
                    {'id': [0, 4], 'text': 'line.', 'start': 2.1, 'end': 2.5},
                ],
                "line_text": "This is a clean line.", 'start': 1.0, 'end': 2.5
            },
            {
                "line_words": [
                    {'id': [1, 0], 'text': 'But', 'start': 3.0, 'end': 3.2},
                    {'id': [1, 1], 'text': 'this', 'start': 3.3, 'end': 3.5},
                    {'id': [1, 2], 'text': 'line', 'start': 3.6, 'end': 4.0},
                    {'id': [1, 3], 'text': 'is', 'start': 4.1, 'end': 4.2},
                    {'id': [1, 4], 'text': 'fridgin', 'start': 4.3, 'end': 4.8}, # A "bad" word
                    {'id': [1, 5], 'text': 'explicit.', 'start': 4.9, 'end': 5.5},
                ],
                "line_text": "But this line is heckin explicit.", 'start': 3.0, 'end': 5.5
            },
            {
                "line_words": [
                    {'id': [2, 0], 'text': 'This', 'start': 6.0, 'end': 6.2},
                    {'id': [2, 1], 'text': 'line', 'start': 6.3, 'end': 6.5},
                    {'id': [2, 2], 'text': 'has', 'start': 6.6, 'end': 6.8},
                    {'id': [2, 3], 'text': 'a', 'start': 6.8, 'end': 6.9},
                    {'id': [2, 4], 'text': 'simulated', 'start': 7.0, 'end': 7.5},
                    {'id': [2, 5], 'text': 'LLM', 'start': 7.6, 'end': 7.9},
                    {'id': [2, 6], 'text': 'error.', 'start': 8.0, 'end': 8.5},
                ],
                "line_text": "This line has a simulated LLM error.", 'start': 6.0, 'end': 8.5
            }
        ],
        "initial_explicit_ids": [
            [1, 4]  
        ],
        "line_errs": [
            2 # Corresponds to the third line (index 2), will show a '⚠️'
        ]
    }
    
    print("Dummy analysis complete. Sending mock data to frontend.")
    return dummy_results

@app.post("/finalize/")
async def dummy_finalize_file(request: FinalizeRequest):
    """
    Simulates the finalization process by returning a pre-made silent MP3 file.
    """
    print(f"Dummy Backend: Received request to finalize job_id: {request.job_id}")
    print("Simulating audio processing for 2 seconds...")
    time.sleep(2)
    
    if not os.path.exists(DUMMY_MP3_PATH):
        raise HTTPException(status_code=404, detail="Dummy MP3 file not found.")

    # Generate a unique filename for the download
    base_name = f"dummy-file-{str(request.job_id)[:4]}-edited.mp3"

    print(f"Finalization complete. Sending dummy file '{base_name}' for download.")
    return FileResponse(path=DUMMY_MP3_PATH, media_type='audio/mpeg', filename=base_name)

@app.on_event("startup")
def on_startup():
    """Create the dummy file when the server starts."""
    create_dummy_mp3()

# --- Main execution block to run the server ---
if __name__ == "__main__":
    print("Starting Dummy FastAPI backend server...")
    uvicorn.run(app, host="127.0.0.1", port=8000)