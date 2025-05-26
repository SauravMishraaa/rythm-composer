from fastapi import FastAPI, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from lyrics import generate_lyrics

app = FastAPI()

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/generate-lyrics")
def generate_lyrics_endpoint(prompt: str = Form(...)):
    lyrics = generate_lyrics(prompt)
    return {"lyrics": lyrics}

# @app.post("/generate-musicaudio")
# def generate_audio_endpoint(lyrics: str = Form(...)):
#     filename = generate_audio_from_text(lyrics)
#     return FileResponse(filename, media_type="audio/wav", filename="song.wav")
