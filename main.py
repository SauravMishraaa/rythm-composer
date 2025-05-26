from fastapi import FastAPI, Form
from fastapi.responses import FileResponse, JSONResponse
from lyrics import generate_lyrics
from music import lyrics_to_audio

app = FastAPI(title="Lyrics & Voice Generator API")

@app.post("/generate-lyrics", tags=["Lyrics"])
async def generate_lyrics_endpoint(prompt: str = Form(...)):
    lyrics = generate_lyrics(prompt)
    return JSONResponse(content={"lyrics": lyrics})
    
@app.post("/generate-music")
async def generate_music(prompt: str = Form(...)):
    audio_path = lyrics_to_audio(prompt)
    return FileResponse(path=audio_path, filename="generated_music.wav", media_type="audio/wav")
