from fastapi import FastAPI, Form
from fastapi.responses import FileResponse, JSONResponse
from lyrics import generate_lyrics
from vocals import lyrics_to_audio

app = FastAPI(title="Lyrics & Voice Generator API")

@app.post("/generate-lyrics", tags=["Lyrics"])
async def generate_lyrics_endpoint(prompt: str = Form(...)):
    lyrics = generate_lyrics(prompt)
    return JSONResponse(content={"lyrics": lyrics})


@app.post("/generate-song", tags=["Lyrics + Audio"])
async def generate_full_song(prompt: str = Form(...)):
    lyrics = generate_lyrics(prompt)
    audio_path = lyrics_to_audio(lyrics)
    return FileResponse(path=audio_path, filename="generated_song.wav", media_type="audio/wav")
