from fastapi import FastAPI, Form
from fastapi.responses import FileResponse, JSONResponse
from lyrics import generate_lyrics
from vocals import lyrics_to_audio

app = FastAPI(title="Lyrics & Voice Generator API")

@app.post("/generate-lyrics", tags=["Lyrics"])
async def generate_lyrics_endpoint(prompt: str = Form(...)):
    lyrics = generate_lyrics(prompt)
    return JSONResponse(content={"lyrics": lyrics})


@app.post("/generate-music")
async def generate_music(prompt: str = Form(...)):
    audio_path = lyrics_to_audio(prompt)
    return FileResponse(path=audio_path, filename="generated_music.wav", media_type="audio/wav")
main.py

# from fastapi import FastAPI, Form
# from fastapi.responses import FileResponse
# from transformers import pipeline
# import scipy
# import os
# import uuid

# app = FastAPI()

# # Initialize the MusicGen pipeline
# musicgen = pipeline("text-to-audio", model="facebook/musicgen-small")

# @app.post("/generate-music")
# async def generate_music(prompt: str = Form(...), duration: int = Form(10)):
#     # Generate music using the prompt
#     music = musicgen(prompt, forward_params={"do_sample": True, "max_new_tokens": duration * 50})

#     # Save the generated audio to a file
#     filename = f"music_{uuid.uuid4().hex[:8]}.wav"
#     output_dir = "audio_outputs"
#     os.makedirs(output_dir, exist_ok=True)
#     filepath = os.path.join(output_dir, filename)
#     scipy.io.wavfile.write(filepath, rate=music["sampling_rate"], data=music["audio"])

#     return FileResponse(path=filepath, filename=filename, media_type="audio/wav")
