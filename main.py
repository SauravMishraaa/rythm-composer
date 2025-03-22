from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from lyrics import generate_lyrics

app = FastAPI(title="Lyrics Generator API", 
              description="API for generating song lyrics")

@app.post("/generate-lyrics", tags=["Lyrics"])
async def generate_lyrics_endpoint(prompt: str = Form(...)):
    
    lyrics = generate_lyrics(prompt)
    return JSONResponse(content={"lyrics": lyrics})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)