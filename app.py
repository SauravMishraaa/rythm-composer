from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from transformers import pipeline

app = FastAPI()

# Function to generate lyrics 
def generate_lyrics(prompt):
    generator = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B')
    response = generator(prompt, max_length=50, temperature=0.7, do_sample=True)
    output = response[0]['generated_text']
    cleaned_output = output.replace("\n", " ")
    formatted_lyrics = f"♪ {cleaned_output} ♪"
    return formatted_lyrics

@app.post("/generate-lyrics")
async def generate_lyrics_endpoint(prompt: str = Form(...)):
    lyrics = generate_lyrics(prompt)
    return JSONResponse(content={"lyrics": lyrics})