import streamlit as st
import requests

st.title("🎤 AI Music Generator")

# Prompt input
prompt = st.text_area("Enter a prompt for the song lyrics:")

if st.button("Generate Lyrics"):
    with st.spinner("Generating lyrics..."):
        response = requests.post("http://localhost:8000/generate-lyrics", data={"prompt": prompt})
        lyrics = response.json()["lyrics"]
        st.text_area("Generated Lyrics", lyrics, height=200)

        # if st.button("Generate Audio"):
        #     with st.spinner("Generating audio..."):
        #         audio_response = requests.post("http://localhost:8000/generate-musicaudio", data={"lyrics": lyrics})
        #         with open("output.wav", "wb") as f:
        #             f.write(audio_response.content)
        #         st.audio("output.wav", format="audio/wav")
