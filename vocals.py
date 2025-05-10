# vocals.py

import torch
import numpy as np
import scipy
import os
import uuid
import time
from nltk import sent_tokenize

# Environment setup to use smaller models on the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["SUNO_USE_SMALL_MODELS"] = "1"

from bark import generate_audio, preload_models, SAMPLE_RATE

# Fix for PyTorch 2.6+ to allow numpy scalar unpickling
torch.serialization.add_safe_globals([np.core.multiarray.scalar])

# Monkey-patch torch.load to force weights_only=False for Bark compatibility
_orig_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = patched_torch_load

# Model Preloading (smaller models will be loaded on CPU)
print("Preloading smaller models on CPU...")
preload_models()

# Quarter second of silence for smooth transitions
silence = np.zeros(int(0.25 * SAMPLE_RATE))  

def split_into_chunks(text, max_length=200):
    """
    Splits the lyrics into chunks of max_length words for smooth processing.
    """
    import nltk
    nltk.download('punkt', quiet=True)

    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def lyrics_to_audio(lyrics_text: str) -> str:
    """Converts input lyrics to **music-like** audio in chunks and saves as a single WAV file."""
    
    print("Splitting lyrics into chunks...")
    chunks = split_into_chunks(lyrics_text)
    print(f"Total Chunks: {len(chunks)}")

    pieces = []

    for idx, chunk in enumerate(chunks):
        print(f"Generating musical audio for chunk {idx + 1}/{len(chunks)}...")

        # 🎵 **Magic Prompt for Music Generation**
        music_prompt = f"""
        ♪ [Verse 1]
        {chunk}
        
        ♪ [Chorus]
        {chunk}
        
        🎹 [Background Music]
        La la la... ♫ ♫ ♫
        """
        
        # Generate audio for the chunk
        t0 = time.time()
        audio_array = generate_audio(music_prompt)
        generation_duration_s = time.time() - t0
        audio_duration_s = audio_array.shape[0] / SAMPLE_RATE

        print(f"Took {generation_duration_s:.0f}s to generate {audio_duration_s:.0f}s of audio")

        # Append the generated chunk and silence to the final array
        pieces += [audio_array, silence.copy()]

    # Concatenate all audio pieces
    final_audio = np.concatenate(pieces)

    # Save the final audio to a file
    filename = f"music_{uuid.uuid4().hex[:8]}.wav"
    output_dir = "audio_outputs"
    filepath = os.path.join(output_dir, filename)

    os.makedirs(output_dir, exist_ok=True)
    scipy.io.wavfile.write(filepath, SAMPLE_RATE, final_audio)

    print(f"Final musical audio successfully generated and saved to: {filepath}")
    return filepath
