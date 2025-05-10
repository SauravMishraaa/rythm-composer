import torch
import numpy as np
import scipy
import os
import uuid
import time
from nltk import sent_tokenize

# Allow GPU usage if available
if torch.cuda.is_available():
    print("Using GPU")
else:
    print("Using CPU")
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

# Preload models
print("Preloading models...")
preload_models()

# Silence padding between audio pieces
silence = np.zeros(int(0.25 * SAMPLE_RATE))

def split_sentences(text: str):
    """
    Splits input text into individual sentences using NLTK's sentence tokenizer.
    """
    import nltk
    nltk.download('punkt', quiet=True)
    return sent_tokenize(text)

def lyrics_to_audio(lyrics_text: str) -> str:
    """
    Converts each sentence of the input text to musical audio and saves as a combined .wav file.
    """
    print("Splitting lyrics into sentences...")
    sentences = split_sentences(lyrics_text)
    print(f"Total Sentences: {len(sentences)}")

    pieces = []

    for idx, sentence in enumerate(sentences):
        formatted_line = f"♪ {sentence.strip()} ♪"
        print(f"Generating audio for sentence {idx + 1}/{len(sentences)}: {formatted_line}")

        # Use the sentence as the prompt
        t0 = time.time()
        audio_array = generate_audio(formatted_line)
        generation_duration_s = time.time() - t0
        audio_duration_s = audio_array.shape[0] / SAMPLE_RATE

        print(f"→ Took {generation_duration_s:.1f}s to generate {audio_duration_s:.1f}s of audio.")

        # Append to the full audio with a bit of silence
        pieces.extend([audio_array, silence.copy()])

    # Concatenate all audio pieces
    final_audio = np.concatenate(pieces)

    # Save the final audio to a file
    filename = f"music_{uuid.uuid4().hex[:8]}.wav"
    output_dir = "audio_outputs"
    filepath = os.path.join(output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)
    scipy.io.wavfile.write(filepath, SAMPLE_RATE, final_audio)

    print(f"✅ Final musical audio successfully saved at: {filepath}")
    return filepath


if __name__ == "__main__":
    lyrics_to_audio(f"""The sun is shining brightly today.  
Birds are singing in the trees.  
I feel the wind brushing my face.  
Everything seems calm and peaceful.  
Let's take a walk and enjoy the day.
""")