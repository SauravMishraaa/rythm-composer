# vocals.py

import torch
import numpy as np
import scipy
import os
import uuid
import time

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

def lyrics_to_audio(lyrics_text: str) -> str:
    """Converts input lyrics to audio and saves as WAV file."""
    
    print("Generating audio...")
    t0 = time.time()
    audio_array = generate_audio(lyrics_text)
    generation_duration_s = time.time() - t0
    audio_duration_s = audio_array.shape[0] / SAMPLE_RATE

    print(f"Took {generation_duration_s:.0f}s to generate {audio_duration_s:.0f}s of audio")

    filename = f"vocals_{uuid.uuid4().hex[:8]}.wav"
    output_dir = "audio_outputs"
    filepath = os.path.join(output_dir, filename)

    os.makedirs(output_dir, exist_ok=True)
    scipy.io.wavfile.write(filepath, SAMPLE_RATE, audio_array)

    return filepath
