# vocals.py

import torch
import numpy as np
import scipy
import os
import uuid

# Fix for PyTorch 2.6+ to allow numpy scalar unpickling
torch.serialization.add_safe_globals([np.core.multiarray.scalar])

# Monkey-patch torch.load to force weights_only=False for Bark compatibility
_orig_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = patched_torch_load

# Now import Bark
from bark import SAMPLE_RATE, generate_audio, preload_models

def lyrics_to_audio(lyrics_text: str) -> str:
    """Converts input lyrics to audio and saves as WAV file."""
    preload_models()  # Ensures Bark models are loaded
    audio_array = generate_audio(lyrics_text)

    filename = f"vocals_{uuid.uuid4().hex[:8]}.wav"
    output_dir = "audio_outputs"
    filepath = os.path.join(output_dir, filename)

    os.makedirs(output_dir, exist_ok=True)
    scipy.io.wavfile.write(filepath, SAMPLE_RATE, audio_array)

    return filepath
