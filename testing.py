from bark import SAMPLE_RATE, generate_audio
import soundfile as sf
import torch
import numpy as np
import scipy
import os
import uuid
import time
from nltk import sent_tokenize
import nltk
# nltk.download('punkt_tab')
# Environment setup
if not torch.cuda.is_available():
    os.environ["SUNO_USE_SMALL_MODELS"] = "1"
    print("Using CPU with small models")
else:
    print("Using GPU")

from bark import generate_audio, preload_models, SAMPLE_RATE

# Fix for PyTorch 2.6+ to allow numpy scalar unpickling
torch.serialization.add_safe_globals([np.core.multiarray.scalar])

# Monkey-patch torch.load for Bark compatibility
_orig_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = patched_torch_load

# Music-style text prompt with notes
text_prompt = """
♪  In the jungle, the mighty lions sleep. ♪
"""

# Generate the audio
audio_array = generate_audio(text_prompt)

# Save the result as a WAV file
sf.write("bark_music_output1.wav", audio_array, SAMPLE_RATE)

