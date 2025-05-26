from bark import SAMPLE_RATE, generate_audio
from scipy.io.wavfile import write as write_wav
from IPython.display import Audio
import torch
import numpy as np

# Fix for PyTorch 2.6+ security restrictions
# Explicitly set weights_only to False in load_model function
def patch_torch_load():
    original_load = torch.load
    def patched_load(f, *args, **kwargs):
        kwargs['weights_only'] = False
        return original_load(f, *args, **kwargs)
    torch.load = patched_load

patch_torch_load()

# Now load models
from bark import preload_models
preload_models()

text_prompt = """
    [WOMAN] 
    ♪ Where midnight blooms and shadows bloom, the jungle breathes in night ♪
"""
audio_array = generate_audio(text_prompt)

# save audio to disk
write_wav("jungle3.wav", SAMPLE_RATE, audio_array)
  
# play text in notebook
Audio(audio_array, rate=SAMPLE_RATE)
