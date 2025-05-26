import transformers, torch, builtins, numpy

processor = transformers.AutoProcessor.from_pretrained('chaowenguo/musicgen')
model = transformers.MusicgenMelodyForConditionalGeneration.from_pretrained('chaowenguo/musicgen').to('cuda')

result = []
for _ in builtins.range(9):
    inputs = processor(audio=result[-1] if result else None, sampling_rate=model.config.audio_encoder.sampling_rate, text='loud romance piano and violin background music for dancing', padding=True, return_tensors='pt').to('cuda')
    audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=1000)
    result += audio_values[0, 0].cpu().numpy(),

from IPython.display import Audio
Audio(numpy.concatenate(result), rate=model.config.audio_encoder.sampling_rate)
