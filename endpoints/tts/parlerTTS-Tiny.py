import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

model = ParlerTTSForConditionalGeneration.from_pretrained(
    "parler-tts/parler-tts-tiny-v1"
    ).to(DEVICE)

tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-tiny-v1")

PROMPT = "Hey, how are you doing today?"
DESCRIPTION = "Monotone yet slightly fast in delivery, with a slight accent."

input_ids = tokenizer(DESCRIPTION, return_tensors="pt").input_ids.to(DEVICE)
prompt_input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(DEVICE)

generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
audio_arr = generation.cpu().numpy().squeeze()

sf.write("parler_tts_out.wav", audio_arr, model.config.sampling_rate)
