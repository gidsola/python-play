import torch
import argparse

from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
 
# Parse command-line arguments
parser = argparse.ArgumentParser();
parser.add_argument("text", type=str, help="prompt for image use");
parser.add_argument("url", type=str, nargs='?', default=None, help="URL of the image to describe");
args = parser.parse_args();

DEVICE = "cuda" if torch.cuda.is_available() else "cpu";

if args.url:
    image2 = load_image(args.url);
else:
    image2 = load_image("https://www.artificialintelligence-news.com/wp-content/uploads/2024/07/ai-7977960_1280.jpg");

# Initialize processor and model
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct");
model = AutoModelForVision2Seq.from_pretrained("HuggingFaceTB/SmolVLM-Instruct",
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager"
).to(DEVICE);

# Create input messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": args.text}
        ]
    },
];

# Prepare inputs
prompt = processor.apply_chat_template(messages, add_generation_prompt=True);
inputs = processor(text=prompt, images=[image2], return_tensors="pt");
inputs = inputs.to(DEVICE);

# Generate outputs
generated_ids = model.generate(**inputs, max_new_tokens=800);
generated_texts = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True
);

# Split the response to get the assistant text
parts = generated_texts[0].split("Assistant:")[1];

print(parts);
