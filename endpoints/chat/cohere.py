from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "CohereForAI/c4ai-command-r-v01-4bit"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

# Format message with the c4ai-command-r7b-12-2024 chat template
messages = [{"role": "user", "content": "Hello, how are you?"}]
input_ids = tokenizer.apply_chat_template(
    messages, tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
)

gen_tokens = model.generate(
    input_ids,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.3
)

gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)

print(gen_text)
