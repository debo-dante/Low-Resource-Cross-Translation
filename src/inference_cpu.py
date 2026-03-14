import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

# --- 1. Setup Paths ---
base_model_name = "facebook/nllb-200-distilled-600M"
adapter_path = "../models/final_nllb_lora_adapters/"

# --- 2. Load Model & Tokenizer for CPU ---
print("Loading base model to CPU RAM...")
model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

print("Applying LoRA adapters...")
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

# --- 3. Define language tags ---
tokenizer.src_lang = "asm_Beng"
forced_bos_token_id = tokenizer.convert_tokens_to_ids("kan_Knda")

# --- 4. Test Translation ---
text = "আজি বৰষুণ দিব পাৰে।"
print(f"\nAssamese Input: {text}")
print("Translating (this might take 1-3 seconds on CPU)...")

inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    generated_tokens = model.generate(
        **inputs, 
        forced_bos_token_id=forced_bos_token_id,
        max_length=100
    )

decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
print(f"Kannada Output: {decoded}")
