import torch
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

# --- 1. Bulletproof Path Setup ---
# This automatically finds the correct path regardless of where you run the script from
current_dir = os.path.dirname(os.path.abspath(__file__))
adapter_path = os.path.join(current_dir, "/home/jiro/Low-Resource-Cross-Translation/models/final_nllb_lora_adapters/")
base_model_name = "facebook/nllb-200-distilled-600M"

# --- 2. Load Model & Tokenizer for CPU ---
print("Loading base model to CPU RAM (this might take a moment)...")
model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

print("Applying LoRA adapters...")
try:
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
except Exception as e:
    print(f"\n Error loading adapters from {adapter_path}")
    print("Ensure you extracted your trained adapter files into the 'models/nllb-assamese-kannada-lora' folder.")
    exit()

# --- 3. Define language tags ---
tokenizer.src_lang = "asm_Beng"
forced_bos_token_id = tokenizer.convert_tokens_to_ids("kan_Knda")

print("\n" + "="*60)
print("Assamese-Kannada Neural Translator Ready!")
print("Type your Assamese sentence below and press Enter.")
print("Type 'quit' or 'exit' to close the program.")
print("="*60 + "\n")

# --- 4. Interactive Translation Loop ---
while True:
    text = input("Enter Assamese text: ")
    
    # Check for exit commands
    if text.strip().lower() in ['quit', 'exit']:
        print("Closing translator. Goodbye!")
        break
        
    # Ignore empty inputs
    if not text.strip():
        continue
        
    print("Translating...")
    
    # Tokenize and Generate
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs, 
            forced_bos_token_id=forced_bos_token_id,
            max_length=128
        )

    # Decode and print output
    decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    print(f"Kannada Output:      {decoded}\n")
    print("-" * 60)
