import torch
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

app = FastAPI(title="Assamese-Kannada NMT API")

# Allow your GitHub Pages frontend to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to your GitHub Pages URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model (Global scope so it loads once when server starts)
current_dir = os.path.dirname(os.path.abspath(__file__))
adapter_path = os.path.join(current_dir, "/home/jiro/Low-Resource-Cross-Translation/models/final_nllb_lora_adapters/")
base_model_name = "facebook/nllb-200-distilled-600M"

print("Loading model for API...")
base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

tokenizer.src_lang = "asm_Beng"
forced_bos_token_id = tokenizer.convert_tokens_to_ids("kan_Knda")
print("Model ready!")

# Define the data structure we expect from the frontend
class TranslationRequest(BaseModel):
    text: str

# Create the translation endpoint
@app.post("/translate")
async def translate(request: TranslationRequest):
    inputs = tokenizer(request.text, return_tensors="pt")
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs, 
            forced_bos_token_id=forced_bos_token_id,
            max_length=128
        )
    decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return {"translation": decoded}
