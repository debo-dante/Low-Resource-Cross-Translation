import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType

# --- 1. Setup Paths & Base Model ---
model_checkpoint = "facebook/nllb-200-distilled-600M"
data_path = "../data/train/samanantar_asm_kan_mined.csv"
output_dir = "../models/final_nllb_lora_adapters/"

# --- 2. Load Tokenizer (Separate Script Architecture) ---
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer.src_lang = "asm_Beng"
tokenizer.tgt_lang = "kan_Knda"

# --- 3. Load & Preprocess Data ---
dataset = load_dataset('csv', data_files=data_path)

def preprocess_function(examples):
    inputs = [ex for ex in examples['assamese_text']]
    targets = [ex for ex in examples['kannada_text']]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# --- 4. Load Model & Apply LoRA ---
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters() # Should show ~0.58% trainable

# --- 5. Training Arguments (Matching the IEEE Paper) ---
args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="no",
    learning_rate=5e-4,
    per_device_train_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    max_steps=600,
    predict_with_generate=True,
    fp16=True, # Enable mixed precision for GPU speed
    logging_steps=10,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# --- 6. Train and Save ---
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

print("Starting LoRA Fine-Tuning...")
trainer.train()
trainer.model.save_pretrained(output_dir)
print(f"Model saved successfully to {output_dir}")
