# train_lora.py
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import argparse
from datasets import load_from_disk

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2")
parser.add_argument("--data_dir", default="tokenized_persona")
parser.add_argument("--output_dir", default="persona_chat_lora")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
# load base model in 4-bit if using bitsandbytes (use device_map="auto")
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    load_in_4bit=True,
    device_map="auto",
    trust_remote_code=True
)

# prepare for k-bit training (if 4-bit)
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj","v_proj","k_proj","o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

data = load_from_disk(args.data_dir)
train_dataset = data

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=20,
    save_total_limit=3,
    save_steps=500,
    optim="paged_adamw_32bit"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator
)

trainer.train()
model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)
print("Saved LoRA model.")
