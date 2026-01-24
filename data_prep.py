# data_prep.py
from datasets import load_dataset
from transformers import AutoTokenizer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2")
parser.add_argument("--file", default="persona_train.jsonl")
parser.add_argument("--max_length", type=int, default=512)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
dataset = load_dataset("json", data_files=args.file)["train"]

def format_example(ex):
    prompt = ex["prompt"] + "\n"
    response = ex["response"]
    full = prompt + response
    toks = tokenizer(full, truncation=True, max_length=args.max_length)
    # label only the response tokens (or all tokens depending on approach)
    return {"input_ids": toks["input_ids"], "attention_mask": toks["attention_mask"]}

tokenized = dataset.map(format_example)
tokenized.save_to_disk("tokenized_persona")
print("Saved tokenized dataset.")
