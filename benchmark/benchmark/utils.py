# utils.py
# Created by Vineeth Animireddy

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd

def load_model(model_name="gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return model, tokenizer

def generate_response(prompt, model, tokenizer, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length, do_sample=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def load_prompts(csv_path):
    df = pd.read_csv(csv_path)
    return df["prompt"].tolist()
