# benchmark/utils.py
# Created by Vineeth Animireddy

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, pandas as pd

def load_model(model_name="gpt2"):
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForCausalLM.from_pretrained(model_name)
    mdl.eval()
    return mdl, tok

def generate_response(prompt, model, tokenizer, max_new_tokens=80):
    inp = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=True)
    return tokenizer.decode(out[0], skip_special_tokens=True)

def load_prompts(path):
    return pd.read_csv(path)["prompt"].astype(str).tolist()

