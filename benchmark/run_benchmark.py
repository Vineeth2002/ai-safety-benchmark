# benchmark/run_benchmark.py
# Created by Vineeth Animireddy

import os, sys, pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchmark.utils import load_model, generate_response, load_prompts
from benchmark.evaluate import evaluate_results  # provided in evaluate.py

MODEL_NAME = "gpt2"
PROMPT_FILE = os.path.join(REPO_ROOT, "data", "sample_prompts.csv")
RESULT_FILE = os.path.join(REPO_ROOT, "results", "gpt2_outputs.csv")

def run():
    print(f"Running benchmark with model: {MODEL_NAME}")
    prompts = load_prompts(PROMPT_FILE)
    model, tokenizer = load_model(MODEL_NAME)

    os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)
    rows = []
    for prompt in prompts:
        print(f"> Prompt: {prompt}")
        response = generate_response(prompt, model, tokenizer)
        print(f"< Response: {response[:200]}\n")
        rows.append({"prompt": prompt, "response": response})

    pd.DataFrame(rows).to_csv(RESULT_FILE, index=False)
    print(f"\n✅ Saved results to {RESULT_FILE}")

    from benchmark.evaluate import evaluate_results
    evaluate_results(RESULT_FILE)

if __name__ == "__main__":
    run()
