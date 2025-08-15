# run_benchmark.py
# Created by Vineeth Animireddy

import os
import pandas as pd
from benchmark.utils import load_model, generate_response, load_prompts
from benchmark.evaluate import evaluate_results

MODEL_NAME = "gpt2"
PROMPT_FILE = "data/sample_prompts.csv"
RESULT_FILE = "results/gpt2_outputs.csv"

def run():
    print(f"Running benchmark with model: {MODEL_NAME}")
    prompts = load_prompts(PROMPT_FILE)
    model, tokenizer = load_model(MODEL_NAME)

    responses = []
    for prompt in prompts:
        print(f"> Prompt: {prompt}")
        response = generate_response(prompt, model, tokenizer)
        print(f"< Response: {response}\n")
        responses.append({"prompt": prompt, "response": response})

    df = pd.DataFrame(responses)
    os.makedirs("results", exist_ok=True)
    df.to_csv(RESULT_FILE, index=False)

    print(f"\n✅ Saved results to {RESULT_FILE}")
    evaluate_results(RESULT_FILE)

if __name__ == "__main__":
    run()
