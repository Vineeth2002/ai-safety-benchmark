# benchmark/run_benchmark.py
# Created by Vineeth Animireddy

import os, sys, pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from benchmark.utils import load_model, generate_response, load_prompts
    from benchmark.evaluate import evaluate_results
except ModuleNotFoundError:
    # Fallback (still works if someone runs from inside benchmark/)
    if CURRENT_DIR not in sys.path:
        sys.path.insert(0, CURRENT_DIR)
    import utils
    import evaluate as _evaluate
    load_model = utils.load_model
    generate_response = utils.generate_response
    load_prompts = utils.load_prompts
    evaluate_results = _evaluate.evaluate_results

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
        print(f"> {prompt}")
        resp = generate_response(prompt, model, tokenizer)
        print(f"< {resp[:200]}\n")
        rows.append({"prompt": prompt, "response": resp})

    pd.DataFrame(rows).to_csv(RESULT_FILE, index=False)
    print(f"✅ Saved: {RESULT_FILE}")
    evaluate_results(RESULT_FILE)

if __name__ == "__main__":
    run()
