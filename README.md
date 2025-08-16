Daily safety snapshot (tiny-gpt2 on live data) is available as workflow artifacts:
[![Daily Tiny Model Safety Snapshot](https://github.com/Vineeth2002/ai-safety-benchmark/actions/workflows/sweep_tiny_daily.yml/badge.svg)](https://github.com/Vineeth2002/ai-safety-benchmark/actions/workflows/sweep_tiny_daily.yml)


[![Daily Google Trends](https://github.com/Vineeth2002/ai-safety-benchmark/actions/workflows/trends_daily.yml/badge.svg)](https://github.com/Vineeth2002/ai-safety-benchmark/actions/workflows/trends_daily.yml)


Safety benchmark for LLM behavior (toxic, refusal, safe completions)

## 📊 Results (GPT-2, Aug 2025)

| Class   | Count |
|---------|------:|
| safe    | 2     |
| refusal | 0     |
| unsafe  | 3     |

Base GPT-2 (not instruction-tuned) produced **0 refusals** and **3 unsafe generations** across 5 prompts, confirming that safety layers or instruction-tuned models are needed for harmful/adversarial queries.

![GPT-2 Safety Summary](results/gpt2_summary.png)

### Model Comparison (Aug 2025)

| Model        | Safe | Refusal | Unsafe |
|--------------|-----:|--------:|------:|
| GPT-2        | 2    | 0       | 3     |
| DistilGPT-2  | 2    | 0       | 3     |

![DistilGPT-2 Safety Summary](results/distilgpt2_summary.png)


### Model Comparison (Aug 2025)
![GPT-2 vs DistilGPT-2](results/gpt_comparison_summary.png)

