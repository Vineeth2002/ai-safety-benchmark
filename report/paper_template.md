# AI Safety Benchmark for Language Models

**Author**: Vineeth Animireddy  
**Created**: August 2025  
**GitHub**: [@Vineeth2002](https://github.com/Vineeth2002)

---

## Abstract

This paper presents a lightweight, reproducible benchmark to evaluate the safety behavior of open-source language models. We assess how models respond to both benign and harmful prompts, focusing on their ability to refuse unsafe completions and avoid toxic outputs.

---

## 1. Introduction

The rapid deployment of large language models (LLMs) introduces new challenges around safety and responsible usage. This benchmark provides a minimal framework for scoring model behavior on safety metrics such as refusal, toxicity, and appropriateness.

---

## 2. Methodology

- Load models via HuggingFace
- Use a small but diverse prompt set
- Generate and log responses
- Evaluate based on keyword matches and refusal phrases

---

## 3. Experiments

- Model: GPT-2 (baseline)
- Dataset: 5 custom prompts
- Outputs: CSV logs + classified responses
- Metrics: % safe / unsafe / refusal

---

## 4. Results

_TODO: Add summary table here after running the benchmark._

---

## 5. Future Work

- Evaluate newer models (Mistral, Phi-2, TinyLlama)
- Add toxicity scoring (e.g., Detoxify)
- Add reward model for alignment feedback

---

## 6. Conclusion

This benchmark acts as a starting point for reproducible safety evaluations. Future improvements can turn this into a formal dataset + research toolkit.

---

## Appendix

- Codebase: [GitHub Repo](https://github.com/Vineeth2002/ai-safety-benchmark)
