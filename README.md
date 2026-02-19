# 🍓 StrawberryBench

**The definitive benchmark for evaluating letter-counting ability in Large Language Models.**

> *"How many r's are in 'strawberry'?"* — The correct answer is **3**. This question went viral in 2023 when ChatGPT famously answered incorrectly, exposing a fundamental limitation tied to sub-word tokenization.

[![Dataset](https://img.shields.io/badge/🤗_Dataset-floleuerer/strawberry--bench-yellow)](https://huggingface.co/datasets/floleuerer/strawberry-bench)
[![Leaderboard](https://img.shields.io/badge/🏆_Leaderboard-strawberrybench.fyi-blue)](https://www.strawberrybench.fyi)

---

## Results (February 2026, Zero-Shot)

| # | Model | Overall | Easy | Medium | Hard | Sentences | Paragraphs |
|---|-------|---------|------|--------|------|-----------|------------|
| 🥇 | Gemini 3 Pro | **99.7%** | 100.0% | 99.6% | 99.3% | 100.0% | 100.0% |
| 🥈 | Grok 4 | 99.7% | 100.0% | 100.0% | 100.0% | 97.8% | 96.4% |
| 🥉 | Kimi K2.5 | 98.9% | 99.5% | 99.1% | 98.6% | 100.0% | 96.4% |
| 4 | Qwen3.5 397B | 98.9% | 100.0% | 99.6% | 99.3% | 95.7% | 94.5% |
| 5 | GLM-5 | 98.0% | 99.5% | 99.1% | 98.6% | 95.7% | 85.5% |
| 6 | Nemotron 3 Nano | 96.8% | 98.4% | 99.6% | 98.6% | 93.5% | 87.3% |
| 7 | GPT-5.2 Codex | 96.7% | 98.9% | 98.2% | 99.3% | 93.5% | 74.6% |
| 8 | Claude Opus 4.6 | 96.3% | 100.0% | 98.2% | 90.4% | 100.0% | 92.7% |
| 9 | MiniMax M2.5 | 94.8% | 98.4% | 100.0% | 99.3% | 80.4% | 52.7% |
| 10 | GPT-5.2 | 93.6% | 97.3% | 95.2% | 97.3% | 84.8% | 58.2% |
| 11 | Gemini 3 Flash | 89.5% | 96.7% | 94.7% | 89.0% | 71.7% | 47.3% |
| 12 | Claude Sonnet 4.6 | 73.8% | 95.6% | 86.4% | 62.3% | 69.6% | 58.2% |
| 13 | DeepSeek V3.2 | 54.5% | 76.5% | 57.0% | 48.6% | 26.1% | 18.2% |
| 14 | MiMo-V2-Flash | 46.6% | 84.7% | 52.2% | 29.4% | 17.4% | 18.2% |

Live leaderboard at **[strawberrybench.fyi](https://www.strawberrybench.fyi)**.

---

## Overview

StrawberryBench evaluates whether LLMs can count the occurrences of individual letters in words and phrases. The task is deceptively simple: it requires reasoning at the character level, which is non-trivial for tokenization-based models.

**560 questions** across four difficulty tiers:

| Tier | Word length | # Questions |
|------|-------------|-------------|
| Easy | 3–6 chars | 185 |
| Medium | 7–12 chars | 204 |
| Hard | 13+ chars | 132 |
| Sentence | phrase | 39 |

~25% of questions have answer 0 (the letter is absent from the word).

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/floleuerer/strawberry-bench
cd strawberry-bench
uv sync

# Set your OpenRouter API key
export OPENROUTER_API_KEY="sk-or-..."

# Run on a single model (loads dataset from HuggingFace automatically)
uv run python -m benchmark.run_benchmark --model openai/gpt-5.2

# Run on all registered models
uv run python -m benchmark.run_benchmark --all

# Quick smoke-test (first 20 questions)
uv run python -m benchmark.run_benchmark --model openai/gpt-5.2 --limit 20

# View the leaderboard locally
open website/index.html
```

Results are saved to `benchmark/results/<model>_<strategy>.json`.

---

## Dataset

The dataset is on HuggingFace and is loaded automatically by the benchmark runner.

```python
from datasets import load_dataset

ds = load_dataset("floleuerer/strawberry-bench", split="test")
print(ds[0])
# {'id': 'sb_00001', 'word': 'cat', 'letter': 't', 'question': "In the word 'cat', how many 't's are there?",
#  'answer': 1, 'difficulty': 'easy', 'word_length': 3, 'zero_count': False, 'template_idx': 1}
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | Unique identifier (`sb_XXXXX`) |
| `word` | str | Word or phrase to count in |
| `letter` | str | Single letter to count (lowercase) |
| `question` | str | Natural language question |
| `answer` | int | Correct count (ground truth) |
| `difficulty` | str | `easy` / `medium` / `hard` / `sentence` |
| `word_length` | int | Character count (excluding spaces) |
| `zero_count` | bool | Whether `answer == 0` |
| `template_idx` | int | Question template used (0 or 1) |

---

## Adding a Model

Register it in `benchmark/models.py`:

```python
ModelConfig(
    id="provider/model-name",   # OpenRouter model ID
    name="Display Name",
    provider="Provider",
    max_tokens=16,
    temperature=0.0,
),
```

Then run:

```bash
uv run python -m benchmark.run_benchmark --model provider/model-name
```

All models are called via [OpenRouter](https://openrouter.ai) for consistency.