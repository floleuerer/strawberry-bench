---
license: cc-by-4.0
task_categories:
  - question-answering
language:
  - en
tags:
  - benchmark
  - evaluation
  - letter-counting
  - tokenization
  - reasoning
size_categories:
  - n<1K
pretty_name: StrawberryBench
---

# StrawberryBench Dataset

## Dataset Description

StrawberryBench is a benchmark dataset for evaluating the ability of Large Language Models to count letter occurrences in words and phrases. The task appears trivial but exposes a fundamental limitation: LLMs operate on sub-word tokens, not individual characters, making direct character-level counting non-trivial.

**The canonical example:** "How many r's are in 'strawberry'?" (answer: **3**). This question went viral when ChatGPT famously answered incorrectly, revealing a systematic weakness tied to Byte-Pair Encoding (BPE) tokenization.

### Supported Tasks

- `letter-counting`: Given a word or phrase and a target letter, count how many times the letter appears.

### Languages

English (en)

## Dataset Structure

### Data Instances

```json
{
  "id": "sb_00001",
  "word": "cat",
  "letter": "t",
  "question": "In the word 'cat', how many 't's are there?",
  "answer": 1,
  "difficulty": "easy",
  "word_length": 3,
  "zero_count": false,
  "template_idx": 1
}
```

### Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique example identifier (`sb_XXXXX`) |
| `word` | string | The word or phrase to count in |
| `letter` | string | The single letter to count (lowercase) |
| `question` | string | Natural language question |
| `answer` | int | Correct count (ground truth) |
| `difficulty` | string | `easy` / `medium` / `hard` / `sentence` / `paragraph` / `names` / `foreign` |
| `word_length` | int | Number of non-space characters in `word` |
| `zero_count` | bool | Whether `answer == 0` (letter absent) |
| `template_idx` | int | Which question template was used (0 or 1) |
| `language` | string | For `foreign` difficulty, the language (optional) |

### Data Splits

| Split | Examples |
|-------|----------|
| train | — |
| test | 847 |

The dataset is released as a single test split only; it is designed for zero-shot / few-shot evaluation, not fine-tuning.

### Difficulty Tiers

| Tier | Description / Word Length | # Examples |
|------|-------------|------------|
| easy | 3–6 chars | 183 |
| medium | 7–12 chars | 228 |
| hard | 13+ chars | 146 |
| sentence | short multi-word phrases | 46 |
| paragraph | longer text passages (100+ chars) | 55 |
| names | common first names (repeated letters) | 112 |
| foreign | non-English words (German, etc.) | 77 |

Approximately 11% of examples have `zero_count = true` (the letter does not appear in the word).

## Dataset Creation

### Source Data

All words and phrases are common English vocabulary, proper nouns, and well-known idioms. No copyrighted text is included. Ground-truth answers are computed deterministically using Python `str.count()`.

### Question Templates

Two main templates are used per example (randomly assigned index 0 or 1):

- **Standard (easy, medium, hard):**
  1. `"How many times does the letter '{letter}' appear in the word '{word}'?"`
  2. `"In the word '{word}', how many '{letter}'s are there?"`
- **Sentences:** Uses "phrase" instead of "word".
- **Paragraphs:** Uses "following text: '{word}'".
- **Names:** `"How many {letter}'s are in the name '{word}'?"`
- **Foreign:** `"How many times does the letter '{letter}' appear in the {language} word '{word}'?"`

### Curation Rationale

The dataset was created to provide a controlled, reproducible evaluation of character-level reasoning in LLMs. Difficulty tiers allow fine-grained analysis of how word length and letter frequency affect performance.

## Evaluation

Scoring is **exact match** on the integer answer. A fuzzy parser handles written-out number words ("three" → 3). Partial credit is not awarded.

```python
from datasets import load_dataset

ds = load_dataset("floleuerer/strawberry-bench")

# Evaluate a model
correct = sum(1 for ex in ds["test"] if model_predict(ex["question"]) == ex["answer"])
accuracy = correct / len(ds["test"])
```

## License

[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
