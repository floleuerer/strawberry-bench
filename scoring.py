"""
Answer extraction and scoring for StrawberryBench.

The model is asked to return an integer. In practice it may return:
  - "3"
  - "three"
  - "The letter appears 3 times."
  - "ANSWER: 3"
  - "There are 3 r's in strawberry."

This module extracts the integer from free-form responses and computes accuracy.
"""

import re
from dataclasses import dataclass, field

# Mapping of number words to integers (0–20 covers all realistic letter counts)
NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20,
}


def extract_answer(response: str) -> int | None:
    """
    Extract an integer count from a model response.

    Priority:
      1. "ANSWER: <n>" pattern (chain-of-thought)
      2. First integer found in the response
      3. Number word (e.g., "three")
    Returns None if no integer could be extracted.
    """
    text = response.strip()

    # 1. Explicit ANSWER: prefix (chain-of-thought strategy)
    m = re.search(r"ANSWER\s*:\s*(\d+)", text, re.IGNORECASE)
    if m:
        return int(m.group(1))

    # 2. First standalone integer in the text
    m = re.search(r"\b(\d+)\b", text)
    if m:
        return int(m.group(1))

    # 3. Number words
    lower = text.lower()
    for word, value in NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", lower):
            return value

    return None


@dataclass
class EvalResult:
    example_id: str
    word: str
    letter: str
    difficulty: str
    expected: int
    predicted: int | None
    raw_response: str
    llm_extraction: bool = False
    correct: bool = field(init=False)

    def __post_init__(self) -> None:
        self.correct = self.predicted == self.expected


@dataclass
class BenchmarkSummary:
    model_id: str
    strategy: str
    total: int
    correct: int
    results_by_difficulty: dict[str, dict]  # difficulty -> {total, correct, accuracy}

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "strategy": self.strategy,
            "total": self.total,
            "correct": self.correct,
            "accuracy": round(self.accuracy, 4),
            "results_by_difficulty": self.results_by_difficulty,
        }


def compute_summary(
    model_id: str,
    strategy: str,
    results: list[EvalResult],
) -> BenchmarkSummary:
    difficulties = ["easy", "medium", "hard", "sentence", "paragraph", "names", "foreign"]
    by_diff: dict[str, dict] = {}

    for diff in difficulties:
        subset = [r for r in results if r.difficulty == diff]
        if not subset:
            continue
        n_correct = sum(1 for r in subset if r.correct)
        by_diff[diff] = {
            "total": len(subset),
            "correct": n_correct,
            "accuracy": round(n_correct / len(subset), 4),
        }

    return BenchmarkSummary(
        model_id=model_id,
        strategy=strategy,
        total=len(results),
        correct=sum(1 for r in results if r.correct),
        results_by_difficulty=by_diff,
    )
