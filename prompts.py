"""
Prompt strategies for StrawberryBench.

Three strategies are supported:
  zero_shot   — bare question, no examples
  few_shot    — 3 demonstrative examples before the question
  chain_of_thought — instruct the model to enumerate each letter position
"""

from enum import Enum


class Strategy(str, Enum):
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"


# System prompt used by all strategies
SYSTEM_PROMPT = (
    "You are a precise assistant. When asked to count letters in a word, "
    "respond with ONLY the integer count and nothing else."
)

COT_SYSTEM_PROMPT = (
    "You are a precise assistant. When asked to count letters in a word, "
    "first list each occurrence of the letter with its position, then state the final count "
    "as an integer on its own line prefixed with 'ANSWER:'."
)

FEW_SHOT_EXAMPLES = [
    {
        "question": "How many times does the letter 'l' appear in the word 'hello'?",
        "answer": "2",
    },
    {
        "question": "Count the occurrences of the letter 'a' in 'banana'.",
        "answer": "3",
    },
    {
        "question": "In the word 'programming', how many 'g's are there?",
        "answer": "2",
    },
]


def build_messages(question: str, strategy: Strategy) -> list[dict]:
    """Return the messages list to send to the OpenRouter chat completions API."""

    if strategy == Strategy.ZERO_SHOT:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

    if strategy == Strategy.FEW_SHOT:
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for ex in FEW_SHOT_EXAMPLES:
            messages.append({"role": "user", "content": ex["question"]})
            messages.append({"role": "assistant", "content": ex["answer"]})
        messages.append({"role": "user", "content": question})
        return messages

    if strategy == Strategy.CHAIN_OF_THOUGHT:
        cot_question = (
            f"{question}\n\n"
            "Think step by step: go through the word character by character, "
            "note each occurrence of the target letter, then give your final answer "
            "as 'ANSWER: <integer>'."
        )
        return [
            {"role": "system", "content": COT_SYSTEM_PROMPT},
            {"role": "user", "content": cot_question},
        ]

    raise ValueError(f"Unknown strategy: {strategy}")
