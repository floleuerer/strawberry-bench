#!/usr/bin/env python3
"""
Generate the StrawberryBench dataset.

Produces 500+ letter-counting questions across six difficulty tiers:
  easy      — short words (3–6 chars)
  medium    — medium words (7–12 chars)
  hard      — long words (13+ chars)
  sentence  — short multi-word phrases
  paragraph — longer prose passages (100+ chars, no famous quotes)
  foreign   — hard non-English words (German, French, Spanish, Italian,
               Dutch, Welsh, Finnish, Portuguese)

Run:
    uv run python data/generate_dataset.py
"""

import json
import csv
import random
from pathlib import Path

SEED = 42
OUTPUT_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Word lists by difficulty
# ---------------------------------------------------------------------------

EASY_WORDS = [
    "cat", "dog", "bee", "see", "add", "egg", "inn", "odd", "all",
    "apple", "grape", "lemon", "melon", "olive", "peach", "plum",
    "hello", "world", "water", "light", "night", "right", "plant",
    "pizza", "queen", "fuzzy", "jazzy", "happy", "sunny", "funny",
    "book", "good", "look", "moon", "pool", "tool", "cool", "wool",
    "tree", "free", "knee", "flee", "three",
    "miss", "kiss", "hiss", "fizz", "buzz",
]

MEDIUM_WORDS = [
    "strawberry", "raspberry", "blueberry", "cranberry", "gooseberry",
    "programming", "algorithm", "database", "computer", "keyboard",
    "beautiful", "wonderful", "excellent", "fantastic", "important",
    "committee", "occurrence", "necessary", "beginning", "different",
    "chocolate", "cinnamon", "avocado", "broccoli", "cucumber",
    "elephant", "alligator", "butterfly", "dragonfly", "centipede",
    "mississippi", "tennessee", "connecticut", "massachusetts",
    "abracadabra", "hippopotamus",
    "vocabulary", "technology", "democracy", "astronomy", "geography",
    "expression", "impression", "possession", "profession", "obsession",
    "recommend", "accelerate", "accessible", "accomplish", "accumulate",
]

HARD_WORDS = [
    "congratulations", "approximately", "establishment", "simultaneously",
    "accommodation", "onomatopoeia", "encyclopedia", "rhinoceros",
    "characteristics", "responsibilities", "opportunities",
    "implementation", "representation", "collaboration", "communication",
    "disappointment", "embarrassment", "Mediterranean",
    "conscientious", "unbelievable", "understanding", "uncomfortable",
    "administration", "international", "misunderstanding",
    "revolutionary", "extraordinary", "parliamentary", "parliamentary",
    "acknowledgment", "circumstances", "consciousness",
    "supercalifragilistic",
]

SENTENCES = [
    "the library closes at nine o clock on friday evenings",
    "she put her red umbrella next to the front door",
    "the children played outside until the street lights came on",
    "he forgot to buy milk when he went to the store",
    "the train arrives on platform three every morning at half past seven",
    "my neighbor has a small garden with tomatoes and herbs",
    "the cat knocked the glass off the kitchen counter again",
    "we ate dinner outside because the weather was so nice",
    "the package arrived two days later than expected",
    "she wrote her phone number on a yellow sticky note",
]

# Longer prose passages — no famous quotes, dense repeated letters
PARAGRAPHS = [
    "the ambitious astronaut carefully adjusted all available apparatus before attempting an astonishing atmospheric analysis above earth",
    "scientific research consistently suggests that regular reading substantially strengthens reasoning skills and expands vocabulary across many subjects",
    "the persistent programmer patiently practiced writing efficient algorithms repeatedly until the complex solution seemed perfectly natural and clear",
    "archaeologists meticulously catalogue each artifact according to established classification criteria at excavation sites around the mediterranean",
    "the total population of the metropolitan area dramatically increased over recent decades attracting considerable international business investment",
    "simultaneously balancing multiple competing responsibilities requires exceptional organizational abilities and consistent thoughtful prioritization strategies",
    "mississippi and massachusetts are notoriously difficult to spell correctly because of their unusual repeated letter patterns and sequences",
    "the enthusiastic committee enthusiastically recommended implementing a comprehensive accessibility improvement plan covering all public facilities immediately",
    "thousands of extraordinarily talented musicians performed simultaneously during the spectacular week-long international summer music festival celebration",
    "conscientious environmental scientists consistently collect detailed measurements carefully documenting the concerning accelerating changes in global ocean temperatures",
]

# ---------------------------------------------------------------------------
# Question templates
# ---------------------------------------------------------------------------

TEMPLATES = [
    "How many times does the letter '{letter}' appear in the word '{word}'?",
    "In the word '{word}', how many '{letter}'s are there?",
]

SENTENCE_TEMPLATES = [
    "How many times does the letter '{letter}' appear in the phrase '{word}'?",
    "In the phrase '{word}', how many '{letter}'s are there?",
]

PARAGRAPH_TEMPLATES = [
    "How many times does the letter '{letter}' appear in the following text: '{word}'?",
    "In the following text: '{word}', how many '{letter}'s are there?",
]

# Common first names chosen for repeated or tricky letter patterns.
NAMES = [
    "Jennifer",
    "Christopher",
    "Stephanie",
    "Cassandra",
    "Annabelle",
    "Nathaniel",
    "Maximilian",
    "Gabrielle",
    "Isabella",
    "Millicent",
    "Valentina",
    "Alessandra",
    "Sebastian",
    "Penelope",
    "Anastasia",
    "Bernadette",
    "Genevieve",
    "Josephine",
    "Magdalena",
    "Emmanuel",
    "Cornelius",
    "Rosemarie",
    "Evangeline",
    "Wilhelmina",
    "Theophilus",
]

# "How many r's are in the name?" — the original strawberry format
NAME_TEMPLATES = [
    "How many {letter}'s are in the name '{word}'?",
    "In the name '{word}', how many times does the letter '{letter}' appear?",
]

# Hard non-English words paired with their language, chosen for dense or
# tricky repeated-letter patterns. All ASCII-representable.
NON_ENGLISH_WORDS: list[tuple[str, str]] = [
    # German
    ("Weltanschauung",       "German"),    # worldview
    ("Schadenfreude",        "German"),    # pleasure at others' misfortune
    ("Entschuldigung",       "German"),    # apology / excuse me
    ("Geschwindigkeit",      "German"),    # speed / velocity
    ("Verschlimmbessern",    "German"),    # making something worse while trying to improve it
    ("Zwangsvorstellung",    "German"),    # obsession / fixation
    # French
    ("questionnaire",        "French"),    # questionnaire
    ("bouillabaisse",        "French"),    # Provençal fish stew
    ("extraordinaire",       "French"),    # extraordinary
    # Spanish
    ("desafortunadamente",   "Spanish"),   # unfortunately
    # Italian
    ("caratteristiche",      "Italian"),   # characteristics
    ("ringraziamento",       "Italian"),   # gratitude / thanksgiving
    # Dutch
    ("verantwoordelijkheid", "Dutch"),     # responsibility
    ("aardappelsoep",        "Dutch"),     # potato soup
    # Welsh
    ("llanfairpwllgwyngyll", "Welsh"),     # famous village name
    # Finnish
    ("tietokoneohjelmointi", "Finnish"),   # computer programming
    # Portuguese
    ("desenvolvimento",      "Portuguese"), # development
]

FOREIGN_TEMPLATES = [
    "How many times does the letter '{letter}' appear in the {language} word '{word}'?",
    "In the {language} word '{word}', how many '{letter}'s are there?",
]


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def get_absent_letters(word: str, n: int = 2) -> list[str]:
    """Return up to n letters that do not appear in word."""
    present = set(word.lower().replace(" ", ""))
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    absent = [c for c in alphabet if c not in present]
    return absent[:n]


def make_examples(
    word: str,
    difficulty: str,
    templates: list[str],
    rng: random.Random,
    id_counter: list[int],
    max_present: int = 4,
    zero_prob: float = 0.45,
    extra_template_vars: dict | None = None,
    extra_fields: dict | None = None,
) -> list[dict]:
    """Build questions for one word/phrase.

    max_present        — how many distinct present-letters to sample.
    zero_prob          — probability of appending one zero-count (absent-letter)
                         question per word. At 0.45 with max_present=4 this gives
                         ~10% zero-count questions overall (0.45 / (4 + 0.45)).
    extra_template_vars — additional format variables for templates (e.g. language).
    extra_fields        — additional key/value pairs merged into each example dict.
    """
    examples = []
    word_lower = word.lower()
    present_letters = sorted(set(word_lower.replace(" ", "")))
    fmt_vars_base = {**(extra_template_vars or {})}

    sampled_letters = rng.sample(present_letters, min(max_present, len(present_letters)))

    for letter in sampled_letters:
        count = word_lower.count(letter)
        template = rng.choice(templates)
        question = template.format(letter=letter, word=word, **fmt_vars_base)

        ex = {
            "id": f"sb_{id_counter[0]:05d}",
            "word": word,
            "letter": letter,
            "question": question,
            "answer": count,
            "difficulty": difficulty,
            "word_length": len(word.replace(" ", "")),
            "zero_count": count == 0,
            "template_idx": templates.index(template),
        }
        if extra_fields:
            ex.update(extra_fields)
        examples.append(ex)
        id_counter[0] += 1

    # Stochastic zero-count question (~10% of all questions at zero_prob=0.45)
    if rng.random() < zero_prob:
        for letter in get_absent_letters(word_lower, n=1):
            template = rng.choice(templates)
            question = template.format(letter=letter, word=word, **fmt_vars_base)
            ex = {
                "id": f"sb_{id_counter[0]:05d}",
                "word": word,
                "letter": letter,
                "question": question,
                "answer": 0,
                "difficulty": difficulty,
                "word_length": len(word.replace(" ", "")),
                "zero_count": True,
                "template_idx": templates.index(template),
            }
            if extra_fields:
                ex.update(extra_fields)
            examples.append(ex)
            id_counter[0] += 1

    return examples


def generate_dataset() -> list[dict]:
    rng = random.Random(SEED)
    counter = [1]
    dataset = []

    for word in EASY_WORDS:
        dataset.extend(make_examples(word, "easy", TEMPLATES, rng, counter))

    for word in MEDIUM_WORDS:
        dataset.extend(make_examples(word, "medium", TEMPLATES, rng, counter))

    for word in HARD_WORDS:
        dataset.extend(make_examples(word, "hard", TEMPLATES, rng, counter))

    for phrase in SENTENCES:
        dataset.extend(make_examples(phrase, "sentence", SENTENCE_TEMPLATES, rng, counter))

    for passage in PARAGRAPHS:
        dataset.extend(make_examples(passage, "paragraph", PARAGRAPH_TEMPLATES, rng, counter, max_present=5))

    for name in NAMES:
        dataset.extend(make_examples(name, "names", NAME_TEMPLATES, rng, counter))

    for word, language in NON_ENGLISH_WORDS:
        dataset.extend(make_examples(
            word, "foreign", FOREIGN_TEMPLATES, rng, counter,
            extra_template_vars={"language": language},
            extra_fields={"language": language},
        ))

    rng.shuffle(dataset)
    return dataset


def print_stats(dataset: list[dict]) -> None:
    from collections import Counter
    diffs = Counter(d["difficulty"] for d in dataset)
    zeros = sum(1 for d in dataset if d["zero_count"])
    print(f"Total examples : {len(dataset)}")
    print(f"By difficulty  : {dict(diffs)}")
    print(f"Zero-count     : {zeros} ({zeros/len(dataset)*100:.1f}%)")


def main() -> None:
    dataset = generate_dataset()
    print_stats(dataset)

    # Save JSON
    json_path = OUTPUT_DIR / "dataset.json"
    json_path.write_text(json.dumps(dataset, indent=2))
    print(f"Saved {len(dataset)} examples → {json_path}")

    # Save JSONL (preferred by HuggingFace)
    jsonl_path = OUTPUT_DIR / "dataset.jsonl"
    with jsonl_path.open("w") as f:
        for ex in dataset:
            f.write(json.dumps(ex) + "\n")
    print(f"Saved JSONL → {jsonl_path}")

    # Save CSV — collect all keys across examples (some rows have extra fields)
    csv_path = OUTPUT_DIR / "dataset.csv"
    all_keys: list[str] = list(dict.fromkeys(k for d in dataset for k in d))
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, restval="")
        writer.writeheader()
        writer.writerows(dataset)
    print(f"Saved CSV → {csv_path}")


if __name__ == "__main__":
    main()
