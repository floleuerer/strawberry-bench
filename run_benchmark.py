#!/usr/bin/env python3
"""
StrawberryBench — main benchmark runner.

Usage examples:
  # Single model, zero-shot
  uv run python -m benchmark.run_benchmark --model openai/gpt-5.2

  # Single model, all strategies
  uv run python -m benchmark.run_benchmark --model openai/gpt-5.2 --strategy all

  # All registered models, zero-shot, update leaderboard
  uv run python -m benchmark.run_benchmark --all --update-leaderboard

  # Run on a subset (first 50 questions)
  uv run python -m benchmark.run_benchmark --model anthropic/claude-3.5-sonnet --limit 50
"""

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import openai
from openai import OpenAI
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from datasets import load_dataset as hf_load_dataset

from benchmark.models import MODELS, ModelConfig, get_model
from benchmark.prompts import Strategy, build_messages
from benchmark.scoring import EvalResult, NUMBER_WORDS, compute_summary, extract_answer

console = Console()

RESULTS_DIR = Path(__file__).parent / "results"
HF_DATASET_ID = "floleuerer/strawberry-bench"
LEADERBOARD_PATH = Path(__file__).parent.parent / "website" / "data" / "leaderboard.json"
EXTRACTION_MODEL = "google/gemini-3-flash-preview"


def get_client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY not set. Export it before running:\n"
            "  export OPENROUTER_API_KEY='your-key-here'"
        )
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def load_dataset(limit: int | None = None) -> list[dict]:
    console.print(f"Loading dataset from [cyan]{HF_DATASET_ID}[/cyan] ...")
    ds = hf_load_dataset(HF_DATASET_ID, split="test")
    data = list(ds)
    if limit:
        data = data[:limit]
    console.print(f"Loaded [bold]{len(data)}[/bold] examples from HuggingFace")
    return data


_MAX_RETRIES = 5
_RETRY_BASE_DELAY = 2.5  # seconds


def _chat_with_retries(client: OpenAI, **kwargs) -> str:
    """Call chat.completions.create with exponential-backoff retries on 429."""
    delay = _RETRY_BASE_DELAY
    for attempt in range(_MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(**kwargs)
            if not response.choices:
                return ""
            return response.choices[0].message.content or ""
        except openai.RateLimitError:
            if attempt == _MAX_RETRIES:
                raise
            console.print(f"[yellow]Rate limited — retrying in {delay:.0f}s (attempt {attempt + 1}/{_MAX_RETRIES})[/yellow]")
            time.sleep(delay)
            delay *= 2


_NUMBER_PATTERN = re.compile(
    r"\b(?:\d+|" + "|".join(NUMBER_WORDS) + r")\b",
    re.IGNORECASE,
)


def _needs_llm_extraction(text: str) -> bool:
    """Return True if the response has multiple numbers or no number could be extracted."""
    return len(_NUMBER_PATTERN.findall(text)) > 1 or extract_answer(text) is None


def _extract_via_llm(client: OpenAI, question: str, response: str) -> int | None:
    """Use a fast LLM to extract the final answer when multiple numbers are present."""
    prompt = (
        f"A model was asked:\n{question}\n\n"
        f"The model responded:\n{response}\n\n"
        "What integer number is the model's final answer to the question? "
        "Reply with ONLY that integer, nothing else."
    )
    try:
        raw = _chat_with_retries(
            client,
            model=EXTRACTION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=16,
            temperature=0.0,
        )
        return extract_answer(raw)
    except Exception as e:
        console.print(f"[yellow]LLM extraction fallback failed: {e}[/yellow]")
        return extract_answer(response)


def _call_one(
    client: OpenAI,
    model: ModelConfig,
    example: dict,
    strategy: Strategy,
) -> EvalResult:
    messages = build_messages(example["question"], strategy)
    try:
        raw = _chat_with_retries(
            client,
            model=model.id,
            messages=messages,
            max_tokens=model.max_tokens,
            temperature=model.temperature,
        )
    except Exception as e:
        console.print(f"[red]API error on {example['id']}: {e}[/red]")
        raw = ""
    llm_extraction = _needs_llm_extraction(raw)
    if llm_extraction:
        predicted = _extract_via_llm(client, example["question"], raw)
    else:
        predicted = extract_answer(raw)
    return EvalResult(
        example_id=example["id"],
        word=example["word"],
        letter=example["letter"],
        difficulty=example["difficulty"],
        expected=example["answer"],
        predicted=predicted,
        raw_response=raw,
        llm_extraction=llm_extraction,
    )


def run_model(
    client: OpenAI,
    model: ModelConfig,
    dataset: list[dict],
    strategy: Strategy,
    concurrency: int = 8,
) -> list[EvalResult]:
    results: list[EvalResult | None] = [None] * len(dataset)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"[{model.name}] {strategy.value}", total=len(dataset)
        )

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_idx = {
                executor.submit(_call_one, client, model, ex, strategy): i
                for i, ex in enumerate(dataset)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
                progress.advance(task)

    return results  # type: ignore[return-value]


def save_results(
    results: list[EvalResult],
    model: ModelConfig,
    strategy: Strategy,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = compute_summary(model.id, strategy.value, results)

    payload = {
        "model_id": model.id,
        "model_name": model.name,
        "provider": model.provider,
        "strategy": strategy.value,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary.to_dict(),
        "examples": [
            {
                "id": r.example_id,
                "word": r.word,
                "letter": r.letter,
                "difficulty": r.difficulty,
                "expected": r.expected,
                "extracted_answer": r.predicted,
                "correct": r.correct,
                "llm_extraction": r.llm_extraction,
                "raw_response": r.raw_response,
            }
            for r in results
        ],
    }

    slug = model.id.replace("/", "__")
    out_path = output_dir / f"{slug}_{strategy.value}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def print_summary_table(summaries: list[dict]) -> None:
    table = Table(title="StrawberryBench Results", show_header=True, header_style="bold magenta")
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Strategy")
    table.add_column("Overall", justify="right")
    table.add_column("Easy", justify="right")
    table.add_column("Medium", justify="right")
    table.add_column("Hard", justify="right")
    table.add_column("Sentence", justify="right")
    table.add_column("Paragraph", justify="right")
    table.add_column("Names", justify="right")
    table.add_column("Foreign", justify="right")

    for s in sorted(summaries, key=lambda x: -x["summary"]["accuracy"]):
        sm = s["summary"]
        diff = sm["results_by_difficulty"]

        def pct(key: str) -> str:
            if key not in diff:
                return "—"
            return f"{diff[key]['accuracy'] * 100:.1f}%"

        overall = f"{sm['accuracy'] * 100:.1f}%"
        table.add_row(
            s["model_name"],
            s["strategy"],
            overall,
            pct("easy"),
            pct("medium"),
            pct("hard"),
            pct("sentence"),
            pct("paragraph"),
            pct("names"),
            pct("foreign"),
        )

    console.print(table)


def update_leaderboard(summaries: list[dict]) -> None:
    """Merge new results into website/data/leaderboard.json."""
    if LEADERBOARD_PATH.exists():
        leaderboard = json.loads(LEADERBOARD_PATH.read_text())
    else:
        leaderboard = {"last_updated": "", "models": []}

    existing_ids = {
        (m["model_id"], m["strategy"]) for m in leaderboard.get("models", [])
    }

    for s in summaries:
        key = (s["model_id"], s["strategy"])
        if key in existing_ids:
            # Update in-place
            for i, m in enumerate(leaderboard["models"]):
                if m["model_id"] == s["model_id"] and m["strategy"] == s["strategy"]:
                    leaderboard["models"][i] = _to_leaderboard_entry(s)
                    break
        else:
            leaderboard["models"].append(_to_leaderboard_entry(s))
            existing_ids.add(key)

    leaderboard["last_updated"] = datetime.now(timezone.utc).isoformat()
    LEADERBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEADERBOARD_PATH.write_text(json.dumps(leaderboard, indent=2))
    console.print(f"[green]Leaderboard updated: {LEADERBOARD_PATH}[/green]")


def _to_leaderboard_entry(s: dict) -> dict:
    sm = s["summary"]
    diff = sm["results_by_difficulty"]
    return {
        "model_id": s["model_id"],
        "model_name": s["model_name"],
        "provider": s["provider"],
        "strategy": s["strategy"],
        "overall_accuracy": sm["accuracy"],
        "easy_accuracy": diff.get("easy", {}).get("accuracy"),
        "medium_accuracy": diff.get("medium", {}).get("accuracy"),
        "hard_accuracy": diff.get("hard", {}).get("accuracy"),
        "sentence_accuracy": diff.get("sentence", {}).get("accuracy"),
        "paragraph_accuracy": diff.get("paragraph", {}).get("accuracy"),
        "names_accuracy": diff.get("names", {}).get("accuracy"),
        "foreign_accuracy": diff.get("foreign", {}).get("accuracy"),
        "evaluated_at": s["evaluated_at"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run StrawberryBench evaluation")
    parser.add_argument("--model", help="OpenRouter model ID (e.g. openai/gpt-4o)")
    parser.add_argument("--all", action="store_true", help="Run all registered models")
    parser.add_argument(
        "--strategy",
        choices=["zero_shot", "few_shot", "chain_of_thought", "all"],
        default="zero_shot",
    )
    parser.add_argument("--limit", type=int, help="Limit to first N examples (for testing)")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--concurrency", type=int, default=8, help="Parallel API calls per model")
    parser.add_argument("--update-leaderboard", action="store_true")
    args = parser.parse_args()

    if not args.model and not args.all:
        parser.error("Specify --model <id> or --all")

    client = get_client()
    dataset = load_dataset(limit=args.limit)
    console.print(f"Loaded [bold]{len(dataset)}[/bold] examples")

    models_to_run: list[ModelConfig] = (
        MODELS if args.all else [get_model(args.model)]
    )
    strategies: list[Strategy] = (
        list(Strategy) if args.strategy == "all" else [Strategy(args.strategy)]
    )

    all_summaries = []
    for model in models_to_run:
        for strategy in strategies:
            console.rule(f"[bold]{model.name}[/bold] — {strategy.value}")
            results = run_model(client, model, dataset, strategy, concurrency=args.concurrency)
            out_path = save_results(results, model, strategy, args.output_dir)
            summary_payload = json.loads(out_path.read_text())
            all_summaries.append(summary_payload)
            console.print(
                f"Saved → [cyan]{out_path}[/cyan]  "
                f"accuracy=[bold]{summary_payload['summary']['accuracy'] * 100:.1f}%[/bold]"
            )

    print_summary_table(all_summaries)

    if args.update_leaderboard:
        update_leaderboard(all_summaries)


if __name__ == "__main__":
    main()
