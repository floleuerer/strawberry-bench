import json
from pathlib import Path
from datetime import datetime, timezone

RESULTS_DIR = Path("benchmark/results")
LEADERBOARD_PATH = Path("website/data/leaderboard.json")

def _to_leaderboard_entry(s: dict) -> dict:
    sm = s["summary"]
    diff = sm["results_by_difficulty"]
    
    # Check for classic strawberry question (sb_00184)
    classic_strawberry = False
    for ex in s.get("examples", []):
        if ex["id"] == "sb_00184":
            classic_strawberry = ex["correct"]
            break

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
        "classic_strawberry": classic_strawberry,
        "evaluated_at": s["evaluated_at"],
    }

def main():
    models = []
    all_qids = set()
    for fpath in RESULTS_DIR.glob("*.json"):
        with open(fpath, "r") as f:
            data = json.load(f)
            models.append(_to_leaderboard_entry(data))
            for ex in data.get("examples", []):
                all_qids.add(ex["id"])
    
    # Sort by accuracy descending
    models.sort(key=lambda x: -x["overall_accuracy"])
    
    leaderboard = {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "total_questions": len(all_qids),
        "models": models
    }
    
    LEADERBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LEADERBOARD_PATH, "w") as f:
        json.dump(leaderboard, f, indent=2)
    print(f"Leaderboard rebuilt with {len(models)} models: {LEADERBOARD_PATH}")

if __name__ == "__main__":
    main()
