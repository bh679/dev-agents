#!/usr/bin/env python3
"""
Trust score manager for dev-agents skill.

Maintains persistent trust scores per model and per task type,
logs assessments after each interaction, and provides rankings
for model selection.

Data is stored in the skill's data/ directory:
  - data/trust_scores.json  — per-model, per-task trust scores
  - data/assessments.jsonl   — append-only assessment log
  - data/comparisons/        — comparison result files

Usage:
    python3 trust_manager.py --action log --model MODEL --provider PROVIDER \
        --task TASK --rating RATING [--prompt-summary TEXT] [--eval-tokens N] \
        [--duration-ms N] [--tokens-per-second N] [--comparison-id ID] [--notes TEXT]

    python3 trust_manager.py --action query --model MODEL [--task TASK]

    python3 trust_manager.py --action rankings [--task TASK] [--json]

    python3 trust_manager.py --action history [--model MODEL] [--task TASK] [--limit N]
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SKILL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(SKILL_DIR, "data")
TRUST_FILE = os.path.join(DATA_DIR, "trust_scores.json")
ASSESSMENTS_FILE = os.path.join(DATA_DIR, "assessments.jsonl")
COMPARISONS_DIR = os.path.join(DATA_DIR, "comparisons")

SCHEMA_VERSION = 1
EWMA_ALPHA = 0.3
EWMA_MIN_COUNT = 3  # Use simple average until this many ratings
DECAY_HALF_LIFE_DAYS = 30

VALID_TASKS = ["code", "review", "debug", "test", "docs", "architecture", "refactor", "general"]


# ---------------------------------------------------------------------------
# Data directory setup
# ---------------------------------------------------------------------------

def ensure_data_dirs():
    """Create data directories if they don't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(COMPARISONS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Trust score I/O
# ---------------------------------------------------------------------------

def _empty_trust_data():
    """Return an empty trust scores structure."""
    return {"schema_version": SCHEMA_VERSION, "models": {}}


def load_trust_scores():
    """Load trust scores from disk. Returns empty structure if file missing."""
    if not os.path.exists(TRUST_FILE):
        return _empty_trust_data()
    try:
        with open(TRUST_FILE, "r") as f:
            data = json.load(f)
        if data.get("schema_version") != SCHEMA_VERSION:
            print(f"Warning: trust_scores.json has schema version {data.get('schema_version')}, "
                  f"expected {SCHEMA_VERSION}. Using as-is.", file=sys.stderr)
        return data
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: Could not load trust scores: {e}. Starting fresh.", file=sys.stderr)
        return _empty_trust_data()


def save_trust_scores(data):
    """Atomically write trust scores to disk."""
    ensure_data_dirs()
    tmp_path = TRUST_FILE + ".tmp"
    try:
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, TRUST_FILE)
    except OSError as e:
        print(f"ERROR: Could not save trust scores: {e}", file=sys.stderr)
        # Clean up temp file
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Trust score updates
# ---------------------------------------------------------------------------

def _iso_now():
    """Return current time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _normalize_rating(rating):
    """Normalize a 1-5 rating to 0-1 scale."""
    return max(0.0, min(1.0, (rating - 1) / 4.0))


def _compute_global_score(task_scores):
    """Compute weighted average of task scores, weighted by count."""
    total_weight = 0
    weighted_sum = 0.0
    for ts in task_scores.values():
        count = ts.get("count", 0)
        if count > 0:
            weighted_sum += ts["score"] * count
            total_weight += count
    if total_weight == 0:
        return 0.0
    return round(weighted_sum / total_weight, 4)


def update_trust_score(model, provider, task_type, rating):
    """
    Update trust score for a model on a specific task type.

    For the first EWMA_MIN_COUNT ratings, uses simple averaging.
    After that, uses exponentially weighted moving average (alpha=0.3).

    Args:
        model: Model name (e.g. 'qwen2.5-coder:32b')
        provider: Provider name ('ollama' or 'gemini')
        task_type: Task type (e.g. 'code', 'review')
        rating: Integer 1-5
    """
    data = load_trust_scores()
    now = _iso_now()
    normalized = _normalize_rating(rating)

    models = data["models"]
    if model not in models:
        models[model] = {
            "provider": provider,
            "global_score": 0.0,
            "task_scores": {},
            "total_runs": 0,
            "first_seen": now,
            "last_used": now,
        }

    entry = models[model]
    entry["last_used"] = now
    entry["total_runs"] += 1

    task_scores = entry["task_scores"]
    if task_type not in task_scores:
        task_scores[task_type] = {
            "score": normalized,
            "count": 1,
            "last_updated": now,
            "_ratings_sum": normalized,  # Track sum for simple averaging
        }
    else:
        ts = task_scores[task_type]
        ts["count"] += 1
        ts["last_updated"] = now

        if ts["count"] <= EWMA_MIN_COUNT:
            # Simple average for first few ratings
            ts["_ratings_sum"] = ts.get("_ratings_sum", ts["score"]) + normalized
            ts["score"] = round(ts["_ratings_sum"] / ts["count"], 4)
        else:
            # EWMA: new_score = alpha * new_value + (1-alpha) * old_score
            ts["score"] = round(
                EWMA_ALPHA * normalized + (1 - EWMA_ALPHA) * ts["score"], 4
            )

    entry["global_score"] = _compute_global_score(task_scores)
    save_trust_scores(data)
    return entry


def apply_time_decay(data):
    """
    Apply time-based decay to scores that haven't been updated recently.

    Scores decay by half every DECAY_HALF_LIFE_DAYS days since last update.
    Only applies to scores older than the half-life period.
    """
    import math
    now = datetime.now(timezone.utc)
    changed = False

    for model_name, model_data in data.get("models", {}).items():
        for task_type, ts in model_data.get("task_scores", {}).items():
            last_updated = ts.get("last_updated")
            if not last_updated:
                continue
            try:
                last_dt = datetime.fromisoformat(last_updated)
                days_since = (now - last_dt).total_seconds() / 86400
                if days_since > DECAY_HALF_LIFE_DAYS:
                    decay_factor = math.pow(2, -days_since / DECAY_HALF_LIFE_DAYS)
                    new_score = round(ts["score"] * decay_factor, 4)
                    if abs(new_score - ts["score"]) > 0.001:
                        ts["score"] = new_score
                        changed = True
            except (ValueError, TypeError):
                continue

        if changed:
            model_data["global_score"] = _compute_global_score(model_data.get("task_scores", {}))

    return changed


# ---------------------------------------------------------------------------
# Assessment logging
# ---------------------------------------------------------------------------

def log_assessment(entry):
    """
    Append an assessment entry to the JSONL log file.

    Args:
        entry: Dict with keys: model, provider, task_type, rating,
               prompt_summary, response_length, eval_tokens, duration_ms,
               tokens_per_second, comparison_id, notes
    """
    ensure_data_dirs()
    entry.setdefault("timestamp", _iso_now())
    try:
        with open(ASSESSMENTS_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError as e:
        print(f"ERROR: Could not write assessment log: {e}", file=sys.stderr)


def read_assessments(model=None, task_type=None, limit=None):
    """
    Read assessment entries from the JSONL log, optionally filtered.

    Returns entries in reverse chronological order (most recent first).
    """
    if not os.path.exists(ASSESSMENTS_FILE):
        return []

    entries = []
    try:
        with open(ASSESSMENTS_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if model and entry.get("model") != model:
                        continue
                    if task_type and entry.get("task_type") != task_type:
                        continue
                    entries.append(entry)
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []

    entries.reverse()
    if limit:
        entries = entries[:limit]
    return entries


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def get_trust_score(model, task_type=None):
    """
    Get trust score for a model, optionally for a specific task type.

    Returns:
        float (0-1) or None if no data exists.
    """
    data = load_trust_scores()
    model_data = data.get("models", {}).get(model)
    if not model_data:
        return None

    if task_type:
        ts = model_data.get("task_scores", {}).get(task_type)
        if ts:
            return ts["score"]
        return None

    return model_data.get("global_score")


def get_task_count(model, task_type):
    """Get the number of assessments for a model on a specific task type."""
    data = load_trust_scores()
    model_data = data.get("models", {}).get(model)
    if not model_data:
        return 0
    ts = model_data.get("task_scores", {}).get(task_type)
    return ts.get("count", 0) if ts else 0


def get_model_rankings(task_type=None, include_details=False):
    """
    Get all models sorted by trust score for a given task type.

    Args:
        task_type: Filter to a specific task type. If None, uses global score.
        include_details: Include full model data in results.

    Returns:
        List of dicts with 'model', 'provider', 'score', 'count', and
        optionally 'details'.
    """
    data = load_trust_scores()

    # Apply time decay
    if apply_time_decay(data):
        save_trust_scores(data)

    rankings = []
    for model_name, model_data in data.get("models", {}).items():
        if task_type:
            ts = model_data.get("task_scores", {}).get(task_type)
            if ts:
                entry = {
                    "model": model_name,
                    "provider": model_data.get("provider", "unknown"),
                    "score": ts["score"],
                    "count": ts["count"],
                    "last_updated": ts.get("last_updated", ""),
                }
                if include_details:
                    entry["details"] = model_data
                rankings.append(entry)
        else:
            entry = {
                "model": model_name,
                "provider": model_data.get("provider", "unknown"),
                "score": model_data.get("global_score", 0),
                "count": model_data.get("total_runs", 0),
                "last_used": model_data.get("last_used", ""),
            }
            if include_details:
                entry["details"] = model_data
            rankings.append(entry)

    rankings.sort(key=lambda x: (-x["score"], -x["count"]))
    return rankings


# ---------------------------------------------------------------------------
# Comparison file I/O
# ---------------------------------------------------------------------------

def save_comparison(comparison_id, comparison_data):
    """Save a comparison result to data/comparisons/."""
    ensure_data_dirs()
    filepath = os.path.join(COMPARISONS_DIR, f"{comparison_id}.json")
    try:
        with open(filepath, "w") as f:
            json.dump(comparison_data, f, indent=2)
        return filepath
    except OSError as e:
        print(f"ERROR: Could not save comparison: {e}", file=sys.stderr)
        return None


def load_comparison(comparison_id):
    """Load a comparison result by ID."""
    filepath = os.path.join(COMPARISONS_DIR, f"{comparison_id}.json")
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


# ---------------------------------------------------------------------------
# Auto-assessment heuristics
# ---------------------------------------------------------------------------

def auto_assess(response, task_type, prompt):
    """
    Provide a basic heuristic assessment of a model response.

    This is a rough estimate. Claude should override with --rating when possible.

    Returns:
        int: Rating 1-5
    """
    if not response or len(response.strip()) < 20:
        return 1  # Empty or near-empty response

    response_len = len(response.strip())

    # Task-specific heuristics
    if task_type in ("code", "test", "refactor", "debug"):
        has_code_block = "```" in response or "def " in response or "function " in response
        has_class = "class " in response
        if not has_code_block and not has_class and response_len < 100:
            return 2  # Code task but no code in response
        if has_code_block and response_len > 100:
            return 4  # Has code, reasonable length
        return 3  # Some content but unclear quality

    if task_type == "review":
        has_structure = any(marker in response for marker in ["- ", "* ", "1.", "##", "**"])
        if has_structure and response_len > 200:
            return 4
        if response_len < 100:
            return 2
        return 3

    if task_type in ("docs", "architecture"):
        if response_len > 500:
            return 4  # Substantial documentation
        if response_len > 200:
            return 3
        return 2

    # General: check for reasonable length
    if response_len > 100:
        return 3
    return 2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_rankings_human(rankings, task_type):
    """Print rankings in human-readable format."""
    label = f"for '{task_type}'" if task_type else "(global)"
    if not rankings:
        print(f"No trust data {label}.")
        return

    print(f"Model rankings {label}:\n")
    for i, r in enumerate(rankings, 1):
        score_pct = r["score"] * 100
        bar = "#" * int(score_pct / 5) + "-" * (20 - int(score_pct / 5))
        print(f"  {i}. {r['model']} [{r['provider']}]")
        print(f"     Score: {score_pct:.1f}% [{bar}]  ({r['count']} runs)")
        print()


def _print_query_human(model, task_type, data):
    """Print query result in human-readable format."""
    model_data = data.get("models", {}).get(model)
    if not model_data:
        print(f"No trust data for model '{model}'.")
        return

    print(f"Trust data for: {model} [{model_data.get('provider', '?')}]")
    print(f"  Global score: {model_data['global_score'] * 100:.1f}%")
    print(f"  Total runs:   {model_data['total_runs']}")
    print(f"  First seen:   {model_data.get('first_seen', '?')}")
    print(f"  Last used:    {model_data.get('last_used', '?')}")

    if task_type:
        ts = model_data.get("task_scores", {}).get(task_type)
        if ts:
            print(f"\n  Task '{task_type}':")
            print(f"    Score: {ts['score'] * 100:.1f}%  ({ts['count']} runs)")
            print(f"    Last:  {ts.get('last_updated', '?')}")
        else:
            print(f"\n  No data for task '{task_type}'.")
    else:
        task_scores = model_data.get("task_scores", {})
        if task_scores:
            print(f"\n  Per-task scores:")
            for t, ts in sorted(task_scores.items(), key=lambda x: -x[1]["score"]):
                print(f"    {t:15s} {ts['score'] * 100:.1f}%  ({ts['count']} runs)")


def _print_history_human(entries):
    """Print assessment history in human-readable format."""
    if not entries:
        print("No assessment history found.")
        return

    print(f"Assessment history ({len(entries)} entries):\n")
    for e in entries:
        ts = e.get("timestamp", "?")[:19]
        model = e.get("model", "?")
        task = e.get("task_type", "?")
        rating = e.get("rating", "?")
        summary = e.get("prompt_summary", "")[:60]
        cmp = f" [cmp:{e['comparison_id'][:8]}]" if e.get("comparison_id") else ""
        print(f"  {ts}  {model}  {task}  rating={rating}{cmp}")
        if summary:
            print(f"    {summary}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Trust score manager for dev-agents")
    parser.add_argument("--action", required=True,
                        choices=["log", "query", "rankings", "history"],
                        help="Action to perform")
    parser.add_argument("--model", help="Model name")
    parser.add_argument("--provider", help="Provider (ollama or gemini)")
    parser.add_argument("--task", help="Task type")
    parser.add_argument("--rating", type=int, choices=[1, 2, 3, 4, 5],
                        help="Assessment rating (1-5)")
    parser.add_argument("--prompt-summary", default="", help="Brief prompt summary")
    parser.add_argument("--eval-tokens", type=int, default=0, help="Tokens generated")
    parser.add_argument("--duration-ms", type=float, default=0, help="Duration in ms")
    parser.add_argument("--tokens-per-second", type=float, default=0, help="Generation speed")
    parser.add_argument("--response-length", type=int, default=0, help="Response character length")
    parser.add_argument("--comparison-id", default=None, help="Comparison run ID")
    parser.add_argument("--notes", default="", help="Additional notes")
    parser.add_argument("--limit", type=int, default=20, help="Max entries for history")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    ensure_data_dirs()

    if args.action == "log":
        if not args.model or not args.provider or not args.task or not args.rating:
            parser.error("--action log requires --model, --provider, --task, and --rating")

        # Log the assessment
        entry = {
            "timestamp": _iso_now(),
            "model": args.model,
            "provider": args.provider,
            "task_type": args.task,
            "rating": args.rating,
            "prompt_summary": args.prompt_summary,
            "response_length": args.response_length,
            "eval_tokens": args.eval_tokens,
            "duration_ms": args.duration_ms,
            "tokens_per_second": args.tokens_per_second,
            "comparison_id": args.comparison_id,
            "notes": args.notes,
        }
        log_assessment(entry)

        # Update trust score
        result = update_trust_score(args.model, args.provider, args.task, args.rating)

        if args.json:
            print(json.dumps({"logged": True, "model_data": result}, indent=2))
        else:
            ts = result["task_scores"].get(args.task, {})
            print(f"Logged: {args.model} [{args.provider}] {args.task} rating={args.rating}")
            print(f"  Task score: {ts.get('score', 0) * 100:.1f}% ({ts.get('count', 0)} runs)")
            print(f"  Global:     {result['global_score'] * 100:.1f}%")

    elif args.action == "query":
        if not args.model:
            parser.error("--action query requires --model")

        data = load_trust_scores()
        if args.json:
            model_data = data.get("models", {}).get(args.model)
            if model_data:
                print(json.dumps({"model": args.model, "data": model_data}, indent=2))
            else:
                print(json.dumps({"model": args.model, "data": None}, indent=2))
        else:
            _print_query_human(args.model, args.task, data)

    elif args.action == "rankings":
        rankings = get_model_rankings(task_type=args.task)

        if args.json:
            print(json.dumps({"task": args.task, "rankings": rankings}, indent=2))
        else:
            _print_rankings_human(rankings, args.task)

    elif args.action == "history":
        entries = read_assessments(
            model=args.model,
            task_type=args.task,
            limit=args.limit,
        )

        if args.json:
            print(json.dumps({"entries": entries}, indent=2))
        else:
            _print_history_human(entries)


if __name__ == "__main__":
    main()
