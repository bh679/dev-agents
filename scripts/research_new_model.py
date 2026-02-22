#!/usr/bin/env python3
"""
Auto-research and seed trust scores for newly discovered models.

When discover_models.py finds a model that has no entry in trust_scores.json,
this script estimates initial trust scores using:

1. **Known model lookup**: Check seed_trust.py's SEED_DATA for exact or variant matches
2. **Family calibration**: Find similar models from the same family and interpolate by size
3. **Heuristic estimation**: Use model type, parameter count, and family to estimate scores
4. **Provider-specific defaults**: Apply Gemini-specific or Ollama-specific baselines

The resulting scores are low-confidence (count=1) so actual assessments quickly override them.

Usage:
    # Seed a single new model (auto-detected from Ollama or Gemini)
    python3 scripts/research_new_model.py --model "qwen4-coder:30b" --provider ollama

    # Seed a model with explicit classification info
    python3 scripts/research_new_model.py --model "new-model:14b" --provider ollama \
        --type code --size 14

    # Seed all new models found by discover_models.py
    python3 scripts/research_new_model.py --auto-discover

    # Dry-run: show what would be seeded without writing
    python3 scripts/research_new_model.py --auto-discover --dry-run

    # Force re-seed (overwrite existing scores)
    python3 scripts/research_new_model.py --model "model:7b" --provider ollama --force

    # Output as JSON for Claude to review
    python3 scripts/research_new_model.py --auto-discover --json
"""

import argparse
import json
import os
import re
import sys

# Add scripts dir to path
scripts_dir = os.path.dirname(os.path.abspath(__file__))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

import trust_manager

# Try to import seed data for known model lookup
try:
    from seed_trust import SEED_DATA
except ImportError:
    SEED_DATA = {}


# ---------------------------------------------------------------------------
# Task types and their weights for different model categories
# ---------------------------------------------------------------------------

TASK_TYPES = ["code", "review", "debug", "test", "docs", "architecture", "refactor", "general"]

# Base score profiles by model type (0-1 scale)
# These represent the "shape" of capability — actual magnitudes are scaled by size
TYPE_PROFILES = {
    "code": {
        "code": 1.0, "review": 0.75, "debug": 0.88, "test": 0.85,
        "docs": 0.55, "architecture": 0.50, "refactor": 0.88, "general": 0.65,
    },
    "reasoning": {
        "code": 0.75, "review": 0.85, "debug": 0.80, "test": 0.70,
        "docs": 0.90, "architecture": 0.95, "refactor": 0.78, "general": 0.88,
    },
    "general": {
        "code": 0.80, "review": 0.75, "debug": 0.72, "test": 0.68,
        "docs": 0.78, "architecture": 0.72, "refactor": 0.72, "general": 0.80,
    },
}

# Size-based scaling factors (multiplied against type profiles)
# Represents how much of the model type's potential is realized at each size
SIZE_SCALING = [
    # (min_params_B, max_params_B, scale_factor)
    (0, 1, 0.25),
    (1, 3, 0.35),
    (3, 7, 0.50),
    (7, 14, 0.65),
    (14, 25, 0.78),
    (25, 40, 0.85),
    (40, 70, 0.88),
    (70, 120, 0.92),
    (120, 250, 0.95),
    (250, float("inf"), 0.98),
]

# Family-specific adjustments (additive, applied after size scaling)
# Positive = family is stronger than average for this task
FAMILY_ADJUSTMENTS = {
    "qwen": {"code": 0.03, "math": 0.02},
    "qwen2": {"code": 0.04, "math": 0.03, "general": 0.02},
    "qwen4": {"code": 0.05, "math": 0.03, "general": 0.03},
    "deepseek": {"code": 0.05, "debug": 0.03, "math": 0.03},
    "llama": {"general": 0.02, "docs": 0.02, "architecture": 0.02},
    "phi": {"code": 0.05, "math": 0.04},  # Phi punches above weight
    "gemma": {"general": 0.02, "docs": 0.02},
    "mistral": {"general": 0.03, "code": 0.02},  # Efficient
    "mixtral": {"general": 0.03, "code": 0.03},
    "starcoder": {"code": 0.02},  # Code-only, weaker on other tasks
    "codellama": {"code": 0.02, "debug": 0.01},
    "command-r": {"docs": 0.03, "architecture": 0.02, "general": 0.03},
}

# Gemini quality tier baseline scores
GEMINI_TIER_BASELINES = {
    "high": 0.85,      # Pro models
    "good": 0.75,      # Flash models
    "moderate": 0.52,   # Lite models
    "basic": 0.35,      # Nano models
}


# ---------------------------------------------------------------------------
# Model name parsing
# ---------------------------------------------------------------------------

def parse_model_name(name):
    """
    Parse a model name to extract family, variant, and size.

    Examples:
        "qwen2.5-coder:32b" -> family="qwen2", variant="coder", size=32
        "llama3.1:70b"       -> family="llama", variant=None, size=70
        "gemini-2.5-flash"   -> family="gemini", variant="flash", size=None
    """
    name_lower = name.lower().strip()

    # Extract size from name
    size = None
    size_match = re.search(r":?(\d+\.?\d*)\s*[bB]", name_lower)
    if size_match:
        size = float(size_match.group(1))

    # Detect code variant
    is_code = bool(re.search(r"coder|codellama|starcoder|codestral|codeqwen|codegeex", name_lower))

    # Detect family
    family = None
    family_patterns = [
        (r"qwen4", "qwen4"),
        (r"qwen2\.?5?", "qwen2"),
        (r"qwen", "qwen"),
        (r"llama3\.?[0-9]*", "llama"),
        (r"codellama", "codellama"),
        (r"deepseek", "deepseek"),
        (r"phi-?[0-9]*", "phi"),
        (r"gemma[0-9]*", "gemma"),
        (r"mistral", "mistral"),
        (r"mixtral", "mixtral"),
        (r"starcoder[0-9]*", "starcoder"),
        (r"command-r", "command-r"),
        (r"g[po]t-oss", "gpt-oss"),
        (r"gemini", "gemini"),
        (r"yi", "yi"),
        (r"internlm", "internlm"),
        (r"dbrx", "dbrx"),
    ]
    for pattern, fam in family_patterns:
        if re.search(pattern, name_lower):
            family = fam
            break

    # Detect model type
    model_type = "general"
    if is_code:
        model_type = "code"
    elif family in ("gpt-oss", "command-r"):
        model_type = "reasoning"
    elif size and size >= 65 and not is_code:
        model_type = "reasoning"

    # Detect variant
    variant = None
    if is_code:
        variant = "code"
    elif "lite" in name_lower:
        variant = "lite"
    elif "plus" in name_lower:
        variant = "plus"

    return {
        "name": name,
        "family": family,
        "variant": variant,
        "model_type": model_type,
        "size_b": size,
        "is_code": is_code,
    }


def get_size_scale(size_b):
    """Get the scaling factor for a given parameter size in billions."""
    if size_b is None:
        return 0.65  # Default to moderate
    for min_b, max_b, scale in SIZE_SCALING:
        if min_b <= size_b < max_b:
            return scale
    return 0.65


# ---------------------------------------------------------------------------
# Score estimation
# ---------------------------------------------------------------------------

def find_similar_seed_models(parsed, provider):
    """
    Find similar models in SEED_DATA to calibrate scores.

    Returns list of (model_name, seed_info, similarity_score) tuples,
    sorted by similarity (highest first).
    """
    if not SEED_DATA:
        return []

    matches = []
    for seed_name, seed_info in SEED_DATA.items():
        if seed_info["provider"] != provider:
            continue

        seed_parsed = parse_model_name(seed_name)
        similarity = 0

        # Same family is strong signal
        if parsed["family"] and seed_parsed["family"] == parsed["family"]:
            similarity += 5

        # Same model type
        if parsed["model_type"] == seed_parsed["model_type"]:
            similarity += 3

        # Same variant (e.g., both are "coder" variants)
        if parsed["variant"] and parsed["variant"] == seed_parsed["variant"]:
            similarity += 2

        # Close in size
        if parsed["size_b"] and seed_parsed["size_b"]:
            size_ratio = min(parsed["size_b"], seed_parsed["size_b"]) / max(parsed["size_b"], seed_parsed["size_b"])
            if size_ratio > 0.7:
                similarity += 2
            elif size_ratio > 0.4:
                similarity += 1

        if similarity >= 3:  # Minimum threshold
            matches.append((seed_name, seed_info, similarity))

    matches.sort(key=lambda x: -x[2])
    return matches[:5]


def estimate_from_similar(similar_models, parsed):
    """
    Estimate scores by interpolating from similar models in seed data.

    Uses weighted average based on similarity scores, with size adjustment.
    """
    if not similar_models:
        return None

    total_weight = 0
    weighted_scores = {t: 0.0 for t in TASK_TYPES}

    for seed_name, seed_info, similarity in similar_models:
        seed_parsed = parse_model_name(seed_name)
        scores = seed_info["scores"]

        # Size-based adjustment
        size_adjustment = 1.0
        if parsed["size_b"] and seed_parsed["size_b"]:
            if parsed["size_b"] > seed_parsed["size_b"]:
                # Larger model: slight boost
                ratio = parsed["size_b"] / seed_parsed["size_b"]
                size_adjustment = min(1.15, 1.0 + (ratio - 1) * 0.1)
            else:
                # Smaller model: slight reduction
                ratio = parsed["size_b"] / seed_parsed["size_b"]
                size_adjustment = max(0.75, ratio)

        weight = similarity
        total_weight += weight

        for task in TASK_TYPES:
            if task in scores:
                adjusted = min(0.98, scores[task] * size_adjustment)
                weighted_scores[task] += adjusted * weight

    if total_weight == 0:
        return None

    result = {}
    for task in TASK_TYPES:
        result[task] = round(weighted_scores[task] / total_weight, 2)

    return result


def estimate_from_heuristics(parsed, provider):
    """
    Estimate scores purely from model metadata heuristics.

    Uses model type profile, size scaling, and family adjustments.
    """
    model_type = parsed["model_type"]
    profile = TYPE_PROFILES.get(model_type, TYPE_PROFILES["general"])

    if provider == "gemini":
        # Gemini: use quality tier baselines
        name_lower = parsed["name"].lower()
        if "pro" in name_lower:
            baseline = GEMINI_TIER_BASELINES["high"]
        elif "lite" in name_lower:
            baseline = GEMINI_TIER_BASELINES["moderate"]
        elif "nano" in name_lower:
            baseline = GEMINI_TIER_BASELINES["basic"]
        else:
            baseline = GEMINI_TIER_BASELINES["good"]

        # Apply profile shape
        scores = {}
        for task in TASK_TYPES:
            task_weight = profile.get(task, 0.7)
            scores[task] = round(min(0.98, baseline * task_weight), 2)

        # Preview models get a small boost (they're usually improvements)
        if "preview" in name_lower or "exp" in name_lower:
            for task in scores:
                scores[task] = round(min(0.98, scores[task] + 0.03), 2)

        return scores

    # Ollama: use size scaling + family adjustments
    size_scale = get_size_scale(parsed["size_b"])

    scores = {}
    for task in TASK_TYPES:
        task_weight = profile.get(task, 0.7)
        base_score = task_weight * size_scale

        # Apply family adjustments
        family = parsed["family"]
        if family and family in FAMILY_ADJUSTMENTS:
            adj = FAMILY_ADJUSTMENTS[family].get(task, 0)
            base_score += adj

        scores[task] = round(max(0.1, min(0.98, base_score)), 2)

    return scores


def estimate_scores(model_name, provider, model_type=None, size_b=None):
    """
    Main entry point: estimate trust scores for a new model.

    Strategy:
    1. Parse model name to extract metadata
    2. Look for exact match in seed data
    3. Try interpolation from similar seed models
    4. Fall back to heuristic estimation

    Returns dict with scores and metadata about the estimation method.
    """
    parsed = parse_model_name(model_name)

    # Allow overrides
    if model_type:
        parsed["model_type"] = model_type
    if size_b is not None:
        parsed["size_b"] = size_b

    # Check for exact match in seed data
    if model_name in SEED_DATA and SEED_DATA[model_name]["provider"] == provider:
        return {
            "scores": SEED_DATA[model_name]["scores"],
            "method": "exact-seed-match",
            "notes": SEED_DATA[model_name].get("notes", ""),
            "confidence": "high",
            "parsed": parsed,
            "similar_models": [],
        }

    # Try to find similar models and interpolate
    similar = find_similar_seed_models(parsed, provider)
    interpolated = estimate_from_similar(similar, parsed)

    if interpolated and similar and similar[0][2] >= 5:
        # Strong match from similar models
        return {
            "scores": interpolated,
            "method": "interpolated-from-similar",
            "notes": f"Based on {len(similar)} similar models: {', '.join(s[0] for s in similar[:3])}",
            "confidence": "medium",
            "parsed": parsed,
            "similar_models": [(s[0], s[2]) for s in similar],
        }

    # Fall back to heuristic estimation
    heuristic = estimate_from_heuristics(parsed, provider)

    # If we have both interpolated and heuristic, blend them
    if interpolated:
        blended = {}
        for task in TASK_TYPES:
            interp_val = interpolated.get(task, 0.5)
            heur_val = heuristic.get(task, 0.5)
            # Weight interpolation more if similarity is high
            best_sim = similar[0][2] if similar else 0
            interp_weight = min(0.7, best_sim / 10)
            blended[task] = round(interp_val * interp_weight + heur_val * (1 - interp_weight), 2)
        return {
            "scores": blended,
            "method": "blended-interpolation-heuristic",
            "notes": f"Blended with similar: {', '.join(s[0] for s in similar[:3])}",
            "confidence": "medium-low",
            "parsed": parsed,
            "similar_models": [(s[0], s[2]) for s in similar],
        }

    return {
        "scores": heuristic,
        "method": "heuristic-estimation",
        "notes": f"Estimated from {parsed['model_type']} type, {parsed['size_b'] or 'unknown'}B params, {parsed['family'] or 'unknown'} family",
        "confidence": "low",
        "parsed": parsed,
        "similar_models": [],
    }


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def seed_model(model_name, provider, scores, notes="", force=False):
    """
    Seed trust scores for a single model.

    Args:
        model_name: Model name
        provider: 'ollama' or 'gemini'
        scores: Dict of task -> score (0-1)
        notes: Description of how scores were derived
        force: Overwrite existing data

    Returns True if seeded, False if skipped.
    """
    trust_manager.ensure_data_dirs()
    data = trust_manager.load_trust_scores()
    now = trust_manager._iso_now()

    # Check if model already has trust data
    if model_name in data.get("models", {}) and not force:
        existing = data["models"][model_name]
        if existing.get("total_runs", 0) > 0:
            return False

    # Build task scores
    task_scores = {}
    for task_type, score in scores.items():
        task_scores[task_type] = {
            "score": score,
            "count": 1,
            "last_updated": now,
            "_ratings_sum": score,
            "_source": "auto-research",
        }

    # Compute global score
    global_score = round(sum(scores.values()) / len(scores), 4) if scores else 0

    data.setdefault("models", {})[model_name] = {
        "provider": provider,
        "global_score": global_score,
        "task_scores": task_scores,
        "total_runs": 1,
        "first_seen": now,
        "last_used": now,
        "_seed_notes": notes,
        "_seeded_from": "auto-research",
    }

    trust_manager.save_trust_scores(data)

    # Log assessments for audit trail
    for task_type, score in scores.items():
        rating = max(1, min(5, round(score * 4 + 1)))
        trust_manager.log_assessment({
            "model": model_name,
            "provider": provider,
            "task_type": task_type,
            "rating": rating,
            "prompt_summary": f"[AUTO-RESEARCH] {notes[:80]}",
            "response_length": 0,
            "eval_tokens": 0,
            "duration_ms": 0,
            "tokens_per_second": 0,
            "comparison_id": None,
            "notes": f"auto-research: {notes}",
        })

    return True


# ---------------------------------------------------------------------------
# Auto-discovery integration
# ---------------------------------------------------------------------------

def auto_discover_and_seed(provider_filter="all", force=False, dry_run=False):
    """
    Discover all available models and seed trust scores for any new ones.

    Returns list of (model_name, provider, estimation_result, was_seeded) tuples.
    """
    # Import discover_models
    try:
        import discover_models
    except ImportError:
        print("ERROR: Cannot import discover_models.py", file=sys.stderr)
        return []

    # Load current trust data
    data = trust_manager.load_trust_scores()
    known_models = set(data.get("models", {}).keys())

    # Discover available models
    classifications = []
    if provider_filter in ("all", "ollama"):
        models, err = discover_models.fetch_ollama_models(discover_models.OLLAMA_DEFAULT_URL)
        if err:
            print(f"Warning: Ollama: {err}", file=sys.stderr)
        else:
            classifications.extend(discover_models.classify_model(m) for m in models)

    if provider_filter in ("all", "gemini"):
        gemini_models, err = discover_models.check_gemini_availability()
        if err:
            print(f"Warning: Gemini: {err}", file=sys.stderr)
        else:
            classifications.extend(
                discover_models.classify_gemini_model_dynamic(m) for m in gemini_models
            )

    results = []
    for c in classifications:
        model_name = c["name"]
        provider = c.get("provider", "ollama")

        # Skip if already known
        if model_name in known_models and not force:
            continue

        # Estimate scores
        model_type = c.get("primary_type")
        size_b = c.get("parameter_size_b")
        estimation = estimate_scores(model_name, provider, model_type=model_type, size_b=size_b)

        was_seeded = False
        if not dry_run:
            was_seeded = seed_model(
                model_name, provider,
                estimation["scores"],
                notes=estimation["notes"],
                force=force,
            )
            if was_seeded:
                # Add to known set so we don't double-seed
                known_models.add(model_name)

        results.append((model_name, provider, estimation, was_seeded))

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Auto-research and seed trust scores for new models"
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--model", help="Seed a specific model")
    mode.add_argument("--auto-discover", action="store_true",
                      help="Discover and seed all new models")

    parser.add_argument("--provider", default="ollama", choices=["ollama", "gemini", "all"],
                        help="Provider (default: ollama, use 'all' with --auto-discover)")
    parser.add_argument("--type", choices=["code", "reasoning", "general"],
                        help="Override model type classification")
    parser.add_argument("--size", type=float,
                        help="Override parameter size in billions")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing trust data")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be seeded without writing")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")

    args = parser.parse_args()

    trust_manager.ensure_data_dirs()

    if args.auto_discover:
        results = auto_discover_and_seed(
            provider_filter=args.provider,
            force=args.force,
            dry_run=args.dry_run,
        )

        if args.json:
            output = []
            for model_name, provider, estimation, was_seeded in results:
                output.append({
                    "model": model_name,
                    "provider": provider,
                    "scores": estimation["scores"],
                    "method": estimation["method"],
                    "confidence": estimation["confidence"],
                    "notes": estimation["notes"],
                    "seeded": was_seeded,
                    "similar_models": estimation.get("similar_models", []),
                })
            print(json.dumps(output, indent=2))
        else:
            if not results:
                print("No new models found. All available models already have trust data.")
                return

            seeded_count = sum(1 for _, _, _, s in results if s)
            print(f"Found {len(results)} new model(s):\n")

            for model_name, provider, estimation, was_seeded in results:
                status = "SEEDED" if was_seeded else ("DRY-RUN" if args.dry_run else "SKIPPED")
                avg = sum(estimation["scores"].values()) / len(estimation["scores"])
                print(f"  [{status}] {model_name} [{provider}]")
                print(f"    Method: {estimation['method']} (confidence: {estimation['confidence']})")
                print(f"    Average: {avg * 100:.1f}%")

                # Show per-task scores
                scores = estimation["scores"]
                best_task = max(scores, key=scores.get)
                worst_task = min(scores, key=scores.get)
                print(f"    Best: {best_task} ({scores[best_task] * 100:.0f}%) | "
                      f"Worst: {worst_task} ({scores[worst_task] * 100:.0f}%)")

                if estimation.get("similar_models"):
                    similar_str = ", ".join(f"{n}({s})" for n, s in estimation["similar_models"][:3])
                    print(f"    Similar: {similar_str}")
                if estimation.get("notes"):
                    print(f"    Notes: {estimation['notes']}")
                print()

            if not args.dry_run:
                print(f"Seeded {seeded_count} of {len(results)} new models.")
            else:
                print(f"Dry run: {len(results)} models would be seeded.")

    else:
        # Single model mode
        provider = args.provider if args.provider != "all" else "ollama"
        estimation = estimate_scores(
            args.model, provider,
            model_type=args.type,
            size_b=args.size,
        )

        if args.json:
            output = {
                "model": args.model,
                "provider": provider,
                "scores": estimation["scores"],
                "method": estimation["method"],
                "confidence": estimation["confidence"],
                "notes": estimation["notes"],
                "parsed": {k: v for k, v in estimation["parsed"].items() if k != "name"},
                "similar_models": estimation.get("similar_models", []),
            }
            print(json.dumps(output, indent=2))
        else:
            avg = sum(estimation["scores"].values()) / len(estimation["scores"])
            print(f"Trust score estimation for: {args.model} [{provider}]")
            print(f"  Method: {estimation['method']} (confidence: {estimation['confidence']})")
            print(f"  Type: {estimation['parsed']['model_type']}")
            print(f"  Family: {estimation['parsed']['family'] or 'unknown'}")
            print(f"  Size: {estimation['parsed']['size_b'] or 'unknown'}B")
            print(f"  Average: {avg * 100:.1f}%")
            print()
            print("  Per-task scores:")
            for task in TASK_TYPES:
                score = estimation["scores"].get(task, 0)
                bar = "#" * int(score * 20) + "-" * (20 - int(score * 20))
                print(f"    {task:15s} {score * 100:5.1f}% [{bar}]")

            if estimation.get("similar_models"):
                print(f"\n  Calibrated from similar models:")
                for name, sim in estimation["similar_models"]:
                    print(f"    - {name} (similarity: {sim})")

            if estimation.get("notes"):
                print(f"\n  Notes: {estimation['notes']}")

        # Seed if not dry-run
        if not args.dry_run:
            was_seeded = seed_model(
                args.model, provider,
                estimation["scores"],
                notes=estimation["notes"],
                force=args.force,
            )
            if was_seeded:
                if not args.json:
                    print(f"\n  Trust scores saved to: {trust_manager.TRUST_FILE}")
            else:
                if not args.json:
                    print(f"\n  Skipped: model already has trust data (use --force to overwrite)")


if __name__ == "__main__":
    main()
