#!/usr/bin/env python3
"""
Discover and assess available LLM models across providers.

Supports two providers:
  - ollama: Local models via Ollama REST API
  - gemini: Google Gemini API (requires GEMINI_API_KEY env var)

Usage:
    python3 discover_models.py [--provider PROVIDER] [--json] [--task TASK_TYPE]

Output:
    Human-readable summary of available models and their assessed capabilities.
    With --json, outputs structured JSON for programmatic use.
    With --task, recommends the best model for that task type.
"""

import argparse
import json
import sys
import urllib.request
import urllib.error
import re
import os

OLLAMA_DEFAULT_URL = "http://localhost:11434"
GEMINI_DEFAULT_URL = "https://generativelanguage.googleapis.com/v1beta"

# ---------------------------------------------------------------------------
# Ollama model classification
# ---------------------------------------------------------------------------

CODE_PATTERNS = [
    r"coder", r"codellama", r"starcoder", r"deepseek-coder",
    r"code-?llama", r"wizardcoder", r"phind", r"codestral",
    r"codegeex", r"codeqwen", r"qwen.*coder"
]

LARGE_REASONING_PATTERNS = [
    r"got-oss", r"gpt-oss", r"command-r", r"mixtral", r"dbrx",
    r"yi-large", r"internlm"
]

FAMILY_PROFILES = {
    "llama": {"type": "general", "strengths": ["reasoning", "instruction-following", "code"]},
    "qwen": {"type": "general", "strengths": ["reasoning", "multilingual", "code", "math"]},
    "qwen2": {"type": "general", "strengths": ["reasoning", "multilingual", "code", "math"]},
    "qwen4": {"type": "general", "strengths": ["reasoning", "multilingual", "code", "math"]},
    "gemma": {"type": "general", "strengths": ["reasoning", "instruction-following"]},
    "mistral": {"type": "general", "strengths": ["reasoning", "code", "efficiency"]},
    "phi": {"type": "general", "strengths": ["reasoning", "code", "efficiency"]},
    "deepseek": {"type": "general", "strengths": ["reasoning", "code", "math"]},
    "command-r": {"type": "general", "strengths": ["reasoning", "retrieval", "long-context"]},
    "starcoder": {"type": "code", "strengths": ["code-generation", "code-completion"]},
    "codellama": {"type": "code", "strengths": ["code-generation", "debugging", "code-review"]},
    "mixtral": {"type": "general", "strengths": ["reasoning", "code", "multilingual"]},
    "got-oss": {"type": "reasoning", "strengths": ["reasoning", "architecture", "planning", "analysis"]},
    "gpt-oss": {"type": "reasoning", "strengths": ["reasoning", "architecture", "planning", "analysis"]},
}

# ---------------------------------------------------------------------------
# Gemini model definitions
# ---------------------------------------------------------------------------

GEMINI_MODELS = {
    "gemini-2.5-flash": {
        "display_name": "Gemini 2.5 Flash",
        "primary_type": "general",
        "tier": "free",
        "rpm_limit": 10,
        "quality_tier": "good",
        "capabilities": ["code", "reasoning", "documentation", "review", "long-context",
                          "code-generation", "debugging", "code-review", "test-writing"],
        "recommended_tasks": [
            "Code generation (free tier)",
            "Code review with large context",
            "Documentation writing",
            "General development tasks",
        ],
        "context_note": "Up to 1M token context window",
        "note": "Best free-tier option. Fast, capable, huge context window.",
    },
    "gemini-2.5-pro": {
        "display_name": "Gemini 2.5 Pro",
        "primary_type": "reasoning",
        "tier": "free-limited",
        "rpm_limit": 5,
        "quality_tier": "high",
        "capabilities": ["reasoning", "architecture", "code", "documentation", "planning",
                          "review", "long-context", "code-generation", "code-review"],
        "recommended_tasks": [
            "Architecture and design analysis",
            "Complex reasoning tasks",
            "Comprehensive documentation",
            "Planning and task breakdown",
        ],
        "context_note": "Up to 1M token context window",
        "note": "Higher quality reasoning. Lower free RPM (5/min).",
    },
    "gemini-2.5-flash-lite": {
        "display_name": "Gemini 2.5 Flash Lite",
        "primary_type": "general",
        "tier": "free",
        "rpm_limit": 15,
        "quality_tier": "moderate",
        "capabilities": ["code", "documentation", "simple-tasks", "long-context"],
        "recommended_tasks": [
            "Simple code generation",
            "Quick documentation",
            "Text transformations",
        ],
        "context_note": "Up to 1M token context window",
        "note": "Fastest and cheapest. Good for simple tasks.",
    },
    "gemini-3-flash-preview": {
        "display_name": "Gemini 3 Flash Preview",
        "primary_type": "general",
        "tier": "preview",
        "rpm_limit": 10,
        "quality_tier": "good",
        "capabilities": ["code", "reasoning", "documentation", "review", "long-context",
                          "code-generation", "debugging", "code-review", "test-writing"],
        "recommended_tasks": [
            "Code generation (next-gen)",
            "Code review",
            "Documentation writing",
        ],
        "context_note": "Up to 1M token context window",
        "note": "Next-gen preview. May change.",
    },
    "gemini-3-pro-preview": {
        "display_name": "Gemini 3 Pro Preview",
        "primary_type": "reasoning",
        "tier": "preview",
        "rpm_limit": 5,
        "quality_tier": "high",
        "capabilities": ["reasoning", "architecture", "code", "documentation", "planning",
                          "review", "long-context", "code-generation", "code-review"],
        "recommended_tasks": [
            "Architecture and design analysis",
            "Complex multi-step reasoning",
            "Comprehensive documentation",
        ],
        "context_note": "Up to 1M token context window",
        "note": "Next-gen pro preview. May change.",
    },
}


# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------

def parse_parameter_size(size_str):
    """Parse parameter size string like '7B', '70B', '1.5B' into billions."""
    if not size_str:
        return None
    match = re.match(r"([\d.]+)\s*([BMK])", size_str.upper())
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2)
    if unit == "K":
        return value / 1_000_000
    elif unit == "M":
        return value / 1_000
    elif unit == "B":
        return value
    return None


def extract_size_from_name(name):
    """Try to extract parameter size from model name like 'qwen4-coder:30b'."""
    match = re.search(r":?(\d+\.?\d*)[bB]", name)
    if match:
        return float(match.group(1))
    return None


def classify_model(model_info):
    """Classify an Ollama model's capabilities based on name, family, and size."""
    name = model_info.get("name", "").lower()
    details = model_info.get("details", {})
    family = details.get("family", "").lower()
    families = [f.lower() for f in details.get("families", [])]
    param_size_str = details.get("parameter_size", "")
    param_size = parse_parameter_size(param_size_str)
    quant = details.get("quantization_level", "")

    if param_size is None:
        param_size = extract_size_from_name(name)
        if param_size and not param_size_str:
            param_size_str = f"{param_size}B"

    classification = {
        "name": model_info["name"],
        "provider": "ollama",
        "family": family,
        "families": families,
        "parameter_size": param_size_str,
        "parameter_size_b": param_size,
        "quantization": quant,
        "size_bytes": model_info.get("size", 0),
        "primary_type": "general",
        "capabilities": [],
        "recommended_tasks": [],
        "context_note": "",
        "quality_tier": "unknown",
    }

    is_code_model = any(re.search(p, name) for p in CODE_PATTERNS)
    is_large_reasoning = any(re.search(p, name) for p in LARGE_REASONING_PATTERNS)

    if is_code_model:
        classification["primary_type"] = "code"
        classification["capabilities"] = [
            "code-generation", "debugging", "code-review",
            "test-writing", "refactoring", "boilerplate"
        ]
        classification["recommended_tasks"] = [
            "Write functions/classes from specs",
            "Debug and fix code snippets",
            "Generate unit tests",
            "Code review",
            "Refactor code",
            "Generate boilerplate/scaffolding",
        ]
    elif is_large_reasoning:
        classification["primary_type"] = "reasoning"
        classification["capabilities"] = [
            "architecture", "planning", "documentation",
            "analysis", "reasoning", "explanation"
        ]
        classification["recommended_tasks"] = [
            "Architecture and design analysis",
            "Documentation writing",
            "Task breakdown and planning",
            "Code explanation",
            "Requirements analysis",
        ]

    if param_size is not None:
        if param_size >= 65:
            classification["quality_tier"] = "high"
            if classification["primary_type"] == "general":
                classification["capabilities"] = [
                    "reasoning", "code", "documentation",
                    "architecture", "planning", "review"
                ]
                classification["recommended_tasks"] = [
                    "Architecture and design discussions",
                    "Documentation writing",
                    "Complex code generation",
                    "Planning and task breakdown",
                    "Code review and analysis",
                ]
            if is_large_reasoning:
                classification["quality_tier"] = "high"
                if "code" not in classification["capabilities"]:
                    classification["capabilities"].append("code")
                if "review" not in classification["capabilities"]:
                    classification["capabilities"].append("review")
        elif param_size >= 25:
            classification["quality_tier"] = "good"
            if classification["primary_type"] == "general":
                classification["capabilities"] = [
                    "reasoning", "code", "documentation", "review"
                ]
                classification["recommended_tasks"] = [
                    "Code generation",
                    "Documentation drafts",
                    "Code review",
                    "Explaining code",
                ]
        elif param_size >= 7:
            classification["quality_tier"] = "moderate"
            if classification["primary_type"] == "general":
                classification["capabilities"] = [
                    "code", "documentation", "simple-tasks"
                ]
                classification["recommended_tasks"] = [
                    "Simple code generation",
                    "Short documentation",
                    "Code formatting",
                    "Simple Q&A about code",
                ]
        else:
            classification["quality_tier"] = "basic"
            if classification["primary_type"] == "general":
                classification["capabilities"] = ["simple-tasks", "formatting"]
                classification["recommended_tasks"] = [
                    "Text transformations",
                    "Short descriptions",
                    "Formatting tasks",
                ]

        if param_size >= 65:
            classification["context_note"] = "Likely 8K-32K+ context window"
        elif param_size >= 25:
            classification["context_note"] = "Likely 4K-16K context window"
        elif param_size >= 7:
            classification["context_note"] = "Likely 2K-8K context window"
        else:
            classification["context_note"] = "Likely 2K-4K context window"

    for fam_key, profile in FAMILY_PROFILES.items():
        if fam_key in family or fam_key in name:
            for strength in profile["strengths"]:
                if strength not in classification["capabilities"]:
                    classification["capabilities"].append(strength)
            break

    return classification


def fetch_ollama_models(base_url):
    """Fetch models from Ollama. Returns (models_list, error_message)."""
    url = f"{base_url}/api/tags"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            return data.get("models", []), None
    except urllib.error.URLError as e:
        return [], f"Cannot connect to Ollama at {base_url} ({e.reason}). Run: ollama serve"
    except Exception as e:
        return [], f"Error querying Ollama: {e}"


def fetch_running_models(base_url):
    """Fetch currently loaded/running Ollama models."""
    url = f"{base_url}/api/ps"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            return data.get("models", [])
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Gemini helpers
# ---------------------------------------------------------------------------

def classify_gemini_model(model_name):
    """Classify a Gemini model using the known model definitions."""
    info = GEMINI_MODELS.get(model_name)
    if not info:
        # Unknown Gemini model — provide basic classification
        return {
            "name": model_name,
            "provider": "gemini",
            "family": "gemini",
            "families": ["gemini"],
            "parameter_size": "Cloud",
            "parameter_size_b": None,
            "quantization": "",
            "size_bytes": 0,
            "primary_type": "general",
            "capabilities": ["reasoning", "code"],
            "recommended_tasks": ["General tasks"],
            "context_note": "Up to 1M token context window",
            "quality_tier": "good",
            "rpm_limit": None,
            "tier": "unknown",
        }

    return {
        "name": model_name,
        "provider": "gemini",
        "family": "gemini",
        "families": ["gemini"],
        "parameter_size": "Cloud",
        "parameter_size_b": None,
        "quantization": "",
        "size_bytes": 0,
        "primary_type": info["primary_type"],
        "capabilities": list(info["capabilities"]),
        "recommended_tasks": list(info["recommended_tasks"]),
        "context_note": info["context_note"],
        "quality_tier": info["quality_tier"],
        "rpm_limit": info["rpm_limit"],
        "tier": info["tier"],
    }


def check_gemini_availability():
    """Check if the Gemini API is accessible. Returns (available_models, error_message)."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return [], "GEMINI_API_KEY not set. Get one at: https://aistudio.google.com/apikey"

    url = f"{GEMINI_DEFAULT_URL}/models"
    req = urllib.request.Request(url)
    req.add_header("x-goog-api-key", api_key)

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
            models = data.get("models", [])
            # Filter to models we know about + any generateContent-capable models
            available = []
            for m in models:
                name = m.get("name", "").replace("models/", "")
                # Check if it supports generateContent
                methods = m.get("supportedGenerationMethods", [])
                if "generateContent" in methods:
                    if name in GEMINI_MODELS:
                        available.append(name)
            # If none of our known models matched, include any that support generation
            if not available:
                for m in models:
                    name = m.get("name", "").replace("models/", "")
                    methods = m.get("supportedGenerationMethods", [])
                    if "generateContent" in methods and "gemini" in name:
                        available.append(name)
            return available, None
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            return [], "Invalid GEMINI_API_KEY. Verify at: https://aistudio.google.com/apikey"
        return [], f"Gemini API error: HTTP {e.code}"
    except urllib.error.URLError as e:
        return [], f"Cannot reach Gemini API ({e.reason}). Check internet connection."
    except Exception as e:
        return [], f"Error checking Gemini: {e}"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_size(size_bytes):
    """Format byte size to human-readable."""
    if size_bytes >= 1e9:
        return f"{size_bytes / 1e9:.1f} GB"
    elif size_bytes >= 1e6:
        return f"{size_bytes / 1e6:.1f} MB"
    return f"{size_bytes} B"


def print_human_report(classifications, running_models, provider_status):
    """Print a human-readable report of all available models."""
    ollama_models = [c for c in classifications if c.get("provider") == "ollama"]
    gemini_models = [c for c in classifications if c.get("provider") == "gemini"]

    if not classifications:
        print("No models found from any provider.\n")
        if provider_status.get("ollama_error"):
            print(f"  Ollama: {provider_status['ollama_error']}")
        if provider_status.get("gemini_error"):
            print(f"  Gemini: {provider_status['gemini_error']}")
        print("\nTo get started:")
        print("  ollama pull qwen2.5-coder:32b        # Local code model")
        print("  export GEMINI_API_KEY=your-key-here   # Cloud models via Gemini")
        return

    running_names = {m.get("name", "") for m in running_models}
    total = len(classifications)
    print(f"Found {total} model(s) across {len(set(c.get('provider') for c in classifications))} provider(s):\n")

    # Report provider errors
    if provider_status.get("ollama_error"):
        print(f"  ⚠ Ollama: {provider_status['ollama_error']}")
    if provider_status.get("gemini_error"):
        print(f"  ⚠ Gemini: {provider_status['gemini_error']}")
    if provider_status.get("ollama_error") or provider_status.get("gemini_error"):
        print()

    def print_ollama_group(title, models):
        if not models:
            return
        print(f"=== {title} (Ollama) ===")
        for m in sorted(models, key=lambda x: x.get("parameter_size_b") or 0, reverse=True):
            loaded = " [LOADED]" if m["name"] in running_names else ""
            size = format_size(m["size_bytes"])
            print(f"  {m['name']}{loaded}")
            print(f"    Size: {m['parameter_size'] or 'unknown'} params, {size} on disk")
            print(f"    Quality: {m['quality_tier']} | Quant: {m['quantization'] or 'unknown'}")
            if m["context_note"]:
                print(f"    Context: {m['context_note']}")
            if m["recommended_tasks"]:
                print(f"    Best for: {', '.join(m['recommended_tasks'][:3])}")
            print()

    # Ollama sections
    if ollama_models:
        code_models = [c for c in ollama_models if c["primary_type"] == "code"]
        reasoning_models = [c for c in ollama_models if c["primary_type"] == "reasoning"]
        general_models = [c for c in ollama_models if c["primary_type"] == "general"]
        print_ollama_group("Code-Specialized Models", code_models)
        print_ollama_group("Large Reasoning Models", reasoning_models)
        print_ollama_group("General-Purpose Models", general_models)

    # Gemini section
    if gemini_models:
        print("=== Cloud Models (Gemini) ===")
        for m in gemini_models:
            rpm = m.get("rpm_limit", "?")
            tier = m.get("tier", "")
            tier_label = f" [{tier}]" if tier else ""
            print(f"  {m['name']}{tier_label}")
            print(f"    Quality: {m['quality_tier']} | Rate limit: {rpm} RPM")
            print(f"    Context: {m['context_note']}")
            if m["recommended_tasks"]:
                print(f"    Best for: {', '.join(m['recommended_tasks'][:3])}")
            print()

    # Routing recommendations
    print("=== Routing Recommendations ===")

    # Best code model (prefer Ollama)
    ollama_code = [c for c in ollama_models if c["primary_type"] == "code"]
    gemini_code = [c for c in gemini_models if "code-generation" in c.get("capabilities", [])]
    if ollama_code:
        best = max(ollama_code, key=lambda x: x.get("parameter_size_b") or 0)
        print(f"  Code tasks     -> {best['name']} [ollama] ({best['parameter_size']})")
    elif gemini_code:
        print(f"  Code tasks     -> {gemini_code[0]['name']} [gemini]")
    else:
        ollama_gen_code = [c for c in ollama_models if "code" in c.get("capabilities", [])]
        if ollama_gen_code:
            best = max(ollama_gen_code, key=lambda x: x.get("parameter_size_b") or 0)
            print(f"  Code tasks     -> {best['name']} [ollama, general-purpose]")
        else:
            print("  Code tasks     -> No model found. Try: ollama pull qwen2.5-coder:32b")

    # Best reasoning model (prefer Ollama, fallback to Gemini)
    ollama_reason = [c for c in ollama_models if c["primary_type"] == "reasoning"]
    ollama_reason += [c for c in ollama_models if c["quality_tier"] in ("high", "good") and c["primary_type"] == "general"]
    gemini_reason = [c for c in gemini_models if c["quality_tier"] == "high"]
    if ollama_reason:
        best = max(ollama_reason, key=lambda x: x.get("parameter_size_b") or 0)
        print(f"  Reasoning      -> {best['name']} [ollama] ({best['parameter_size']})")
    elif gemini_reason:
        print(f"  Reasoning      -> {gemini_reason[0]['name']} [gemini]")
    else:
        print("  Reasoning      -> No model found. Try: export GEMINI_API_KEY=...")

    # Best for large context
    if gemini_models:
        print(f"  Large context  -> {gemini_models[0]['name']} [gemini] (1M tokens)")

    print()


def score_models_for_task(classifications, task):
    """Score all models for a given task type. Returns sorted list of (score, classification)."""
    task_type_map = {
        "code": ["code-generation", "code"],
        "review": ["code-review", "review"],
        "docs": ["documentation"],
        "documentation": ["documentation"],
        "architecture": ["architecture", "reasoning"],
        "planning": ["planning", "reasoning"],
        "test": ["test-writing", "code-generation"],
        "debug": ["debugging", "code"],
        "refactor": ["refactoring", "code"],
        "general": ["reasoning"],
    }
    target_caps = task_type_map.get(task, ["reasoning"])

    scored = []
    for c in classifications:
        score = sum(1 for cap in target_caps if cap in c.get("capabilities", []))

        # Size bonus for Ollama models
        psb = c.get("parameter_size_b") or 0
        if psb >= 65:
            score += 2
        elif psb >= 25:
            score += 1

        # Quality tier bonus for Gemini (since they have no parameter_size_b)
        if c.get("provider") == "gemini":
            tier = c.get("quality_tier", "")
            if tier == "high":
                score += 2
            elif tier == "good":
                score += 1

        # Slight preference for Ollama (local-first)
        if c.get("provider") == "ollama" and score > 0:
            score += 0.1

        scored.append((score, c))

    scored.sort(key=lambda x: (-x[0], -(x[1].get("parameter_size_b") or 0)))
    return scored


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Discover and assess available LLM models")
    parser.add_argument("--provider", default="all", choices=["all", "ollama", "gemini"],
                        help="Which provider(s) to check (default: all)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--url", default=OLLAMA_DEFAULT_URL, help="Ollama API URL")
    parser.add_argument("--task",
                        help="Suggest best model for a task: code, review, docs, architecture, test, debug, general")
    args = parser.parse_args()

    classifications = []
    running = []
    provider_status = {}

    # Discover Ollama models
    if args.provider in ("all", "ollama"):
        models, ollama_err = fetch_ollama_models(args.url)
        if ollama_err:
            provider_status["ollama_error"] = ollama_err
            if args.provider == "ollama":
                print(f"ERROR: {ollama_err}", file=sys.stderr)
                sys.exit(1)
            else:
                print(f"Warning: {ollama_err}", file=sys.stderr)
        else:
            provider_status["ollama"] = "ok"
            running = fetch_running_models(args.url)
            classifications.extend(classify_model(m) for m in models)

    # Discover Gemini models
    if args.provider in ("all", "gemini"):
        gemini_models, gemini_err = check_gemini_availability()
        if gemini_err:
            provider_status["gemini_error"] = gemini_err
            if args.provider == "gemini":
                print(f"ERROR: {gemini_err}", file=sys.stderr)
                sys.exit(1)
            else:
                print(f"Warning: {gemini_err}", file=sys.stderr)
        else:
            provider_status["gemini"] = "ok"
            classifications.extend(classify_gemini_model(name) for name in gemini_models)

    # Task-specific recommendation
    if args.task:
        scored = score_models_for_task(classifications, args.task)

        if args.json:
            if scored and scored[0][0] > 0:
                best = scored[0][1]
                print(json.dumps({
                    "recommended": best["name"],
                    "provider": best.get("provider", "unknown"),
                    "all": [s[1] for s in scored if s[0] > 0],
                }, indent=2))
            else:
                print(json.dumps({"recommended": None, "all": []}, indent=2))
        else:
            if scored and scored[0][0] > 0:
                best = scored[0][1]
                provider = best.get("provider", "unknown")
                print(f"Best model for '{args.task}': {best['name']} [{provider}]")
                print(f"  Quality: {best['quality_tier']}")
                print(f"  Capabilities: {', '.join(best['capabilities'][:5])}")
                if best.get("parameter_size"):
                    print(f"  Size: {best['parameter_size']}")
            else:
                print(f"No suitable model found for '{args.task}' tasks.")
        return

    # Full report
    if args.json:
        output = {
            "providers": provider_status,
            "model_count": len(classifications),
            "models": classifications,
            "running": [m.get("name") for m in running],
        }
        print(json.dumps(output, indent=2))
    else:
        print_human_report(classifications, running, provider_status)


if __name__ == "__main__":
    main()
