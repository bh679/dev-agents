#!/usr/bin/env python3
"""
Seed initial trust scores from benchmark research data.

Run this once to populate trust_scores.json with baseline scores
derived from published benchmarks (HumanEval, MBPP, SWE-bench, etc.).

These are "research-seeded" scores — they carry less weight than
actual observed performance. Each model starts with count=1 so the
first real assessment will blend 50/50 with the seed, and after
3 real assessments the EWMA will dominate.

Usage:
    python3 scripts/seed_trust.py [--force]

    --force: Overwrite existing trust data (default: skip models that already have scores)
"""

import json
import os
import sys

# Add scripts dir to path for trust_manager import
scripts_dir = os.path.dirname(os.path.abspath(__file__))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

import trust_manager

# ---------------------------------------------------------------------------
# Benchmark-derived initial trust scores
#
# Scale: 0.0 - 1.0
# Sources: HumanEval, MBPP, BigCodeBench, SWE-bench, LiveCodeBench,
#          MMLU, ARC, GPQA Diamond, Aider, published model cards
#
# Last updated: 2026-02-23
#
# Methodology:
#   code:         HumanEval pass@1 mapped to 0-1 scale (60%=0.5, 90%+=0.9)
#   review:       Inferred from reasoning + code capabilities
#   debug:        Inferred from code + reasoning benchmarks
#   test:         Inferred from code generation benchmarks (slightly lower)
#   docs:         Inferred from reasoning + language quality
#   architecture: MMLU/reasoning benchmarks mapped to 0-1 scale
#   refactor:     Blend of code + reasoning scores
#   general:      Overall average of capabilities
# ---------------------------------------------------------------------------

SEED_DATA = {
    # ===== Ollama Models =====

    # --- Qwen2.5-Coder family ---
    # HumanEval: 88.4% (32B), LiveCodeBench: 51.2%, Aider: 73.7
    # BigCodeBench: state-of-art open source, competitive with GPT-4o
    "qwen2.5-coder:32b": {
        "provider": "ollama",
        "scores": {
            "code": 0.88, "review": 0.78, "debug": 0.82,
            "test": 0.80, "docs": 0.65, "architecture": 0.62,
            "refactor": 0.82, "general": 0.72,
        },
        "notes": "Top open-source code model. HumanEval 88.4%, competitive with GPT-4o on Aider.",
    },
    "qwen2.5-coder:14b": {
        "provider": "ollama",
        "scores": {
            "code": 0.78, "review": 0.68, "debug": 0.72,
            "test": 0.70, "docs": 0.55, "architecture": 0.50,
            "refactor": 0.72, "general": 0.62,
        },
        "notes": "Good mid-size code model. Strong code gen, weaker on reasoning tasks.",
    },
    "qwen2.5-coder:7b": {
        "provider": "ollama",
        "scores": {
            "code": 0.68, "review": 0.55, "debug": 0.60,
            "test": 0.58, "docs": 0.45, "architecture": 0.38,
            "refactor": 0.60, "general": 0.50,
        },
        "notes": "Decent small code model. Best for simple code gen and boilerplate.",
    },

    # --- Qwen2.5 general family ---
    "qwen2.5:72b": {
        "provider": "ollama",
        "scores": {
            "code": 0.80, "review": 0.78, "debug": 0.75,
            "test": 0.72, "docs": 0.80, "architecture": 0.78,
            "refactor": 0.75, "general": 0.78,
        },
        "notes": "Strong all-rounder at 72B. Excellent reasoning and documentation.",
    },
    "qwen2.5:32b": {
        "provider": "ollama",
        "scores": {
            "code": 0.72, "review": 0.68, "debug": 0.68,
            "test": 0.65, "docs": 0.70, "architecture": 0.68,
            "refactor": 0.68, "general": 0.68,
        },
        "notes": "Good general model. Balanced across tasks.",
    },
    "qwen2.5:14b": {
        "provider": "ollama",
        "scores": {
            "code": 0.62, "review": 0.55, "debug": 0.58,
            "test": 0.55, "docs": 0.58, "architecture": 0.52,
            "refactor": 0.58, "general": 0.55,
        },
        "notes": "Mid-size general model. Adequate for most tasks.",
    },
    "qwen2.5:7b": {
        "provider": "ollama",
        "scores": {
            "code": 0.52, "review": 0.45, "debug": 0.48,
            "test": 0.45, "docs": 0.48, "architecture": 0.40,
            "refactor": 0.48, "general": 0.45,
        },
        "notes": "Small general model. Suitable for simple tasks only.",
    },

    # --- DeepSeek family ---
    # HumanEval: 90.2% (V2-236B), MBPP: 76.2%, LiveCodeBench: 43.4%
    "deepseek-coder-v2:236b": {
        "provider": "ollama",
        "scores": {
            "code": 0.90, "review": 0.80, "debug": 0.85,
            "test": 0.82, "docs": 0.70, "architecture": 0.72,
            "refactor": 0.85, "general": 0.78,
        },
        "notes": "HumanEval 90.2%. On par with GPT-4 Turbo for code. Very large model.",
    },
    "deepseek-coder-v2:16b": {
        "provider": "ollama",
        "scores": {
            "code": 0.78, "review": 0.65, "debug": 0.72,
            "test": 0.68, "docs": 0.55, "architecture": 0.52,
            "refactor": 0.72, "general": 0.62,
        },
        "notes": "Efficient MoE model. Strong code gen for its active parameter count.",
    },
    "deepseek-v3": {
        "provider": "ollama",
        "scores": {
            "code": 0.85, "review": 0.80, "debug": 0.82,
            "test": 0.78, "docs": 0.78, "architecture": 0.80,
            "refactor": 0.82, "general": 0.80,
        },
        "notes": "Top-tier open model. Excellent across code and reasoning.",
    },

    # --- Llama family ---
    # HumanEval: 80.5% (70B), 72.6% (8B)
    "llama3.1:70b": {
        "provider": "ollama",
        "scores": {
            "code": 0.75, "review": 0.72, "debug": 0.72,
            "test": 0.68, "docs": 0.75, "architecture": 0.75,
            "refactor": 0.72, "general": 0.75,
        },
        "notes": "HumanEval 80.5%. Strong reasoning, good code. Well-rounded.",
    },
    "llama3.1:8b": {
        "provider": "ollama",
        "scores": {
            "code": 0.58, "review": 0.48, "debug": 0.52,
            "test": 0.48, "docs": 0.52, "architecture": 0.45,
            "refactor": 0.50, "general": 0.50,
        },
        "notes": "HumanEval 72.6%. Good for its size, better than Mistral 7B at code.",
    },
    "llama3.3:70b": {
        "provider": "ollama",
        "scores": {
            "code": 0.78, "review": 0.75, "debug": 0.75,
            "test": 0.70, "docs": 0.78, "architecture": 0.78,
            "refactor": 0.75, "general": 0.78,
        },
        "notes": "Improved over 3.1. Strong all-rounder for reasoning and code.",
    },

    # --- CodeLlama family ---
    "codellama:70b": {
        "provider": "ollama",
        "scores": {
            "code": 0.72, "review": 0.62, "debug": 0.68,
            "test": 0.65, "docs": 0.50, "architecture": 0.48,
            "refactor": 0.65, "general": 0.55,
        },
        "notes": "Specialized code model. Good at code gen but weaker on reasoning.",
    },
    "codellama:34b": {
        "provider": "ollama",
        "scores": {
            "code": 0.65, "review": 0.55, "debug": 0.60,
            "test": 0.58, "docs": 0.42, "architecture": 0.40,
            "refactor": 0.58, "general": 0.48,
        },
        "notes": "Mid-size code specialist. Reasonable for code tasks.",
    },
    "codellama:13b": {
        "provider": "ollama",
        "scores": {
            "code": 0.55, "review": 0.45, "debug": 0.50,
            "test": 0.48, "docs": 0.35, "architecture": 0.30,
            "refactor": 0.48, "general": 0.38,
        },
        "notes": "Small code model. Best for Python/Java/C++ snippets.",
    },
    "codellama:7b": {
        "provider": "ollama",
        "scores": {
            "code": 0.45, "review": 0.35, "debug": 0.40,
            "test": 0.38, "docs": 0.28, "architecture": 0.22,
            "refactor": 0.38, "general": 0.30,
        },
        "notes": "Smallest CodeLlama. Only for simple snippets and completion.",
    },

    # --- Mistral / Mixtral ---
    "mistral:7b": {
        "provider": "ollama",
        "scores": {
            "code": 0.52, "review": 0.48, "debug": 0.48,
            "test": 0.45, "docs": 0.52, "architecture": 0.45,
            "refactor": 0.48, "general": 0.50,
        },
        "notes": "Efficient 7B model. Punches above weight, fast inference.",
    },
    "mixtral:8x7b": {
        "provider": "ollama",
        "scores": {
            "code": 0.65, "review": 0.60, "debug": 0.62,
            "test": 0.58, "docs": 0.62, "architecture": 0.58,
            "refactor": 0.60, "general": 0.60,
        },
        "notes": "MoE model, 12.9B active params. Good balance of speed and quality.",
    },
    "mixtral:8x22b": {
        "provider": "ollama",
        "scores": {
            "code": 0.75, "review": 0.70, "debug": 0.72,
            "test": 0.68, "docs": 0.72, "architecture": 0.70,
            "refactor": 0.72, "general": 0.72,
        },
        "notes": "Large MoE. Top open model for code+math. 39B active params.",
    },

    # --- Phi family ---
    "phi-4:14b": {
        "provider": "ollama",
        "scores": {
            "code": 0.72, "review": 0.62, "debug": 0.65,
            "test": 0.62, "docs": 0.58, "architecture": 0.55,
            "refactor": 0.65, "general": 0.60,
        },
        "notes": "Microsoft Phi-4. Excellent for its size, especially at code and math.",
    },
    "phi-3:14b": {
        "provider": "ollama",
        "scores": {
            "code": 0.65, "review": 0.55, "debug": 0.58,
            "test": 0.55, "docs": 0.52, "architecture": 0.48,
            "refactor": 0.58, "general": 0.55,
        },
        "notes": "Phi-3 medium. Good efficiency-to-quality ratio.",
    },
    "phi-3:3.8b": {
        "provider": "ollama",
        "scores": {
            "code": 0.48, "review": 0.38, "debug": 0.42,
            "test": 0.38, "docs": 0.40, "architecture": 0.32,
            "refactor": 0.40, "general": 0.38,
        },
        "notes": "Phi-3 mini. Lightweight, for simple tasks on constrained hardware.",
    },

    # --- StarCoder2 ---
    "starcoder2:15b": {
        "provider": "ollama",
        "scores": {
            "code": 0.62, "review": 0.48, "debug": 0.55,
            "test": 0.52, "docs": 0.35, "architecture": 0.28,
            "refactor": 0.55, "general": 0.40,
        },
        "notes": "Code-only model. Strong on low-resource languages (Julia, Lua, Perl).",
    },
    "starcoder2:7b": {
        "provider": "ollama",
        "scores": {
            "code": 0.52, "review": 0.38, "debug": 0.45,
            "test": 0.42, "docs": 0.28, "architecture": 0.22,
            "refactor": 0.45, "general": 0.32,
        },
        "notes": "Smaller StarCoder2. Best for code completion, not instruction following.",
    },
    "starcoder2:3b": {
        "provider": "ollama",
        "scores": {
            "code": 0.42, "review": 0.28, "debug": 0.35,
            "test": 0.32, "docs": 0.22, "architecture": 0.18,
            "refactor": 0.35, "general": 0.25,
        },
        "notes": "Tiny code model. Only for autocomplete-style use.",
    },

    # --- Gemma2 ---
    "gemma2:27b": {
        "provider": "ollama",
        "scores": {
            "code": 0.72, "review": 0.65, "debug": 0.68,
            "test": 0.62, "docs": 0.68, "architecture": 0.62,
            "refactor": 0.65, "general": 0.65,
        },
        "notes": "Google's Gemma2 27B. HumanEval ~87.8% (Gemma3 27B). Strong all-around.",
    },
    "gemma2:9b": {
        "provider": "ollama",
        "scores": {
            "code": 0.55, "review": 0.48, "debug": 0.50,
            "test": 0.48, "docs": 0.52, "architecture": 0.45,
            "refactor": 0.50, "general": 0.50,
        },
        "notes": "Gemma2 9B. Outperforms Llama 3 8B on many benchmarks.",
    },
    "gemma2:2b": {
        "provider": "ollama",
        "scores": {
            "code": 0.32, "review": 0.25, "debug": 0.28,
            "test": 0.25, "docs": 0.30, "architecture": 0.22,
            "refactor": 0.28, "general": 0.28,
        },
        "notes": "Tiny model. Only for edge deployment, not serious dev tasks.",
    },

    # --- Command-R ---
    "command-r:35b": {
        "provider": "ollama",
        "scores": {
            "code": 0.62, "review": 0.60, "debug": 0.58,
            "test": 0.55, "docs": 0.65, "architecture": 0.62,
            "refactor": 0.58, "general": 0.62,
        },
        "notes": "Cohere Command-R. Strong at retrieval, reasoning, long-context.",
    },
    "command-r-plus:104b": {
        "provider": "ollama",
        "scores": {
            "code": 0.72, "review": 0.70, "debug": 0.68,
            "test": 0.65, "docs": 0.75, "architecture": 0.72,
            "refactor": 0.68, "general": 0.72,
        },
        "notes": "Command-R+. Excellent reasoning and docs. Large model.",
    },

    # --- GPT-OSS ---
    "gpt-oss:120b": {
        "provider": "ollama",
        "scores": {
            "code": 0.78, "review": 0.75, "debug": 0.75,
            "test": 0.72, "docs": 0.78, "architecture": 0.80,
            "refactor": 0.75, "general": 0.78,
        },
        "notes": "Large reasoning model. Strong on architecture and planning.",
    },

    # ===== Gemini Models =====

    # --- Gemini 2.5 Pro ---
    # SWE-bench: 63.8%, Aider: 74.0%, LiveCodeBench: 70.4%
    # HumanEval: 68.9%, HumanEval+: 73%
    "gemini-2.5-pro": {
        "provider": "gemini",
        "scores": {
            "code": 0.82, "review": 0.85, "debug": 0.82,
            "test": 0.78, "docs": 0.88, "architecture": 0.90,
            "refactor": 0.80, "general": 0.85,
        },
        "notes": "SWE-bench 63.8%, Aider 74%. Top reasoning model. Excellent for architecture/docs.",
    },

    # --- Gemini 2.5 Flash ---
    # HumanEval: 74.3%, lower hallucination (3.1%)
    "gemini-2.5-flash": {
        "provider": "gemini",
        "scores": {
            "code": 0.78, "review": 0.75, "debug": 0.75,
            "test": 0.72, "docs": 0.78, "architecture": 0.72,
            "refactor": 0.75, "general": 0.75,
        },
        "notes": "HumanEval 74.3%. Low hallucination rate. Best free-tier option.",
    },

    # --- Gemini 2.5 Flash Lite ---
    "gemini-2.5-flash-lite": {
        "provider": "gemini",
        "scores": {
            "code": 0.58, "review": 0.52, "debug": 0.55,
            "test": 0.50, "docs": 0.55, "architecture": 0.45,
            "refactor": 0.52, "general": 0.52,
        },
        "notes": "Lowest cost. Good for simple tasks. Optional thinking mode improves math/code.",
    },

    # --- Gemini 3 Flash ---
    # SWE-bench: 78%, GPQA Diamond: 90.4%, outperforms 2.5 Pro
    "gemini-3-flash-preview": {
        "provider": "gemini",
        "scores": {
            "code": 0.88, "review": 0.85, "debug": 0.85,
            "test": 0.82, "docs": 0.85, "architecture": 0.88,
            "refactor": 0.85, "general": 0.88,
        },
        "notes": "SWE-bench 78%, GPQA 90.4%. Outperforms 2.5 Pro at 3x speed. Preview model.",
    },

    # --- Gemini 3 Pro Preview ---
    "gemini-3-pro-preview": {
        "provider": "gemini",
        "scores": {
            "code": 0.85, "review": 0.88, "debug": 0.85,
            "test": 0.80, "docs": 0.90, "architecture": 0.92,
            "refactor": 0.85, "general": 0.90,
        },
        "notes": "Strongest reasoning. Best for architecture/planning. Preview, may change.",
    },

    # --- Gemini 3.1 Pro ---
    "gemini-3.1-pro": {
        "provider": "gemini",
        "scores": {
            "code": 0.88, "review": 0.90, "debug": 0.88,
            "test": 0.82, "docs": 0.92, "architecture": 0.95,
            "refactor": 0.88, "general": 0.92,
        },
        "notes": "Doubled reasoning scores vs 3.0 Pro. Record-breaking reasoning benchmarks.",
    },
}


def seed_trust_scores(force=False):
    """Seed trust scores from benchmark research data."""
    trust_manager.ensure_data_dirs()
    data = trust_manager.load_trust_scores()
    now = trust_manager._iso_now()
    seeded = 0
    skipped = 0

    for model_name, seed_info in SEED_DATA.items():
        provider = seed_info["provider"]
        scores = seed_info["scores"]
        notes = seed_info.get("notes", "")

        # Check if model already has trust data
        if model_name in data.get("models", {}) and not force:
            existing = data["models"][model_name]
            if existing.get("total_runs", 0) > 0:
                skipped += 1
                continue

        # Build task scores
        task_scores = {}
        for task_type, score in scores.items():
            task_scores[task_type] = {
                "score": score,
                "count": 1,  # Low count so real assessments quickly override
                "last_updated": now,
                "_ratings_sum": score,
                "_source": "benchmark-seed",
            }

        # Compute global score
        total_weight = sum(1 for _ in scores.values())
        global_score = round(sum(scores.values()) / total_weight, 4) if total_weight else 0

        data.setdefault("models", {})[model_name] = {
            "provider": provider,
            "global_score": global_score,
            "task_scores": task_scores,
            "total_runs": 1,
            "first_seen": now,
            "last_used": now,
            "_seed_notes": notes,
            "_seeded_from": "benchmark-research",
        }
        seeded += 1

        # Also log as assessment for audit trail
        for task_type, score in scores.items():
            # Convert 0-1 score back to 1-5 rating
            rating = max(1, min(5, round(score * 4 + 1)))
            trust_manager.log_assessment({
                "model": model_name,
                "provider": provider,
                "task_type": task_type,
                "rating": rating,
                "prompt_summary": f"[SEED] Benchmark-derived initial score",
                "response_length": 0,
                "eval_tokens": 0,
                "duration_ms": 0,
                "tokens_per_second": 0,
                "comparison_id": None,
                "notes": f"seed: {notes}",
            })

    trust_manager.save_trust_scores(data)
    return seeded, skipped


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Seed initial trust scores from benchmark research")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing trust data for seeded models")
    parser.add_argument("--list", action="store_true",
                        help="List all seed data without applying")
    parser.add_argument("--json", action="store_true",
                        help="Output seed data as JSON")
    args = parser.parse_args()

    if args.list or args.json:
        if args.json:
            print(json.dumps(SEED_DATA, indent=2))
        else:
            print(f"Seed data for {len(SEED_DATA)} models:\n")
            for name, info in sorted(SEED_DATA.items()):
                provider = info["provider"]
                scores = info["scores"]
                avg = sum(scores.values()) / len(scores)
                print(f"  {name} [{provider}]")
                print(f"    Average: {avg*100:.1f}%")
                best_task = max(scores, key=scores.get)
                worst_task = min(scores, key=scores.get)
                print(f"    Best: {best_task} ({scores[best_task]*100:.0f}%) | Worst: {worst_task} ({scores[worst_task]*100:.0f}%)")
                if info.get("notes"):
                    print(f"    {info['notes']}")
                print()
        return

    seeded, skipped = seed_trust_scores(force=args.force)
    print(f"Seeded {seeded} models, skipped {skipped} (already had data).")
    if seeded > 0:
        print(f"Trust data saved to: {trust_manager.TRUST_FILE}")
        print(f"Assessment log: {trust_manager.ASSESSMENTS_FILE}")
    if skipped > 0 and not args.force:
        print(f"Use --force to overwrite existing data.")


if __name__ == "__main__":
    main()
