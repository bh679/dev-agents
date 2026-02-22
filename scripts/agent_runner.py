#!/usr/bin/env python3
"""
Run a development sub-task through a local or cloud LLM.

Supports two providers:
  - ollama: Local models via Ollama REST API (default)
  - gemini: Google Gemini API (requires GEMINI_API_KEY env var)

Usage:
    python3 agent_runner.py --provider PROVIDER --model MODEL --prompt PROMPT [options]

Examples:
    # Ollama: code generation with local model
    python3 agent_runner.py \\
        --model "qwen2.5-coder:32b" \\
        --task code \\
        --prompt "Write a function to merge two sorted lists."

    # Gemini: code generation with cloud model
    python3 agent_runner.py \\
        --provider gemini \\
        --model "gemini-2.5-flash" \\
        --task code \\
        --prompt "Write a function to merge two sorted lists."

    # Gemini: code review with file context
    python3 agent_runner.py \\
        --provider gemini \\
        --model "gemini-2.5-pro" \\
        --task review \\
        --prompt "Review this code for bugs and improvements:" \\
        --file /path/to/code.py

    # Ollama: documentation generation
    python3 agent_runner.py \\
        --model "gpt-oss:120b" \\
        --task docs \\
        --prompt "Write API docs for this module:" \\
        --file /path/to/module.py \\
        --temperature 0.5

    # Compare two models:
    python3 agent_runner.py \\
        --model "qwen2.5-coder:32b" \\
        --model-b "gemini-2.5-flash" --provider-b gemini \\
        --task code \\
        --prompt "Write a function to merge two sorted lists." \\
        --compare --no-stream

    # Run with assessment:
    python3 agent_runner.py \\
        --model "qwen2.5-coder:32b" \\
        --task code \\
        --prompt "Write an email validator." \\
        --assess --rating 4
"""

import argparse
import json
import random
import subprocess
import sys
import time
import tempfile
import urllib.request
import urllib.error
import os

OLLAMA_DEFAULT_URL = "http://localhost:11434"
GEMINI_DEFAULT_URL = "https://generativelanguage.googleapis.com/v1beta"

# Default system prompts per task type
TASK_SYSTEM_PROMPTS = {
    "code": (
        "You are an expert software developer. Write clean, well-structured, "
        "production-ready code. Include type hints where appropriate. "
        "Return only the code unless explicitly asked for explanation."
    ),
    "review": (
        "You are a thorough code reviewer. Identify bugs, performance issues, "
        "security concerns, and style problems. Be specific — reference line "
        "numbers or code snippets. Prioritize issues by severity. Be concise."
    ),
    "debug": (
        "You are a debugging expert. Analyze the code and error information to "
        "identify the root cause. Explain the bug clearly and provide a fix. "
        "Show the corrected code."
    ),
    "test": (
        "You are a test engineering expert. Write comprehensive tests that cover "
        "happy paths, edge cases, and error conditions. Use the appropriate test "
        "framework for the language. Include descriptive test names."
    ),
    "docs": (
        "You are a technical writer. Write clear, comprehensive documentation. "
        "Include: overview, parameters, return values, examples, and edge cases. "
        "Use the appropriate documentation format for the language."
    ),
    "architecture": (
        "You are a software architect. Analyze the design, identify patterns, "
        "suggest improvements, and consider trade-offs. Be practical — focus on "
        "actionable recommendations rather than theoretical ideals."
    ),
    "refactor": (
        "You are a refactoring expert. Improve code structure, readability, and "
        "maintainability while preserving behavior. Explain what you changed and why. "
        "Show the refactored code."
    ),
    "general": (
        "You are a helpful software development assistant. Be concise and direct."
    ),
}

# Recommended temperature per task type
TASK_TEMPERATURES = {
    "code": 0.3,
    "review": 0.4,
    "debug": 0.2,
    "test": 0.3,
    "docs": 0.5,
    "architecture": 0.6,
    "refactor": 0.3,
    "general": 0.5,
}


# ---------------------------------------------------------------------------
# Trust manager integration
# ---------------------------------------------------------------------------

def _get_trust_manager():
    """Import trust_manager from the same scripts directory."""
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    try:
        import trust_manager
        return trust_manager
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def read_file_content(file_path):
    """Read file content, with error handling."""
    try:
        with open(file_path, "r") as f:
            content = f.read()
        return content
    except FileNotFoundError:
        print(f"ERROR: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Cannot read file {file_path}: {e}", file=sys.stderr)
        sys.exit(1)


def build_user_content(user_prompt, file_content=None, file_path=None):
    """Build the user message content string, optionally including file context."""
    content = user_prompt
    if file_content:
        filename = os.path.basename(file_path) if file_path else "input"
        content = f"{user_prompt}\n\n```{filename}\n{file_content}\n```"
    return content


# ---------------------------------------------------------------------------
# Ollama provider
# ---------------------------------------------------------------------------

def build_ollama_messages(system_prompt, user_content):
    """Build Ollama chat messages array."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    return messages


def call_ollama_streaming(base_url, model, messages, temperature, num_ctx, timeout):
    """Call Ollama chat API with streaming, printing tokens as they arrive."""
    url = f"{base_url}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": temperature,
        },
    }

    if num_ctx:
        payload["options"]["num_ctx"] = num_ctx

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    full_response = []
    stats = {}

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            buffer = b""
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line.decode("utf-8"))
                    except json.JSONDecodeError:
                        continue

                    if obj.get("message", {}).get("content"):
                        token = obj["message"]["content"]
                        sys.stdout.write(token)
                        sys.stdout.flush()
                        full_response.append(token)

                    if obj.get("done"):
                        stats = {
                            "total_duration_ms": obj.get("total_duration", 0) / 1e6,
                            "prompt_eval_count": obj.get("prompt_eval_count", 0),
                            "eval_count": obj.get("eval_count", 0),
                            "eval_duration_ms": obj.get("eval_duration", 0) / 1e6,
                            "tokens_per_second": (
                                obj.get("eval_count", 0) / (obj.get("eval_duration", 1) / 1e9)
                                if obj.get("eval_duration", 0) > 0
                                else 0
                            ),
                        }

            if buffer.strip():
                try:
                    obj = json.loads(buffer.decode("utf-8"))
                    if obj.get("message", {}).get("content"):
                        token = obj["message"]["content"]
                        sys.stdout.write(token)
                        sys.stdout.flush()
                        full_response.append(token)
                except json.JSONDecodeError:
                    pass

    except urllib.error.URLError as e:
        print(f"\nERROR: Cannot connect to Ollama at {base_url}", file=sys.stderr)
        print(f"  Reason: {e.reason}", file=sys.stderr)
        print(f"  Make sure Ollama is running: ollama serve", file=sys.stderr)
        sys.exit(1)
    except TimeoutError:
        print(f"\nERROR: Request timed out after {timeout}s", file=sys.stderr)
        print(f"  The model may still be loading. Try again or increase --timeout", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)

    sys.stdout.write("\n")
    sys.stdout.flush()
    return "".join(full_response), stats


def call_ollama_sync(base_url, model, messages, temperature, num_ctx, timeout):
    """Call Ollama chat API without streaming."""
    url = f"{base_url}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
        },
    }

    if num_ctx:
        payload["options"]["num_ctx"] = num_ctx

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        print(f"ERROR: Cannot connect to Ollama at {base_url}", file=sys.stderr)
        print(f"  Reason: {e.reason}", file=sys.stderr)
        sys.exit(1)
    except TimeoutError:
        print(f"ERROR: Request timed out after {timeout}s", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    content = result.get("message", {}).get("content", "")
    stats = {
        "total_duration_ms": result.get("total_duration", 0) / 1e6,
        "prompt_eval_count": result.get("prompt_eval_count", 0),
        "eval_count": result.get("eval_count", 0),
        "eval_duration_ms": result.get("eval_duration", 0) / 1e6,
        "tokens_per_second": (
            result.get("eval_count", 0) / (result.get("eval_duration", 1) / 1e9)
            if result.get("eval_duration", 0) > 0
            else 0
        ),
    }
    return content, stats


# ---------------------------------------------------------------------------
# Gemini provider
# ---------------------------------------------------------------------------

def get_gemini_api_key():
    """Get the Gemini API key from environment. Exits with clear error if missing."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        print("ERROR: GEMINI_API_KEY environment variable is not set.", file=sys.stderr)
        print("  Get an API key from: https://aistudio.google.com/apikey", file=sys.stderr)
        sys.exit(1)
    return key


def build_gemini_request(system_prompt, user_content, temperature, max_tokens):
    """Build a Gemini API request body."""
    request_body = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_content}],
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }

    if system_prompt:
        request_body["system_instruction"] = {
            "parts": [{"text": system_prompt}]
        }

    return request_body


def _handle_gemini_http_error(e, base_url):
    """Handle HTTP errors from the Gemini API with user-friendly messages."""
    if hasattr(e, "code"):
        if e.code == 401 or e.code == 403:
            print("ERROR: Invalid or unauthorized GEMINI_API_KEY.", file=sys.stderr)
            print("  Verify your key at: https://aistudio.google.com/apikey", file=sys.stderr)
        elif e.code == 429:
            print("ERROR: Gemini API rate limit exceeded.", file=sys.stderr)
            print("  Free tier limits: 10 RPM (Flash), 5 RPM (Pro).", file=sys.stderr)
            print("  Wait a moment and try again, or switch to Ollama.", file=sys.stderr)
        elif e.code == 404:
            print(f"ERROR: Gemini model not found.", file=sys.stderr)
            print("  Available models: gemini-2.5-flash, gemini-2.5-pro,", file=sys.stderr)
            print("  gemini-3-flash-preview, gemini-3-pro-preview", file=sys.stderr)
        elif e.code == 400:
            try:
                body = e.read().decode("utf-8")
                err_data = json.loads(body)
                msg = err_data.get("error", {}).get("message", "Bad request")
                print(f"ERROR: Gemini API error: {msg}", file=sys.stderr)
            except Exception:
                print(f"ERROR: Gemini API returned HTTP {e.code}", file=sys.stderr)
        else:
            print(f"ERROR: Gemini API returned HTTP {e.code}", file=sys.stderr)
    else:
        print(f"ERROR: Cannot connect to Gemini API at {base_url}", file=sys.stderr)
        print(f"  Reason: {e}", file=sys.stderr)
    sys.exit(1)


def call_gemini_streaming(base_url, model, api_key, request_body, timeout):
    """Call Gemini API with SSE streaming, printing tokens as they arrive."""
    url = f"{base_url}/models/{model}:streamGenerateContent?alt=sse"

    data = json.dumps(request_body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )

    full_response = []
    stats = {}

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            buffer = ""
            for raw_line in resp:
                line = raw_line.decode("utf-8")
                buffer += line

                # Process complete lines (SSE events are newline-delimited)
                while "\n" in buffer:
                    event_line, buffer = buffer.split("\n", 1)
                    event_line = event_line.strip()

                    if not event_line.startswith("data: "):
                        continue

                    json_str = event_line[len("data: "):]
                    if not json_str:
                        continue

                    try:
                        chunk = json.loads(json_str)
                    except json.JSONDecodeError:
                        continue

                    candidates = chunk.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts and "text" in parts[0]:
                            token = parts[0]["text"]
                            sys.stdout.write(token)
                            sys.stdout.flush()
                            full_response.append(token)

                    # Extract stats from final chunk
                    usage = chunk.get("usageMetadata", {})
                    if usage:
                        prompt_tokens = usage.get("promptTokenCount", 0)
                        gen_tokens = usage.get("candidatesTokenCount", 0)
                        stats = {
                            "total_duration_ms": 0,  # Gemini doesn't report this
                            "prompt_eval_count": prompt_tokens,
                            "eval_count": gen_tokens,
                            "eval_duration_ms": 0,
                            "tokens_per_second": 0,
                        }

    except urllib.error.HTTPError as e:
        _handle_gemini_http_error(e, base_url)
    except urllib.error.URLError as e:
        print(f"ERROR: Cannot connect to Gemini API", file=sys.stderr)
        print(f"  Reason: {e.reason}", file=sys.stderr)
        print(f"  Check your internet connection.", file=sys.stderr)
        sys.exit(1)
    except TimeoutError:
        print(f"ERROR: Request timed out after {timeout}s", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)

    sys.stdout.write("\n")
    sys.stdout.flush()
    return "".join(full_response), stats


def call_gemini_sync(base_url, model, api_key, request_body, timeout):
    """Call Gemini API without streaming."""
    url = f"{base_url}/models/{model}:generateContent"

    data = json.dumps(request_body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        _handle_gemini_http_error(e, base_url)
    except urllib.error.URLError as e:
        print(f"ERROR: Cannot connect to Gemini API", file=sys.stderr)
        print(f"  Reason: {e.reason}", file=sys.stderr)
        sys.exit(1)
    except TimeoutError:
        print(f"ERROR: Request timed out after {timeout}s", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # Extract response text
    candidates = result.get("candidates", [])
    if not candidates:
        print("ERROR: Gemini returned no candidates.", file=sys.stderr)
        sys.exit(1)

    parts = candidates[0].get("content", {}).get("parts", [])
    content = parts[0].get("text", "") if parts else ""

    # Extract stats
    usage = result.get("usageMetadata", {})
    prompt_tokens = usage.get("promptTokenCount", 0)
    gen_tokens = usage.get("candidatesTokenCount", 0)
    stats = {
        "total_duration_ms": 0,
        "prompt_eval_count": prompt_tokens,
        "eval_count": gen_tokens,
        "eval_duration_ms": 0,
        "tokens_per_second": 0,
    }

    return content, stats


# ---------------------------------------------------------------------------
# Unified model execution
# ---------------------------------------------------------------------------

def run_model(provider, model, system_prompt, user_content, temperature,
              base_url=None, num_ctx=None, max_tokens=8192, timeout=120,
              stream=True):
    """
    Run a single model and return (response, stats).

    When stream=True, tokens are printed to stdout as they arrive.
    When stream=False, the full response is returned silently.
    """
    if provider == "gemini":
        api_key = get_gemini_api_key()
        gemini_url = base_url or GEMINI_DEFAULT_URL
        request_body = build_gemini_request(system_prompt, user_content, temperature, max_tokens)

        if stream:
            return call_gemini_streaming(gemini_url, model, api_key, request_body, timeout)
        else:
            return call_gemini_sync(gemini_url, model, api_key, request_body, timeout)

    else:  # ollama
        ollama_url = base_url or OLLAMA_DEFAULT_URL
        messages = build_ollama_messages(system_prompt, user_content)

        if stream:
            return call_ollama_streaming(ollama_url, model, messages, temperature, num_ctx, timeout)
        else:
            return call_ollama_sync(ollama_url, model, messages, temperature, num_ctx, timeout)


# ---------------------------------------------------------------------------
# Assessment
# ---------------------------------------------------------------------------

def _do_assessment(response, stats, provider, model, task, prompt, rating_override, tm):
    """Log an assessment and update trust scores."""
    if rating_override is not None:
        rating = rating_override
    else:
        rating = tm.auto_assess(response, task, prompt)

    entry = {
        "model": model,
        "provider": provider,
        "task_type": task,
        "rating": rating,
        "prompt_summary": prompt[:100],
        "response_length": len(response),
        "eval_tokens": stats.get("eval_count", 0),
        "duration_ms": stats.get("total_duration_ms", 0),
        "tokens_per_second": stats.get("tokens_per_second", 0),
        "comparison_id": None,
        "notes": "auto-assessed" if rating_override is None else "manual-rating",
    }
    tm.log_assessment(entry)
    tm.update_trust_score(model, provider, task, rating)

    print(f"\n--- Assessment ---", file=sys.stderr)
    print(f"  Model: {model} [{provider}]", file=sys.stderr)
    print(f"  Task: {task}", file=sys.stderr)
    print(f"  Rating: {rating}/5 ({'auto' if rating_override is None else 'manual'})",
          file=sys.stderr)
    return rating


# ---------------------------------------------------------------------------
# Comparison mode
# ---------------------------------------------------------------------------

def _run_comparison(args, system_prompt, temperature, user_content):
    """
    Run two models on the same task and output structured comparison.

    Strategy:
    - Mixed providers (ollama + gemini): run in parallel via subprocess
    - Same provider (ollama + ollama): run sequentially to avoid VRAM contention
    """
    from datetime import datetime, timezone

    comparison_id = f"cmp-{int(time.time())}-{random.randint(1000, 9999)}"
    provider_a = args.provider
    provider_b = args.provider_b or args.provider
    model_a = args.model
    model_b = args.model_b
    url_a = args.url if args.url != OLLAMA_DEFAULT_URL else (
        OLLAMA_DEFAULT_URL if provider_a == "ollama" else GEMINI_DEFAULT_URL
    )
    url_b = args.url if args.url != OLLAMA_DEFAULT_URL else (
        OLLAMA_DEFAULT_URL if provider_b == "ollama" else GEMINI_DEFAULT_URL
    )

    print(f"Comparison mode: {model_a} [{provider_a}] vs {model_b} [{provider_b}]",
          file=sys.stderr)

    mixed_providers = (provider_a != provider_b)

    if mixed_providers:
        # Run in parallel: launch model B as subprocess, run model A in main process
        print(f"  Running in parallel (mixed providers)...", file=sys.stderr)

        # Create temp file for subprocess output
        scripts_dir = os.path.dirname(os.path.abspath(__file__))
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".json", prefix="cmp_b_")
        os.close(tmp_fd)

        # Build subprocess command for model B
        cmd = [
            sys.executable, os.path.join(scripts_dir, "agent_runner.py"),
            "--provider", provider_b,
            "--model", model_b,
            "--task", args.task,
            "--prompt", args.prompt,
            "--no-stream",
            "--json-out", tmp_path,
            "--timeout", str(args.timeout),
            "--temperature", str(temperature),
        ]
        if args.system:
            cmd.extend(["--system", args.system])
        if args.file:
            cmd.extend(["--file", args.file])
        if args.max_tokens:
            cmd.extend(["--max-tokens", str(args.max_tokens)])

        # Start subprocess for model B
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=os.environ.copy(),
        )

        # Run model A in main process (no streaming for clean capture)
        start_a = time.time()
        response_a, stats_a = run_model(
            provider_a, model_a, system_prompt, user_content, temperature,
            base_url=url_a, num_ctx=args.num_ctx, max_tokens=args.max_tokens,
            timeout=args.timeout, stream=False,
        )
        duration_a = (time.time() - start_a) * 1000

        # Wait for subprocess
        proc.wait()

        # Read subprocess result
        response_b = ""
        stats_b = {}
        duration_b = 0
        try:
            with open(tmp_path, "r") as f:
                result_b = json.load(f)
            response_b = result_b.get("response", "")
            stats_b = result_b.get("stats", {})
        except (json.JSONDecodeError, OSError) as e:
            stderr_out = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
            print(f"  Warning: Model B failed: {e}", file=sys.stderr)
            if stderr_out:
                print(f"  Stderr: {stderr_out[:500]}", file=sys.stderr)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    else:
        # Same provider: run sequentially
        print(f"  Running sequentially (same provider)...", file=sys.stderr)

        # Model A
        print(f"  Running model A: {model_a}...", file=sys.stderr)
        start_a = time.time()
        response_a, stats_a = run_model(
            provider_a, model_a, system_prompt, user_content, temperature,
            base_url=url_a, num_ctx=args.num_ctx, max_tokens=args.max_tokens,
            timeout=args.timeout, stream=False,
        )
        duration_a = (time.time() - start_a) * 1000

        # Model B
        print(f"  Running model B: {model_b}...", file=sys.stderr)
        start_b = time.time()
        response_b, stats_b = run_model(
            provider_b, model_b, system_prompt, user_content, temperature,
            base_url=url_b, num_ctx=args.num_ctx, max_tokens=args.max_tokens,
            timeout=args.timeout, stream=False,
        )
        duration_b = (time.time() - start_b) * 1000

    # Build comparison result
    comparison = {
        "comparison_id": comparison_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task_type": args.task,
        "prompt": args.prompt,
        "model_a": {
            "model": model_a,
            "provider": provider_a,
            "response": response_a,
            "stats": stats_a,
            "wall_time_ms": round(duration_a, 1),
        },
        "model_b": {
            "model": model_b,
            "provider": provider_b,
            "response": response_b,
            "stats": stats_b,
            "wall_time_ms": round(duration_b, 1) if duration_b else None,
        },
    }

    # Save comparison file
    tm = _get_trust_manager()
    if tm:
        filepath = tm.save_comparison(comparison_id, comparison)
        if filepath:
            print(f"  Comparison saved: {filepath}", file=sys.stderr)

    # Print structured output for Claude to evaluate
    print(json.dumps(comparison, indent=2))

    return comparison


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run a task through a local (Ollama) or cloud (Gemini) LLM sub-agent"
    )
    parser.add_argument("--provider", default="ollama", choices=["ollama", "gemini"],
                        help="LLM provider (default: ollama)")
    parser.add_argument("--model", required=True,
                        help="Model name (e.g. qwen2.5-coder:32b for Ollama, gemini-2.5-flash for Gemini)")
    parser.add_argument("--prompt", required=True, help="The task prompt")
    parser.add_argument("--system", help="Custom system prompt (overrides task default)")
    parser.add_argument("--task", default="general",
                        choices=list(TASK_SYSTEM_PROMPTS.keys()),
                        help="Task type for default system prompt and temperature")
    parser.add_argument("--file", help="Path to a file to include as context")
    parser.add_argument("--temperature", type=float, help="Override temperature (0.0-2.0)")
    parser.add_argument("--max-tokens", type=int, default=8192,
                        help="Max output tokens, Gemini only (default: 8192)")
    parser.add_argument("--num-ctx", type=int, help="Context window size override (Ollama only)")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Request timeout in seconds (default: 120)")
    parser.add_argument("--no-stream", action="store_true",
                        help="Disable streaming (wait for full response)")
    parser.add_argument("--url", default=OLLAMA_DEFAULT_URL, help="Provider API URL override")
    parser.add_argument("--stats", action="store_true", help="Print generation stats to stderr")
    parser.add_argument("--json-out", help="Save response and stats to JSON file")

    # Assessment flags
    parser.add_argument("--assess", action="store_true",
                        help="Assess the response and update trust scores")
    parser.add_argument("--rating", type=int, choices=[1, 2, 3, 4, 5],
                        help="Explicit rating (1-5). Overrides auto-assessment. Implies --assess")

    # Comparison flags
    parser.add_argument("--compare", action="store_true",
                        help="Compare two models on the same task")
    parser.add_argument("--model-b", help="Second model name for comparison")
    parser.add_argument("--provider-b", choices=["ollama", "gemini"],
                        help="Provider for second model (default: same as --provider)")

    args = parser.parse_args()

    # --rating implies --assess
    if args.rating is not None:
        args.assess = True

    # Validate comparison mode
    if args.compare and not args.model_b:
        parser.error("--compare requires --model-b")

    # Resolve system prompt
    system_prompt = args.system or TASK_SYSTEM_PROMPTS.get(args.task, TASK_SYSTEM_PROMPTS["general"])

    # Resolve temperature
    temperature = (args.temperature if args.temperature is not None
                   else TASK_TEMPERATURES.get(args.task, 0.5))

    # Read file if provided
    file_content = None
    if args.file:
        file_content = read_file_content(args.file)

    # Build user content (shared across providers)
    user_content = build_user_content(args.prompt, file_content, args.file)

    # --- Comparison mode ---
    if args.compare:
        _run_comparison(args, system_prompt, temperature, user_content)
        return

    # --- Normal single-model mode ---
    base_url = args.url
    if args.provider == "gemini" and args.url == OLLAMA_DEFAULT_URL:
        base_url = GEMINI_DEFAULT_URL

    response, stats = run_model(
        args.provider, args.model, system_prompt, user_content, temperature,
        base_url=base_url, num_ctx=args.num_ctx, max_tokens=args.max_tokens,
        timeout=args.timeout, stream=not args.no_stream,
    )

    # Print response if non-streaming (streaming already printed to stdout)
    if args.no_stream:
        print(response)

    # Print stats if requested
    if args.stats and stats:
        print(f"\n--- Stats ({args.provider}) ---", file=sys.stderr)
        print(f"  Prompt tokens: {stats['prompt_eval_count']}", file=sys.stderr)
        print(f"  Generated tokens: {stats['eval_count']}", file=sys.stderr)
        if stats.get("total_duration_ms"):
            print(f"  Total time: {stats['total_duration_ms']:.0f}ms", file=sys.stderr)
        if stats.get("tokens_per_second"):
            print(f"  Speed: {stats['tokens_per_second']:.1f} tok/s", file=sys.stderr)

    # Save JSON output if requested
    if args.json_out:
        output = {
            "provider": args.provider,
            "model": args.model,
            "task": args.task,
            "prompt": args.prompt,
            "response": response,
            "stats": stats,
        }
        with open(args.json_out, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Saved to {args.json_out}", file=sys.stderr)

    # Assess if requested
    if args.assess:
        tm = _get_trust_manager()
        if tm:
            _do_assessment(response, stats, args.provider, args.model,
                           args.task, args.prompt, args.rating, tm)
        else:
            print("Warning: trust_manager not available, skipping assessment.",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
