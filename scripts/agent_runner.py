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
"""

import argparse
import json
import sys
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
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run a dev task through a local (Ollama) or cloud (Gemini) LLM"
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

    args = parser.parse_args()

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

    # Dispatch to the right provider
    if args.provider == "gemini":
        api_key = get_gemini_api_key()
        # Use Gemini URL unless user explicitly overrode --url
        gemini_url = args.url if args.url != OLLAMA_DEFAULT_URL else GEMINI_DEFAULT_URL
        request_body = build_gemini_request(system_prompt, user_content, temperature, args.max_tokens)

        if args.no_stream:
            response, stats = call_gemini_sync(
                gemini_url, args.model, api_key, request_body, args.timeout
            )
            print(response)
        else:
            response, stats = call_gemini_streaming(
                gemini_url, args.model, api_key, request_body, args.timeout
            )

    else:  # ollama
        messages = build_ollama_messages(system_prompt, user_content)

        if args.no_stream:
            response, stats = call_ollama_sync(
                args.url, args.model, messages, temperature, args.num_ctx, args.timeout
            )
            print(response)
        else:
            response, stats = call_ollama_streaming(
                args.url, args.model, messages, temperature, args.num_ctx, args.timeout
            )

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


if __name__ == "__main__":
    main()
