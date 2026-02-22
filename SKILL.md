---
name: dev-agents
description: >
  Delegate development sub-tasks to local or cloud LLMs. Use this skill whenever the user
  asks to offload work to local models, mentions Ollama or Gemini, wants to use local AI or
  cloud AI for coding tasks, or when a development task could benefit from LLM sub-agents
  (code generation, debugging, code review, documentation, architecture analysis, test writing).
  Also use this when the user says things like "use my local models", "run this through Ollama",
  "delegate to local AI", "use Gemini", "use a cloud model", or asks about what models are available.
---

# Development Sub-Agents

This skill delegates development sub-tasks to LLMs running either locally (via Ollama) or
in the cloud (via Google Gemini API). It discovers what's available, assesses each model's
strengths, and routes tasks to the most appropriate model and provider.

## Why This Exists

Claude acts as the **orchestrator**: it decides what to delegate, picks the right model,
crafts the prompt, and integrates the results. Some tasks — generating boilerplate, writing
tests, reviewing code snippets, drafting docs — can be handled by other models, saving API
costs and leveraging the user's local GPU or free cloud tiers.

Two providers are supported:
- **Ollama** — local models, fully private, no rate limits, works offline
- **Gemini** — Google's cloud models, up to 1M token context, free tier available

## How It Works

### Step 1: Discover Available Models

Before delegating, check what's available across both providers:

```bash
python3 scripts/discover_models.py
```

This checks Ollama at `localhost:11434` and (if `GEMINI_API_KEY` is set) the Gemini API.
It returns a unified report of all models with their capabilities and routing recommendations.

Options:
- `--provider all|ollama|gemini` — which providers to check (default: `all`)
- `--json` — structured JSON output
- `--task code|review|docs|architecture|test|debug|general` — recommend best model for a task

If a provider is unavailable (Ollama not running, no Gemini key), discovery continues
gracefully with the other provider.

### Step 2: Assess Model Fitness for the Task

Read `references/model-profiles.md` for detailed capability profiles.

**Provider selection (Ollama first, Gemini as escalation):**

| Scenario | Provider | Why |
|----------|----------|-----|
| Private or sensitive code | Ollama | Data stays local |
| Large context needed (>32K tokens) | Gemini | 1M token window |
| No local GPU or weak hardware | Gemini | Cloud-hosted |
| Batch of similar tasks | Ollama | No rate limits |
| No suitable local model for the task | Gemini | Quality escalation |
| User explicitly requests a provider | Whichever they ask for | Respect preference |
| Neither provider available | Do it yourself in Claude | Fallback |

**Model type routing:**

**Code-focused models** (Ollama: names with `coder`, `codellama`, `starcoder`;
Gemini: any model) are best for:
- Writing functions, classes, or modules from specs
- Debugging and fixing code
- Code review, unit tests, refactoring, boilerplate

**Large reasoning models** (Ollama: 70B+ general models, `gpt-oss`;
Gemini: `gemini-2.5-pro`, `gemini-3-pro-preview`) are best for:
- Architecture and design discussions
- Documentation writing, planning, requirements analysis
- Explaining complex code

If no model is a strong match, don't delegate — do it yourself.

### Step 3: Delegate the Task

**Ollama example (local):**
```bash
python3 scripts/agent_runner.py \
  --model "qwen2.5-coder:32b" \
  --task code \
  --prompt "Write a function that validates email addresses using regex." \
  --temperature 0.3
```

**Gemini example (cloud):**
```bash
python3 scripts/agent_runner.py \
  --provider gemini \
  --model "gemini-2.5-flash" \
  --task code \
  --prompt "Write a function that validates email addresses using regex." \
  --temperature 0.3
```

**With file context (works with both providers):**
```bash
python3 scripts/agent_runner.py \
  --provider gemini \
  --model "gemini-2.5-pro" \
  --task review \
  --prompt "Review this code for bugs, performance, and style issues:" \
  --file "/path/to/code.py" \
  --temperature 0.4
```

Key flags:
- `--provider ollama|gemini` — which provider (default: `ollama`)
- `--model` — model name (e.g. `qwen2.5-coder:32b` or `gemini-2.5-flash`)
- `--task` — task type for default system prompt and temperature
- `--file` — include a file as context
- `--stats` — print generation stats to stderr
- `--timeout 300` — increase for large models
- `--max-tokens 8192` — max output tokens (Gemini only)
- `--no-stream` — wait for full response instead of streaming

### Step 4: Integrate Results

After getting a response from a sub-agent:

1. **Review the output** — always sanity-check before applying
2. **Apply selectively** — use the good parts, fix or discard the rest
3. **Tell the user** what was delegated, which model handled it, and which provider

## Prompt Engineering for Sub-Agent Models

These models respond best to:
- **Clear, specific instructions** — be explicit about what you want
- **Structured output requests** — "Return only the function, no explanation"
- **Concrete examples** — show the format you expect
- **Limited scope** — one task per request, not multi-step workflows
- **System prompts** — the `--task` flag sets good defaults

Avoid:
- Vague instructions ("make this better")
- Multi-part tasks in a single prompt
- Assuming the model knows about the broader codebase
- Sending more context than needed (especially to Ollama models with limited windows)

## Error Handling

**Ollama errors:**
- **Not running**: Inform user, suggest `ollama serve`
- **Model not found**: Show available models, suggest `ollama pull <model>`
- **Timeout**: Default 120s. For large models use `--timeout 300`
- **Out of memory**: Suggest a smaller or more quantized variant

**Gemini errors:**
- **No API key**: Inform user, direct to `https://aistudio.google.com/apikey`
- **Invalid key (401/403)**: Ask user to verify key
- **Rate limit (429)**: Free tier is 10 RPM (Flash), 5 RPM (Pro). Wait or switch to Ollama
- **Model not found (404)**: Check model name spelling

## Important Constraints

- **Never send sensitive data** (API keys, credentials, secrets) to any sub-agent model —
  Ollama logs to disk, Gemini sends data to Google
- **Never log or expose `GEMINI_API_KEY`** — read from env var only, never print it
- **Don't over-delegate** — if crafting the prompt takes longer than doing the task, just do it
- **Always review output** — treat sub-agent output as a draft, never as final
- **Respect context limits** — Ollama models typically have 2K-32K; Gemini supports up to 1M
- **One Ollama model at a time** — switching models incurs a VRAM load penalty. Batch similar
  tasks for the same model. Gemini has no such constraint.
- **Gemini rate limits** — 5-15 RPM on free tier. Don't rapid-fire requests.
- **Gemini sends data to Google** — do not send code containing secrets, credentials, or
  proprietary information that shouldn't leave the machine
