---
name: other-sub-agents
description: >
  Delegate development sub-tasks to local or cloud LLMs. Use this skill whenever the user
  asks to offload work to local models, mentions Ollama or Gemini, wants to use local AI or
  cloud AI for coding tasks, or when a development task could benefit from LLM sub-agents
  (code generation, debugging, code review, documentation, architecture analysis, test writing).
  Also use this when the user says things like "use my local models", "run this through Ollama",
  "delegate to local AI", "use Gemini", "use a cloud model", or asks about what models are available.
---

# Other Sub Agents

This skill delegates tasks to other LLM sub-agents running either locally (via Ollama) or
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

### Step 0: Seed Initial Trust Scores (First Run Only)

On first use, seed trust scores from published benchmark data:

```bash
python3 ~/.claude/skills/ollama-agents/ollama-agents/scripts/seed_trust.py
```

This populates `~/.claude/skills/ollama-agents/ollama-agents/data/trust_scores.json` with baseline scores derived from HumanEval,
SWE-bench, MBPP, BigCodeBench, MMLU, and other published benchmarks for ~35 common
models. Seed scores have count=1, so actual assessments quickly override them.

### Step 1: Discover Available Models

Before delegating, check what's available across both providers:

```bash
python3 ~/.claude/skills/ollama-agents/ollama-agents/scripts/discover_models.py
```

This checks Ollama at `localhost:11434` and (if `GEMINI_API_KEY` is set) the Gemini API.
It returns a unified report of all models with their capabilities and routing recommendations.

Options:
- `--provider all|ollama|gemini` — which providers to check (default: `all`)
- `--json` — structured JSON output
- `--task code|review|docs|architecture|test|debug|general` — recommend best model for a task

Options:
- `--seed-new` — automatically seed trust scores for any newly discovered models

If a provider is unavailable (Ollama not running, no Gemini key), discovery continues
gracefully with the other provider.

**When a new model appears** (one not in `~/.claude/skills/ollama-agents/ollama-agents/data/trust_scores.json`):

**Option A: Automatic seeding (recommended)**

Run discovery with `--seed-new` to auto-research and seed new models:

```bash
python3 ~/.claude/skills/ollama-agents/ollama-agents/scripts/discover_models.py --seed-new
```

Or use the dedicated research script directly:

```bash
# Auto-discover and seed all new models
python3 ~/.claude/skills/ollama-agents/ollama-agents/scripts/research_new_model.py --auto-discover

# Seed a specific new model
python3 ~/.claude/skills/ollama-agents/ollama-agents/scripts/research_new_model.py --model "new-model:14b" --provider ollama

# Preview what would be seeded (dry run)
python3 ~/.claude/skills/ollama-agents/ollama-agents/scripts/research_new_model.py --auto-discover --dry-run
```

The auto-research script estimates scores by:
1. Looking for exact matches in the benchmark seed database
2. Interpolating from similar models (same family, type, and size)
3. Falling back to heuristic estimation based on model type, size, and family
4. Scores start with count=1, so real assessments quickly override them

**Option B: Manual research (for higher confidence)**

1. Search the web for its published benchmarks (HumanEval, MBPP, SWE-bench, MMLU, etc.)
2. Use those benchmarks to estimate initial trust scores per task type (0-1 scale)
3. Log seed scores via `trust_manager.py` with count=1 so real assessments override quickly
4. If no benchmarks exist, start with the static classification and use comparison mode
   to build trust data against a known model

### Step 1.5: Check Trust Scores

After discovery, check historical trust scores to inform model selection:

```bash
python3 ~/.claude/skills/ollama-agents/ollama-agents/scripts/trust_manager.py --action rankings --task code --json
```

This shows all models ranked by their observed trust score for a given task type.
Trust scores are built from your assessments after each sub-agent interaction.

For blended recommendations (static capabilities + trust data):

```bash
python3 ~/.claude/skills/ollama-agents/ollama-agents/scripts/discover_models.py --task code --with-trust --json
```

**Trust score thresholds:**
- **> 0.7 (70%)**: Use confidently — model has proven itself on this task type
- **0.4–0.7**: Use but review output carefully
- **< 0.4**: Consider an alternative or use comparison mode
- **No data**: Use static recommendation, then assess to build trust data

### Step 2: Assess Model Fitness for the Task

Read `~/.claude/skills/ollama-agents/ollama-agents/references/model-profiles.md` for detailed capability profiles.

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
python3 ~/.claude/skills/ollama-agents/ollama-agents/scripts/agent_runner.py \
  --model "qwen2.5-coder:32b" \
  --task code \
  --prompt "Write a function that validates email addresses using regex." \
  --temperature 0.3
```

**Gemini example (cloud):**
```bash
python3 ~/.claude/skills/ollama-agents/ollama-agents/scripts/agent_runner.py \
  --provider gemini \
  --model "gemini-2.5-flash" \
  --task code \
  --prompt "Write a function that validates email addresses using regex." \
  --temperature 0.3
```

**With file context (works with both providers):**
```bash
python3 ~/.claude/skills/ollama-agents/ollama-agents/scripts/agent_runner.py \
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

### Step 3.5: Assess the Sub-Agent's Output

After reviewing the sub-agent response, **always** log an assessment to build trust data:

```bash
python3 ~/.claude/skills/ollama-agents/ollama-agents/scripts/trust_manager.py --action log \
  --model "qwen2.5-coder:32b" --provider ollama \
  --task code --rating 4 \
  --prompt-summary "Write email validation function"
```

Or combine execution with assessment in one command:

```bash
python3 ~/.claude/skills/ollama-agents/ollama-agents/scripts/agent_runner.py \
  --model "qwen2.5-coder:32b" --task code \
  --prompt "Write a function that validates email addresses." \
  --assess --rating 4
```

**Rating scale (1–5):**
- **5 (Excellent)**: Directly usable, no or trivial edits needed
- **4 (Good)**: Mostly correct, minor fixes needed
- **3 (Adequate)**: Partially useful, needs significant editing
- **2 (Poor)**: Mostly wrong, major misunderstandings
- **1 (Failed)**: Unusable — wrong language, ignored instructions, errored out

Omit `--rating` to use auto-assessment (heuristic based on response length, code
blocks, and structure). Always prefer explicit ratings when you can judge quality.

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

## Comparison Mode (Dual-Model)

When unsure which model is better for a task, or to build trust data faster,
run two models on the same task simultaneously:

```bash
python3 ~/.claude/skills/ollama-agents/ollama-agents/scripts/agent_runner.py \
  --model "qwen2.5-coder:32b" --provider ollama \
  --model-b "gemini-2.5-flash" --provider-b gemini \
  --task code \
  --prompt "Write a function that validates email addresses." \
  --compare --no-stream
```

This outputs both responses as structured JSON. Review both outputs, pick the
best result, and **assess both models**:

```bash
# Rate the winner
python3 ~/.claude/skills/ollama-agents/ollama-agents/scripts/trust_manager.py --action log \
  --model "qwen2.5-coder:32b" --provider ollama \
  --task code --rating 4 --prompt-summary "email validation"

# Rate the other
python3 ~/.claude/skills/ollama-agents/ollama-agents/scripts/trust_manager.py --action log \
  --model "gemini-2.5-flash" --provider gemini \
  --task code --rating 2 --prompt-summary "email validation"
```

**When to use comparison mode:**
- A new model has been added and you need trust data quickly
- Two models seem similarly capable for a task type
- The user wants to see alternative approaches
- Trust scores are close (within 10%) for the top two candidates

**Execution strategy:**
- Mixed providers (Ollama + Gemini): Runs in parallel — no contention
- Same provider (Ollama + Ollama): Runs sequentially — avoids VRAM swapping

Comparison results are saved in `~/.claude/skills/ollama-agents/ollama-agents/data/comparisons/` for reference.

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
