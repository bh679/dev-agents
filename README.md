# other-sub-agents

A [Claude Code](https://docs.anthropic.com/en/docs/claude-code) skill that delegates development sub-tasks to local and cloud LLMs.

Claude acts as the orchestrator -- it decides what to delegate, picks the right model, crafts the prompt, and integrates the results. Tasks like generating boilerplate, writing tests, reviewing code, and drafting docs can be handled by cheaper or local models, saving API costs and leveraging your hardware.

## Providers

| Provider | Privacy | Cost | Context | Offline |
|----------|---------|------|---------|---------|
| **Ollama** (local) | Full -- data stays on machine | Free (your GPU) | 2K-32K typical | Yes |
| **Gemini** (cloud) | Data sent to Google | Free tier available | Up to 1M tokens | No |

## Supported Tasks

- **Code generation** -- functions, classes, modules from specs
- **Debugging** -- find and fix bugs
- **Code review** -- style, performance, correctness
- **Test writing** -- unit tests, edge cases
- **Documentation** -- docstrings, READMEs, API docs
- **Architecture** -- design discussions, planning
- **Refactoring** -- restructure and clean up code

## Installation

### As a Claude Code Skill

Copy or symlink this directory into your Claude Code skills folder:

```bash
# Clone the repo
git clone https://github.com/bh679/other-sub-agents.git

# Symlink into Claude Code skills
ln -s "$(pwd)/other-sub-agents" ~/.claude/skills/other-sub-agents
```

### Prerequisites

**Ollama** (for local models):
```bash
# Install Ollama (macOS)
brew install ollama

# Start the server
ollama serve

# Pull a code model
ollama pull qwen2.5-coder:32b
```

**Gemini** (for cloud models):
```bash
# Get an API key from https://aistudio.google.com/apikey
export GEMINI_API_KEY="your-key-here"
```

## Usage

### Discover Available Models

```bash
python3 scripts/discover_models.py                    # All providers
python3 scripts/discover_models.py --provider ollama   # Local only
python3 scripts/discover_models.py --provider gemini   # Cloud only
python3 scripts/discover_models.py --task code --json  # Best model for code tasks
```

### Run a Sub-Agent Task

**Local (Ollama):**
```bash
python3 scripts/agent_runner.py \
  --model "qwen2.5-coder:32b" \
  --task code \
  --prompt "Write a Python function that validates email addresses."
```

**Cloud (Gemini):**
```bash
python3 scripts/agent_runner.py \
  --provider gemini \
  --model "gemini-2.5-flash" \
  --task code \
  --prompt "Write a Python function that validates email addresses."
```

**With file context:**
```bash
python3 scripts/agent_runner.py \
  --provider gemini \
  --model "gemini-2.5-pro" \
  --task review \
  --prompt "Review this code for bugs and performance issues:" \
  --file "/path/to/code.py"
```

### Key Flags

| Flag | Description |
|------|-------------|
| `--provider ollama\|gemini` | Which provider (default: `ollama`) |
| `--model` | Model name (e.g. `qwen2.5-coder:32b`, `gemini-2.5-flash`) |
| `--task` | Task type: `code`, `review`, `debug`, `test`, `docs`, `architecture`, `refactor`, `general` |
| `--file` | Include a file as context |
| `--temperature` | Override default temperature |
| `--stats` | Print generation stats to stderr |
| `--timeout` | Timeout in seconds (default: 120) |
| `--max-tokens` | Max output tokens (Gemini only) |
| `--no-stream` | Wait for full response instead of streaming |

## Routing Logic

The skill uses an Ollama-first strategy:

1. **Use Ollama** when: code is private/sensitive, batch processing needed, suitable local model exists
2. **Escalate to Gemini** when: large context needed (>32K tokens), no suitable local model, no local GPU
3. **Fall back to Claude** when: neither provider is available

See `references/model-profiles.md` for detailed routing decisions.

## File Structure

```
other-sub-agents/
  SKILL.md                       # Orchestration instructions for Claude Code
  scripts/
    agent_runner.py              # Core execution engine (both providers)
    discover_models.py           # Unified model discovery
  references/
    model-profiles.md            # Model capabilities and routing matrix
```

## License

MIT
