# Model Capability Profiles

Reference for understanding which models are good at what across providers.
Helps with routing decisions when multiple models are available.

## Trust-Based Selection

This reference provides **static** capability profiles. The skill also maintains
**dynamic trust scores** in `data/trust_scores.json` based on actual performance.

When both are available, the blended score (40% static + 60% trust) guides
model selection. Use `discover_models.py --task TYPE --with-trust` to see
blended rankings. Trust scores decay over time — a model unused for 30 days
will see its scores halved, ensuring stale data doesn't dominate.

To query trust data directly:
```bash
python3 scripts/trust_manager.py --action rankings --task code
python3 scripts/trust_manager.py --action query --model "qwen2.5-coder:32b"
```

## How to Use This Reference

Read this when you need to decide which model and provider to assign a task to.
The `discover_models.py` script does automated classification, but this document
provides deeper context for ambiguous cases.

---

## Known Models — Ollama (Local)

### Qwen2.5-Coder:32b
- **Provider**: Ollama (local)
- **Type**: Code specialist
- **Parameters**: 32B
- **Strengths**: Code generation, debugging, refactoring, test writing, code review
- **Best for**: All code-centric tasks — writing functions, fixing bugs, generating tests,
  reviewing pull requests, refactoring modules
- **Temperature**: 0.2-0.4 for precise code, 0.5+ for creative solutions
- **Note**: One of the best open-source code models at this size.

### GPT-OSS:120b
- **Provider**: Ollama (local)
- **Type**: Large reasoning model
- **Parameters**: 120B
- **Strengths**: Architecture analysis, planning, documentation, complex reasoning
- **Best for**: Architecture decisions, system design, comprehensive documentation
- **Temperature**: 0.4-0.7 depending on task creativity needed
- **Note**: Large model — allow extra timeout (300s+) for initial load.

---

## Known Models — Gemini (Cloud)

### Gemini 2.5 Flash
- **Provider**: Google Gemini API
- **Type**: General-purpose cloud model
- **Quality**: Good
- **Context**: Up to 1M tokens
- **Rate limit**: 10 RPM (free tier)
- **Strengths**: Code generation, debugging, review, documentation, reasoning
- **Best for**: Tasks needing large context, when no local model available, free cloud access
- **Temperature**: 0.2-0.4 for code, 0.5+ for creative tasks
- **Note**: Best balance of capability and free access. Use `--provider gemini --model gemini-2.5-flash`

### Gemini 2.5 Pro
- **Provider**: Google Gemini API
- **Type**: Cloud reasoning model
- **Quality**: High
- **Context**: Up to 1M tokens
- **Rate limit**: 5 RPM (free tier)
- **Strengths**: Complex reasoning, architecture, planning, documentation, code
- **Best for**: Architecture analysis, complex multi-step reasoning, comprehensive documentation
- **Temperature**: 0.4-0.7
- **Note**: Strongest stable Gemini model. Lower RPM. Use `--provider gemini --model gemini-2.5-pro`

### Gemini 2.5 Flash Lite
- **Provider**: Google Gemini API
- **Type**: Lightweight cloud model
- **Quality**: Moderate
- **Context**: Up to 1M tokens
- **Rate limit**: 15 RPM (free tier)
- **Best for**: Simple tasks, quick code snippets, formatting, text transforms
- **Note**: Fastest and cheapest option. Good for high-volume simple tasks.

### Gemini 3 Flash/Pro Preview
- **Provider**: Google Gemini API
- **Type**: Next-generation preview models
- **Quality**: Good (Flash) / High (Pro)
- **Context**: Up to 1M tokens
- **Note**: Preview models — capabilities may change. Try for latest quality.

---

## Ollama Model Families

### Qwen / Qwen2.5 / Qwen4 Family
- **Code variants** (qwen-coder): Excellent code generation, completion, and review.
  Strong at Python, JavaScript, TypeScript, Go, Rust, Java.
- **General variants**: Good all-rounders with multilingual support and math.

### Llama 3 Family
- Strong instruction following and reasoning
- Good at structured output and formatting
- **Best use**: Architecture, planning, documentation, code review at 70B+

### DeepSeek / DeepSeek-Coder
- Exceptional code generation, strong math and logic
- **Best use**: Code generation, debugging, algorithmic problems

### Mistral / Mixtral
- Efficient — punches above weight class
- **Best use**: Speed + quality balance

### GOT-OSS / GPT-OSS
- Large reasoning-focused model
- Strong at analysis, planning, architectural thinking
- **Best use**: Architecture analysis, planning, complex reasoning

---

## Provider Comparison

| Aspect | Ollama (Local) | Gemini (Cloud) |
|--------|---------------|----------------|
| **Privacy** | Full — data stays on machine | Data sent to Google |
| **Cost** | Free (your hardware) | Free tier available |
| **Context window** | 2K-32K typical | Up to 1M tokens |
| **Rate limits** | None (hardware bound) | 5-15 RPM (free) |
| **Latency** | Depends on GPU | Network + inference |
| **Offline** | Yes | No |
| **Sensitive code** | Safe | Do not send secrets |
| **Best for batch** | Yes (no rate limits) | Limited by RPM |

---

## Routing Decision Matrix

| Task | Ollama Minimum | Preferred Type | Temp | Provider Preference |
|------|---------------|----------------|------|---------------------|
| Code generation | 14B+ code model | Code specialist | 0.2-0.4 | Ollama if 25B+, else Gemini Flash |
| Unit tests | 14B+ code model | Code specialist | 0.3 | Ollama if available |
| Code review | 25B+ any | Code or large general | 0.3-0.5 | Ollama if 25B+, else Gemini |
| Debugging | 14B+ code model | Code specialist | 0.1-0.3 | Ollama if available |
| Refactoring | 25B+ code model | Code specialist | 0.2-0.4 | Ollama if 25B+ |
| Documentation | 25B+ general | General purpose | 0.4-0.6 | Either — Gemini for large files |
| Architecture | 65B+ general | Reasoning | 0.5-0.7 | Gemini Pro if no 65B+ local |
| Planning | 65B+ general | Reasoning | 0.5-0.7 | Gemini Pro if no 65B+ local |
| Large file review | N/A | Any with big context | 0.3-0.5 | Gemini (1M context) |
| Boilerplate | 7B+ | Code specialist | 0.2 | Ollama (fast, no rate limit) |
| Batch tasks | Any | Any | Varies | Ollama (no RPM limits) |
