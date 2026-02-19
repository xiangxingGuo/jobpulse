# JobPulse

> **Local-first LLM job intelligence pipeline with fallback orchestration, structured validation, and personalized skill-gap reporting.**

JobPulse is an end-to-end LLM system that transforms raw job postings into structured intelligence and actionable reports. It combines scraping, structured extraction, validation, model fine-tuning, orchestration, and reliability engineering into a single reproducible workflow.

------

# 🚀 What JobPulse Does

JobPulse:

- Scrapes job postings (Handshake + extensible connectors)
- Extracts structured requirements using LLMs
- Fine-tunes compact models using LoRA
- Validates structured outputs with QC gates
- Falls back across providers (Local → API)
- Generates personalized skill-gap reports
- Exposes functionality via MCP tools
- Orchestrates workflows with LangGraph

------

# 🏗 Architecture Overview

```
Scrape → Clean → Extract → Validate → Fallback (if needed) → Report → Persist Artifacts
```

### Design Principles

- **Local-first inference**
- **Provider abstraction (OpenAI-compatible APIs)**
- **Strict schema validation**
- **Structured artifact logging**
- **Traceable execution**
- **Composable orchestration**

------

# ✨ Key Capabilities

## 🔎 Job Data Pipeline

- Playwright-based scraping
- Multi-platform connector abstraction
- SQLite storage
- Raw → Structured → Report artifact flow

## 🤖 LLM Extraction

- Local HuggingFace inference (PyTorch)
- LoRA fine-tuned Qwen 0.5B
- OpenAI-compatible API backends
- Prompt versioning
- Robust JSON repair + schema validation

## 🔁 Reliability Engineering

- Local-first extraction strategy
- Automatic cloud fallback
- QC validation gate before report generation
- JSON repair heuristics:
  - Code fence stripping
  - Tail extraction
  - Bracket repair
  - Balanced truncation
- Step-level trace logging

## 🧠 Orchestration Layer

- LangGraph state machine
- Conditional routing
- MCP tool server (stdio transport)
- Short-lived MCP sessions for stability

------

# 📂 Project Structure

```
├── data/
│   └── artifacts/          # Structured outputs + trace logs
├── scripts/                # CLI entrypoints & experiments
├── src/
│   ├── connectors/         # Platform scraping adapters
│   ├── extractors/         # Local extraction logic
│   ├── llm/                # Prompt + provider abstraction
│   ├── mcp_server/         # Tool-based MCP server
│   ├── orch/               # LangGraph orchestration
│   ├── schemas/            # Pydantic structured models
│   ├── training/           # LoRA fine-tuning pipeline
│   ├── report.py           # Markdown report generation
│   └── db.py               # SQLite persistence
└── models/                 # LoRA adapters (generated)
```

Artifacts generated per job:

```
data/artifacts/<job_id>/
  structured.json
  qc.json
  report.md
  trace.json
```

`trace.json` records:

- Extraction path (local or API)
- QC result
- Fallback events
- Report generation metadata

------

# ⚙️ Environment Setup

## 1️⃣ Install Dependencies

Project uses `uv` for dependency management.

```
uv sync
```

## 2️⃣ Install Playwright Browsers (Required)

```
python -m playwright install --with-deps
```

⚠️ Without this step, scraping will fail.

## 3️⃣ Set API Keys (Optional if using local-only)

```
export OPENAI_API_KEY="your_key"
export NVIDIA_API_KEY="your_key"
```

------

# 🚀 Usage

## 1️⃣ Scrape Job Postings

```
uv run python scripts/run_pipeline.py --pages 1 --limit 10
```

Stored in:

```
data/db/jobs.db
```

------

## 2️⃣ Run Single Job via MCP Tool Chain

```
uv run python scripts/run_one_job_mcp.py \
  --job-id 10704289 \
  --provider openai
```

------

## 3️⃣ Run LangGraph Orchestration (Local-first + Fallback)

```
uv run python scripts/run_graph_one.py \
  --job-id 10704289 \
  --local-first \
  --local-model Qwen/Qwen2.5-3B-Instruct \
  --local-mode plain \
  --extract-provider openai \
  --report-provider openai
```

------

# 🧠 LoRA Fine-Tuning

## Dataset

Located in:

```
src/training/datasets/
  jd_struct_train.jsonl
  jd_struct_val.jsonl
```

Generated from teacher model outputs.

## Train

```
uv run python src/training/train_lora.py
```

## Output

LoRA adapter saved to:

```
models/qwen2.5-0.5b-jd-lora/
```

------

# 🔄 Provider Support

JobPulse supports OpenAI-compatible backends:

| Provider | Example Model        |
| -------- | -------------------- |
| OpenAI   | gpt-4o-mini          |
| NVIDIA   | moonshotai/kimi-k2.5 |
| Local HF | Qwen + LoRA          |

Example NVIDIA endpoint:

```
https://integrate.api.nvidia.com/v1
```

------

# 🛠 MCP Tool Server

Start manually:

```
python -m src.mcp_server.server
```

Available tools:

- `fetch_jd`
- `extract_local`
- `extract_api`
- `qc_validate`
- `generate_report_api`

------

# 📈 Reliability Strategy

## Extraction Flow

```
local → qc_fail → api → qc_pass → report
```

## JSON Hardening

- Code fence stripping
- Tail extraction
- Bracket repair
- Balanced truncation
- Final brace-matching fallback

This significantly reduces malformed LLM output failure rates.

------

# 🧩 Tech Stack

- Python
- PyTorch
- HuggingFace Transformers
- PEFT (LoRA / QLoRA)
- LangGraph
- MCP
- Playwright
- SQLite

------

# 🎯 Engineering Highlights

This project demonstrates:

- Production-style LLM system design
- Model fine-tuning workflow
- Provider abstraction layer
- Structured validation patterns
- Fallback orchestration
- Artifact-based observability
- Reliability engineering for LLM output

------

# 🔮 Future Improvements

- Batch graph runner
- Structured extraction evaluation dashboard
- Resume ingestion + automated skill-gap comparison
- Async scraping
- Dockerized deployment
- Monitoring & metrics layer