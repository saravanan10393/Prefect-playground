# QnA Agent with Durable Execution — System Design

## Overview

A QnA agent built with **Pydantic AI** (agent framework) + **Prefect 3.0** (durable execution).
The agent answers user questions by routing through 4 tools, each wrapped as a Prefect task
with independent retry/timeout/caching policies.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Prefect Flow                             │
│                  (qa_agent_flow)                            │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              PrefectAgent(qa_agent)                    │  │
│  │                                                       │  │
│  │   User Question                                       │  │
│  │        │                                              │  │
│  │        ▼                                              │  │
│  │   ┌─────────┐                                         │  │
│  │   │  LLM    │  ◄── model_task (retries=3, backoff)    │  │
│  │   │ (Claude)│                                         │  │
│  │   └────┬────┘                                         │  │
│  │        │  decides which tool(s) to call                │  │
│  │        ▼                                              │  │
│  │   ┌─────────────────────────────────────────────┐     │  │
│  │   │            Tool Tasks                        │     │  │
│  │   │                                              │     │  │
│  │   │  ┌──────────────┐  ┌──────────────────────┐ │     │  │
│  │   │  │ search_local │  │ search_db            │ │     │  │
│  │   │  │ _files       │  │ (SQLite/Postgres)    │ │     │  │
│  │   │  │              │  │                      │ │     │  │
│  │   │  │ retries=2    │  │ retries=2            │ │     │  │
│  │   │  │ timeout=10s  │  │ timeout=15s          │ │     │  │
│  │   │  └──────────────┘  └──────────────────────┘ │     │  │
│  │   │                                              │     │  │
│  │   │  ┌──────────────┐  ┌──────────────────────┐ │     │  │
│  │   │  │ search_web   │  │ summarize_answer     │ │     │  │
│  │   │  │ (Tavily/     │  │ (final synthesis)    │ │     │  │
│  │   │  │  DuckDuckGo) │  │                      │ │     │  │
│  │   │  │              │  │ no retries           │ │     │  │
│  │   │  │ retries=3    │  │ (pure LLM, handled   │ │     │  │
│  │   │  │ timeout=30s  │  │  by model_task)      │ │     │  │
│  │   │  └──────────────┘  └──────────────────────┘ │     │  │
│  │   └─────────────────────────────────────────────┘     │  │
│  │        │                                              │  │
│  │        ▼                                              │  │
│  │   Structured Output (QnAResponse)                     │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  Prefect Dashboard: full observability of each step         │
└─────────────────────────────────────────────────────────────┘
```

---

## The 4 Tools

| # | Tool Name          | Purpose                                  | Data Source         | TaskConfig                          |
|---|--------------------|------------------------------------------|---------------------|-------------------------------------|
| 1 | `search_local_files` | Search local markdown/txt knowledge base | Local filesystem    | retries=2, timeout=10s              |
| 2 | `search_database`    | Query a SQLite FAQ/knowledge database    | SQLite DB           | retries=2, timeout=15s              |
| 3 | `search_web`         | Search the internet for answers          | Tavily API / httpx  | retries=3, timeout=30s, backoff     |
| 4 | `summarize_answer`   | Synthesize final answer from sources     | Pure LLM reasoning  | None (disable — handled by model)   |

### Tool Routing (handled by LLM)

The LLM decides which tools to call based on the question. The system prompt instructs it to:
1. Try `search_local_files` first for domain-specific knowledge
2. Try `search_database` for structured FAQ-type questions
3. Fall back to `search_web` for current events or unknown topics
4. Always call `summarize_answer` at the end to synthesize results

---

## Project Structure

```
Prefect-playground/
├── pyproject.toml              # Dependencies & project metadata
├── README.md                   # (optional)
├── knowledge/                  # Local knowledge base
│   ├── python_faq.md
│   └── prefect_guide.md
├── src/
│   └── qa_agent/
│       ├── __init__.py
│       ├── agent.py            # Pydantic AI agent + PrefectAgent wrapper
│       ├── tools.py            # 4 tool implementations
│       ├── models.py           # Pydantic models (deps, output schema)
│       ├── db.py               # SQLite setup & queries
│       └── config.py           # Settings (API keys, paths, DB URL)
├── scripts/
│   ├── seed_db.py              # Seed the SQLite FAQ database
│   └── run_agent.py            # Entry point — run the QnA flow
└── tests/
    ├── test_tools.py           # Unit tests for each tool
    └── test_agent.py           # Integration test for agent flow
```

---

## Key Components

### 1. Pydantic Models (`models.py`)

```python
from pydantic import BaseModel

class QnADeps:
    """Dependencies injected into tool RunContext."""
    knowledge_dir: str          # path to knowledge/ folder
    db_path: str                # path to SQLite DB
    web_search_api_key: str     # Tavily API key (optional)

class SourceReference(BaseModel):
    source_type: str            # "local_file" | "database" | "web"
    reference: str              # file path, DB row ID, or URL

class QnAResponse(BaseModel):
    answer: str
    sources: list[SourceReference]
    confidence: float           # 0.0 - 1.0
```

### 2. Agent Definition (`agent.py`)

```python
from pydantic_ai import Agent
from pydantic_ai.durable_exec.prefect import PrefectAgent, TaskConfig
from qa_agent.models import QnADeps, QnAResponse
from qa_agent.tools import search_local_files, search_database, search_web, summarize_answer

qa_agent = Agent(
    'anthropic:claude-sonnet-4-20250514',
    deps_type=QnADeps,
    output_type=QnAResponse,
    name='qa_agent',
    instructions="""You are a QnA assistant. To answer questions:
    1. First search local files for domain knowledge
    2. Then search the database for FAQ matches
    3. Use web search for current events or if other sources lack info
    4. Finally, synthesize all findings into a clear answer with sources
    """,
)

# Register tools on the agent (imported from tools.py)
# ... tools are decorated with @qa_agent.tool

# Wrap for durable execution
prefect_qa_agent = PrefectAgent(
    qa_agent,
    model_task_config=TaskConfig(
        retries=3,
        retry_delay_seconds=[1.0, 2.0, 4.0],
        timeout_seconds=60.0,
    ),
    tool_task_config=TaskConfig(retries=2),  # default for all tools
    tool_task_config_by_name={
        'search_local_files': TaskConfig(retries=2, timeout_seconds=10.0),
        'search_database':    TaskConfig(retries=2, timeout_seconds=15.0),
        'search_web':         TaskConfig(retries=3, timeout_seconds=30.0,
                                         retry_delay_seconds=[1.0, 2.0, 4.0]),
        'summarize_answer':   None,  # disable — pure LLM, no external I/O
    },
)
```

### 3. Tool Implementations (`tools.py`)

```python
from pydantic_ai import RunContext
from qa_agent.models import QnADeps

@qa_agent.tool
async def search_local_files(ctx: RunContext[QnADeps], query: str) -> str:
    """Search local markdown/text files for relevant content."""
    # Glob knowledge_dir for .md/.txt files
    # Simple keyword/fuzzy search over file contents
    # Return matching snippets with file paths
    ...

@qa_agent.tool
async def search_database(ctx: RunContext[QnADeps], query: str) -> str:
    """Search the SQLite FAQ database for matching answers."""
    # Use FTS5 full-text search on the FAQ table
    # Return top matching rows
    ...

@qa_agent.tool
async def search_web(ctx: RunContext[QnADeps], query: str) -> str:
    """Search the internet for answers using Tavily or httpx+DuckDuckGo."""
    # Call web search API
    # Return top results with URLs and snippets
    ...

@qa_agent.tool_plain
def summarize_answer(findings: list[str]) -> str:
    """Synthesize multiple source findings into a coherent answer."""
    # This is a pure formatting tool — joins and structures findings
    # The actual synthesis is done by the LLM in its response
    ...
```

### 4. Entry Point (`scripts/run_agent.py`)

```python
import asyncio
from qa_agent.agent import prefect_qa_agent
from qa_agent.models import QnADeps

async def main():
    deps = QnADeps(
        knowledge_dir="./knowledge",
        db_path="./qa.db",
        web_search_api_key="...",
    )
    result = await prefect_qa_agent.run(
        "What is durable execution in Prefect?",
        deps=deps,
    )
    print(result.output)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Durable Execution: What Happens on Failure?

```
Run 1 (fails at step 3):
  ✅ LLM call #1 → tool decision         (cached)
  ✅ search_local_files("prefect")        (cached)
  ❌ search_web("prefect durable") → TIMEOUT

Run 2 (retry — resumes from failure):
  ⏭️  LLM call #1 → skipped (cached result reused)
  ⏭️  search_local_files → skipped (cached result reused)
  ✅ search_web("prefect durable") → SUCCESS (retried)
  ✅ LLM call #2 → synthesize answer
  ✅ Output: QnAResponse(...)
```

This is the core value: **Run 2 doesn't re-pay for LLM call #1 or the local file search.**

---

## Dependencies

```toml
[project]
name = "qa-agent"
requires-python = ">=3.11"
dependencies = [
    "pydantic-ai[prefect]",     # Pydantic AI + Prefect durable execution
    "prefect>=3.0",             # Workflow orchestration
    "httpx",                    # HTTP client for web search
    "aiosqlite",                # Async SQLite driver
]

[project.optional-dependencies]
dev = ["pytest", "pytest-asyncio"]
```

---

## Implementation Order

1. **Project scaffolding** — pyproject.toml, directory structure, config
2. **Models** — QnADeps, QnAResponse, SourceReference
3. **Database** — SQLite schema + seed script + search_database tool
4. **Local file search** — knowledge/ folder + search_local_files tool
5. **Web search** — search_web tool (httpx-based, mock-friendly)
6. **Summarize** — summarize_answer tool
7. **Agent wiring** — Agent definition, PrefectAgent wrapper, system prompt
8. **Entry point** — run_agent.py script
9. **Tests** — unit tests for tools, integration test for full flow
