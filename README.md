# Multi-Agent Research System

A production-grade multi-agent research system using LangGraph with a Supervisor pattern. Three specialized agents (Web Scraper, Data Analyst, Report Writer) collaborate to produce structured research reports from any topic.

## Architecture

```
                    +------------+
                    | Supervisor |
                    | (Sonnet 4) |
                    +-----+------+
                          |
            +-------------+-------------+
            |             |             |
       +----v----+  +-----v-----+  +----v----+
       | Scraper |  |  Analyst  |  |  Writer |
       | (Tavily)|  | (Haiku 4) |  |(Haiku 4)|
       +---------+  +-----------+  +---------+

  START -> Supervisor -> {Scraper, Analyst, Writer} -> Supervisor -> END
```

| Agent | Role | LLM |
|-------|------|-----|
| **Supervisor** | Plans research tasks, dispatches agents, aggregates results | Claude Sonnet 4 |
| **Web Scraper** | Searches Tavily + scrapes URLs for content | N/A (tool-based) |
| **Data Analyst** | Extracts structured insights from scraped data | Claude Haiku 4.5 |
| **Report Writer** | Synthesizes insights into a scored report | Claude Haiku 4.5 |

## Quick Start

### One-command setup (Mac, Linux, and Windows)

```bash
python start.py
```

This interactive script will:
1. Copy `.env.example` to `.env`
2. Prompt you for required API keys (input is hidden)
3. Install all dependencies
4. Start the server on `http://localhost:8005`

You can also run individual steps:

```bash
python start.py --setup   # Only configure .env and install deps
python start.py --run     # Only start the server (skip setup)
```

### Manual setup

```bash
git clone https://github.com/eightlabs08/multi-agent-research.git
cd multi-agent-research

# 1. Configure environment
cp .env.example .env
# Edit .env and add your API keys (see Environment Variables below)

# 2. Install dependencies
pip install -e ".[dev]"

# 3. Run tests (optional - verifies everything works)
make test

# 4. Start the server
make run
```

## Environment Variables

Create a `.env` file at the project root (or use `python start.py` to do it interactively):

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Your [Anthropic API key](https://console.anthropic.com/) for Claude LLMs |
| `TAVILY_API_KEY` | Yes | Your [Tavily API key](https://tavily.com/) for web search |
| `OPENAI_API_KEY` | No | Optional fallback LLM |
| `PORT` | No | Server port (default: `8005`) |
| `LOG_LEVEL` | No | Logging level (default: `info`) |

## API Usage

### Health Check

```bash
curl http://localhost:8005/health
```

Response:
```json
{"status": "healthy"}
```

### Run a Research Query

```bash
curl -X POST http://localhost:8005/research \
  -H "Content-Type: application/json" \
  -d '{"query": "AI agents market analysis 2026"}'
```

Response:
```json
{
  "status": "done",
  "iterations": 4,
  "report": {
    "title": "AI Agents Market Analysis 2026",
    "executive_summary": "A comprehensive analysis of the AI agents market...",
    "sections": [
      {
        "heading": "Market Overview",
        "content": "The AI agents market is projected to...",
        "key_points": [
          "Market size exceeds $X billion",
          "Year-over-year growth of X%"
        ]
      }
    ],
    "insights": [
      {
        "claim": "Enterprise adoption of AI agents grew 40% in 2025",
        "evidence": "According to industry reports...",
        "confidence": 0.85,
        "sources": ["https://example.com/report"]
      }
    ],
    "sources": ["https://example.com/report"],
    "quality_score": 0.87
  },
  "errors": []
}
```

### Example Queries

```bash
# Technology analysis
curl -X POST http://localhost:8005/research \
  -H "Content-Type: application/json" \
  -d '{"query": "comparison of LangGraph vs CrewAI vs AutoGen for multi-agent systems"}'

# Market research
curl -X POST http://localhost:8005/research \
  -H "Content-Type: application/json" \
  -d '{"query": "enterprise SaaS pricing trends 2026"}'

# Competitive analysis
curl -X POST http://localhost:8005/research \
  -H "Content-Type: application/json" \
  -d '{"query": "Claude vs GPT-4 vs Gemini for code generation benchmarks"}'
```

### Interactive API Docs

Once the server is running, visit:

- Swagger UI: [http://localhost:8005/docs](http://localhost:8005/docs)
- ReDoc: [http://localhost:8005/redoc](http://localhost:8005/redoc)

## Project Structure

```
multi-agent-research/
├── src/
│   ├── agents/
│   │   ├── supervisor.py        # Supervisor agent (plans, routes, aggregates)
│   │   ├── scraper.py           # Web Scraper agent (Tavily search + URL scraping)
│   │   ├── analyst.py           # Data Analyst agent (insight extraction)
│   │   └── writer.py            # Report Writer agent (report synthesis + scoring)
│   ├── config/
│   │   └── env.py               # Environment config with validation
│   ├── guardrails/
│   │   ├── circuit_breaker.py   # Circuit breaker with fallback support
│   │   ├── retry.py             # Exponential backoff with jitter
│   │   └── output_validator.py  # Pydantic validation + quality scoring
│   ├── tools/
│   │   ├── schemas.py           # Pydantic schemas for tool I/O
│   │   ├── web_search.py        # Tavily web search wrapper
│   │   └── web_scrape.py        # HTML scraping + text extraction
│   ├── graph.py                 # LangGraph StateGraph pipeline definition
│   ├── state.py                 # Shared state schema (TypedDict + Pydantic models)
│   └── server.py                # FastAPI server with /health and /research endpoints
├── tests/                       # 100% test coverage (115 tests)
├── start.py                     # Cross-platform setup & start script
├── Makefile                     # Dev commands (test, lint, format, run)
├── pyproject.toml               # Project config, deps, tool settings
└── .env.example                 # Environment variable template
```

## How It Works

1. **You send a query** via `POST /research` (e.g., "AI agents market analysis 2026")
2. **Supervisor** receives the query, creates a research plan, and decides which agent to dispatch first
3. **Scraper** searches the web via Tavily API and scrapes the top results for full content
4. **Supervisor** reviews the scraped data and dispatches the Analyst
5. **Analyst** extracts structured insights (claims, evidence, confidence scores, sources)
6. **Supervisor** reviews insights and dispatches the Writer
7. **Writer** synthesizes everything into a structured report with quality scoring
8. **Supervisor** reviews the report. If quality is sufficient, marks as done. Otherwise, loops back for more research
9. **Response** returns the full report, iteration count, and any errors encountered

The supervisor has a max iteration guard (default: 10) to prevent infinite loops.

## Guardrails

Multi-agent systems can fail in unpredictable ways — LLMs return malformed output, APIs go down, agents get stuck in loops. This project ships with five production guardrails out of the box.

### 1. Retry with Exponential Backoff + Jitter

**File:** `src/guardrails/retry.py`

When a transient error occurs (API timeout, rate limit, network blip), the system retries automatically with increasing delays to avoid hammering a recovering service.

```
Attempt 1 → fail → wait ~1s
Attempt 2 → fail → wait ~2s
Attempt 3 → fail → wait ~4s
Attempt 4 → fail → RetryExhaustedError
```

- **Max retries:** 3 (configurable via `RetryConfig`)
- **Base delay:** 1 second, doubles each attempt
- **Max delay cap:** 60 seconds
- **Jitter:** Adds randomness (0.5x-1.5x) to prevent thundering herd when multiple agents retry simultaneously

```python
from src.guardrails.retry import retry_with_backoff, RetryConfig

result = await retry_with_backoff(
    my_async_function,
    config=RetryConfig(max_retries=5, base_delay=0.5),
    retry_on=(TimeoutError, ConnectionError),
    args=(arg1, arg2),
)
```

### 2. Circuit Breaker

**File:** `src/guardrails/circuit_breaker.py`

Prevents cascading failures by stopping calls to a service that is consistently failing. Uses a three-state model:

```
CLOSED (normal) → failures hit threshold → OPEN (reject all calls)
                                              ↓
                              timeout elapses (60s default)
                                              ↓
                                         HALF_OPEN (allow 1 test call)
                                           ↓              ↓
                                      success → CLOSED   fail → OPEN
```

- **Failure threshold:** 5 consecutive failures before opening (configurable)
- **Reset timeout:** 60 seconds before trying again
- **Fallback support:** Optionally provide a fallback function when the circuit is open

```python
from src.guardrails.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

breaker = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3, reset_timeout=30.0))

result = await breaker.call(
    risky_api_call,
    fallback=cached_response,  # called when circuit is open
)
```

### 3. Output Validation + LLM Response Sanitization

**File:** `src/guardrails/output_validator.py`

Every LLM response passes through two layers of validation before entering the pipeline:

1. **Markdown fence stripping** — LLMs often wrap JSON in ` ```json ``` ` code blocks. The `strip_markdown_fences()` function removes these before parsing.
2. **Pydantic schema validation** — The raw JSON is validated against strict Pydantic models (`SupervisorDecision`, `Insight`, `Report`). Invalid responses are caught and logged as recoverable errors rather than crashing the pipeline.

```python
from src.guardrails.output_validator import validate_output, strip_markdown_fences
from src.state import Report

raw_llm_output = strip_markdown_fences(response.content)
report = validate_output(raw_llm_output, Report)  # raises ValidationError if invalid
```

### 4. Report Quality Scoring

**File:** `src/guardrails/output_validator.py`

Every generated report receives an automated quality score (0.0-1.0) based on weighted criteria:

| Criterion | Weight | Target |
|-----------|--------|--------|
| Title length | 10% | 20+ characters |
| Executive summary | 20% | 100+ characters |
| Number of sections | 25% | 3+ sections |
| Number of insights | 20% | 3+ insights |
| Number of sources | 15% | 3+ sources |
| Key points coverage | 10% | Every section has key points |

The supervisor can use this score to decide whether to accept the report or send it back to the writer for improvement.

### 5. Max Iteration Guard

**File:** `src/graph.py`

The supervisor loop is capped at **10 iterations** (configurable via `max_iterations` parameter). This prevents:

- Infinite loops when the LLM keeps requesting more research
- Runaway API costs from unbounded agent cycles
- Stuck pipelines that never reach a "done" state

When the limit is reached, the pipeline returns whatever results it has, along with any errors encountered.

### 6. Content Truncation

**File:** `src/state.py`

Scraped web content is automatically truncated to **10,000 characters** via a Pydantic field validator. This prevents:

- Token budget overflows when passing content to LLMs
- Memory issues from storing very large pages
- Wasted API costs on content that exceeds useful context

## Development

```bash
make install      # Install with dev dependencies
make test         # Run tests with coverage (100% required)
make lint         # Run ruff linter
make type-check   # Run mypy strict type checking
make format       # Auto-format code with ruff
make clean        # Remove caches and build artifacts
```

### Requirements

- Python 3.12+
- An [Anthropic API key](https://console.anthropic.com/)
- A [Tavily API key](https://tavily.com/)

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Orchestration | LangGraph StateGraph |
| LLM (Supervisor) | Claude Sonnet 4 |
| LLM (Workers) | Claude Haiku 4.5 |
| Web Search | Tavily API |
| Schema Validation | Pydantic v2 |
| API Server | FastAPI + Uvicorn |
| HTTP Client | httpx (async) |
| Testing | pytest + pytest-asyncio (100% coverage) |
| Linting | ruff |
| Type Checking | mypy (strict) |

## License

MIT
