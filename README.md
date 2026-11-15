# Data Assistant

[![tests](https://github.com/Av-code80/data-assistant/actions/workflows/tests.yml/badge.svg)](https://github.com/Av-code80/data-assistant/actions/workflows/tests.yml)

**Ask a business question in plain French, get the verified numbers from the company
database in seconds, with the SQL that produced them.**

Built for non-technical teams (marketing, sales, finance, logistics) who today wait
2 to 5 days for the data team to answer an ad hoc question. Not a chatbot: every number
comes from a query run on the company's data, never from the model's memory.

> Work in progress: an AI engineering project, built the way it would be delivered to a
> client (configuration, tests, evals, structured logs, cost tracking). See the roadmap.

The company is fictional: ModaNova, a European fashion e-commerce site, with ~3 years of
realistic synthetic data (8,000 customers, 30,000 orders, 50,000 order items).

## Example

`POST /questions/ask` with `{"question": "Top 3 des pays par chiffre d'affaires en 2025 ?"}`

```json
{
  "status": "answered",
  "sql": "SELECT country, ROUND(SUM(revenue), 2) AS revenue FROM sales WHERE order_date BETWEEN '2025-01-01' AND '2025-12-31' GROUP BY country ORDER BY revenue DESC LIMIT 3",
  "columns": ["country", "revenue"],
  "rows": [
    ["France", 458866.72],
    ["Germany", 178950.73],
    ["Spain", 146753.47]
  ],
  "chart": { "type": "bar", "x": "country", "y": ["revenue"] },
  "steps": [
    { "tool": "run_sql", "ok": true },
    { "tool": "submit_answer", "ok": true }
  ],
  "total_cost_usd": 0.043
}
```

Response shortened: it also carries the interpretation, warnings, cited sources and
per-stage token usage.

## How it works

```
question ──> 1. Retrieve ──> 2. Interpret ──> 3. SQL agent ──> 4. Explain
             company glossary  structured plan    write, run, fix    French summary
             (hybrid RAG)      checked by server  the query          of the result
```

| Step         | What happens                                                                                                                | Why it matters                                                                                                            |
| ------------ | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| 1. Retrieve  | Hybrid search (multilingual embeddings + BM25, merged with reciprocal rank fusion) over the company glossary and data rules | "Ventes" means what _this_ company says it means                                                                          |
| 2. Interpret | Claude turns the question into a typed plan (metric, dimensions, filters, period) with structured output                    | The server, not the model, decides if the question is answerable, needs clarification or is out of scope                  |
| 3. SQL agent | A ReAct loop written by hand: the model explores with `run_sql`, then submits with `submit_answer`                          | It reads its own errors and fixes its query; the server re-runs the final query, so rows never come from the model's text |
| 4. Explain   | The result table is summarised in French, streamed (Server-Sent Events)                                                     | Users read an answer, not a table                                                                                         |

## Reliability, safety, cost

| Concern                            | How it is handled                                                                                                                                                              |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| The model writes a dangerous query | Defense in depth: DuckDB's parser accepts one `SELECT` only, the database file is opened read-only, file and network access are disabled and locked, every query has a timeout |
| The model reads data it should not | The agent connects to a separate view layer (sales, customers, products marts): source tables and sensitive columns (product cost) do not exist for it                         |
| Wrong business rules               | Rules (cancelled items excluded, returned items bring no revenue) are applied once in the view layer, and revenue is reconciled with the source at build time                  |
| Periods outside the data           | Clamped or rejected by the server, with a warning shown to the user                                                                                                            |
| Runaway agent                      | At most 6 LLM calls per question                                                                                                                                               |
| Cost                               | Every call logs tokens, cost, latency, model and prompt version as JSON; each answer returns its total cost                                                                    |
| Model refusal or outage            | Server-side fallback model; errors mapped to HTTP 422 / 502 / 503                                                                                                              |
| Prompt injection through data      | Query results and tables are passed as data, with explicit instructions to ignore any text that looks like an instruction                                                      |

## Measured quality

| Benchmark                                                     | Result                                                                                  | Run                                |
| ------------------------------------------------------------- | --------------------------------------------------------------------------------------- | ---------------------------------- |
| Retrieval, 20 real questions                                  | hit@3 100 %, hit@1 70 %, MRR 0.85                                                       | `uv run python -m evals.retrieval` |
| End to end, 10 questions with human-checked reference queries | execution accuracy 100 %, 1.6 tool calls, 0.038 USD and 6.9 s per question              | `uv run python -m evals.agent`     |
| Tests                                                         | 108 unit and API tests with a fake LLM (free), plus integration tests on the real model | `uv run pytest`                    |

The end-to-end eval compares result rows, not SQL text (two different queries can both
be right), and exits with code 1 below 80 %, so a CI job can block a bad prompt change.
It already paid off: it caught a missing business dimension (product department), fixed
in the catalog and prompt v5.

## Stack

Python 3.12 · FastAPI · Pydantic · Claude API (Anthropic SDK: structured output, tool use,
streaming) · DuckDB · fastembed (local multilingual embeddings) · Pandas · pytest · uv

## Run it

```bash
uv sync
cp .env.example .env                       # then set ANTHROPIC_API_KEY
uv run python scripts/generate_data.py     # ~3 s, writes data/shop.duckdb (source tables)
uv run python scripts/build_analytics.py   # writes data/analytics.duckdb (view layer)
bash scripts/check_setup.sh                # checks tools, key, permissions, data
uv run uvicorn app.main:app --reload       # first start downloads the embedding model (~220 MB)
```

Interactive API docs: http://127.0.0.1:8000/docs. From the terminal:
`uv run python scripts/ask.py "Quel est le CA par catégorie au T3 2026 ?"`

```bash
uv run pytest --ignore=tests/test_integration.py   # free: no API call
uv run pytest -v -rs                               # also real Claude calls when a key is set
```

## Routes

| Method | Path                      | Purpose                                                                         |
| ------ | ------------------------- | ------------------------------------------------------------------------------- |
| GET    | `/health`                 | health probe                                                                    |
| GET    | `/knowledge/search?q=...` | what the RAG retrieves for a question                                           |
| POST   | `/questions/interpret`    | question → plan, status, warnings, sources, cost                                |
| POST   | `/questions/ask`          | question → interpretation, SQL, rows, chart suggestion, agent trace, total cost |

| POST | `/answers/stream` | question + result table → explanation (Server-Sent Events) |

## Layout

```
app/
  agent.py           SQL agent: hand-written ReAct loop over run_sql / submit_answer
  analytics.py       view layer: the marts the SQL agent may read, business rules applied
  charts.py          suggests a chart (line, bar or none) for a result table, by rules
  catalog.py         business metrics and dimensions (single source of truth)
  config.py          settings from the environment (.env), validated at startup
  knowledge.py       loads knowledge/*.md and splits it into sections and passages
  retrieval.py       hybrid search: embeddings + BM25, merged with reciprocal rank fusion
  sql_tool.py        runs the agent's SQL safely: SELECT only, read-only, no files, timeout
  prompts.py         versioned prompts
  schemas.py         Pydantic contracts: HTTP payloads and the shape the LLM must return
  llm.py             service layer: business rules + the only code that talks to the LLM
  logging_config.py  JSON logs, one line per event
  main.py            HTTP routes
knowledge/           business glossary, table documentation, data rules (French content)
scripts/             data generation, view layer build, setup check, terminal client
evals/               retrieval and end-to-end benchmarks
tests/               unit, API, retrieval quality gate, real integration tests
.github/workflows/   CI: free tests on every push and pull request
```

## Roadmap

- [x] Hybrid RAG over the business glossary, with a retrieval benchmark
- [x] Question interpretation with structured output and server-side checks
- [x] Secure SQL tool and text-to-SQL agent, with an end-to-end eval
- [x] Chart suggestion
- [ ] Explanation streamed directly from `/questions/ask`
- [ ] Lint (ruff), coverage, cost report
- [x] GitHub Actions: the free test suite runs on every push
- [ ] Docker image, lint and build in the pipeline
- [ ] API key / JWT authentication
- [ ] Web frontend (React)
- [ ] Deployment on Cloud Run, monitoring

## Platform note

`onnxruntime` is pinned to 1.19.2 on Intel Macs only: newer versions ship no wheel for
macOS 12 on Intel. Other platforms get a recent version.

## License

MIT
