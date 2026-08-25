# Multi Turn Company Search Benchmark

Open runner and offline judge for the [OpenBenchmarks Multi Turn Company Search
Benchmark](https://openbenchmarks.com/multi-turn-company-search), maintained by
**[OpenBenchmarks Labs](https://openbenchmarks.com)**.

The benchmark asks one fixed research agent to find the complete set of
companies matching three- or four-part constraints. The question, model,
prompt, budgets, output schema, and three independent trials remain fixed while
the search provider changes.

There are two separate boards:

- **Search only** — the agent can search and read normalized titles, URLs, and
  snippets, but cannot fetch page text.
- **Search + fetch** — the same agent can fetch exact URLs returned by search,
  using provider-native extraction when available and a bounded local fetcher
  otherwise.

No hosted service is needed. The runner reads a frozen dataset JSON file and
writes immutable trial JSON files locally; the judge reads those files and
writes a leaderboard JSON file locally.

## Endpoints

- **Live benchmark** — https://openbenchmarks.com/multi-turn-company-search
- **GitHub** — https://github.com/openbenchmarks-labs/multi-turn-company-search
- **Markdown agent docs** — https://openbenchmarks.com/llms.txt
- **OpenAPI 3.1 spec** — https://openbenchmarks.com/openapi.json

## What is measured

Every returned company is matched to the frozen canonical answer set by domain,
canonical name, or a declared alias.

- **Precision** — true positives divided by all returned companies.
- **Recall** — true positives divided by all gold companies.
- **F1** — harmonic mean of precision and recall for one agent run.
- **Exact-set accuracy** — whether a run has no false positives or negatives.
- **Operational metrics** — end-to-end time, model turns, cited-return rate, and
  estimated provider/model cost.

For each quality metric, the judge first averages each trial over the complete
question set. It then reports the mean and sample standard deviation of the
three trial-level values. Standard deviation is expressed in percentage points;
it describes run-to-run variation and is not a confidence interval.

## Repository map

| Path | Purpose |
|---|---|
| `scripts/run_company_research_benchmark.py` | Offline planner and explicitly paid runner. |
| `scripts/judge_company_research.py` | Deterministic offline scorer and leaderboard builder. |
| `scripts/company_research/agent.py` | Fixed model/tool loop and complete search/fetch traces. |
| `scripts/company_research/vendors.py` | Measured search and native-fetch adapters. |
| `scripts/company_research/custom_fetch.py` | Bounded HTTP fetch with a Playwright fallback. |
| `scripts/company_research/dataset.py` | Frozen input artifact validation and hashing. |
| `scripts/company_research/parser.py` | Structured company-set parser and normalization. |
| `scripts/company_research/runner.py` | Concurrent execution, resume rules, and local artifact writer. |
| `scripts/company_research/judge.py` | Exact matching, completeness checks, and mean ± SD aggregation. |

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m playwright install chromium
cp .env.example .env.local
```

Fill only the credentials for the providers you plan to run. Never commit
`.env` or `.env.local`.

## Dataset input

Supply a frozen `company-research-dataset-v1` JSON artifact with `--dataset`.
The validator checks the schema and SHA-256 content receipt before any run is
planned or executed. Each case contains a question, stable case key, constraint
metadata, and canonical gold companies. Gold answers are used only by the
offline judge and are never included in the model prompt.

The dataset is intentionally an explicit input rather than hidden runner state:

```text
dataset.json
├── schema_version
├── dataset { slug, name, ... }
├── cases[]
│   ├── case_key, question, constraints[]
│   └── gold[] { entity_key, name, domain, aliases[] }
└── content_sha256
```

## Verify without network calls

```bash
PYTHONPATH=scripts .venv/bin/python -m unittest discover \
  -s scripts/company_research -p 'test_*.py'
```

The benchmark command is also offline by default. It validates the dataset and
prints its execution plan without model, vendor, or file writes:

```bash
PYTHONPATH=scripts .venv/bin/python scripts/run_company_research_benchmark.py \
  --dataset /absolute/path/to/dataset.json \
  --research-mode search_only
```

## Run the benchmark

Paid execution has a double guard; both flags are required:

```bash
PYTHONPATH=scripts .venv/bin/python scripts/run_company_research_benchmark.py \
  --dataset /absolute/path/to/dataset.json \
  --research-mode search_only \
  --execute --confirm-paid
```

Run search + fetch separately:

```bash
PYTHONPATH=scripts .venv/bin/python scripts/run_company_research_benchmark.py \
  --dataset /absolute/path/to/dataset.json \
  --research-mode search_and_fetch \
  --execute --confirm-paid
```

Useful controls include `--vendors`, `--trials`, `--query-offset`,
`--query-limit`, `--vendor-concurrency`, `--trial-concurrency`, `--max-turns`,
`--max-searches`, and `--max-fetches`.

Artifacts are written to:

```text
runs/<dataset>/<mode>/<vendor>/<case>/trial-<n>.json
```

Existing successful trials are skipped. Existing errors are preserved unless
`--retry-errors` is supplied. Each artifact retains the literal model response,
normalized provider results, redacted request/response envelopes, timings,
usage, cost receipts, parsed company set, question, and dataset content hash.

## Judge saved runs

Judging is deterministic and makes no network calls:

```bash
PYTHONPATH=scripts .venv/bin/python scripts/judge_company_research.py \
  --runs runs/multi-constraint-company-research-v1 \
  --output leaderboard.json \
  --trials 3
```

The judge rejects incomplete comparisons: every vendor in a mode must have the
same cases and all expected trial indexes. Failed or unparseable completed
trials score as empty predictions rather than disappearing from the average.
Rows are ranked by mean F1, then mean precision.

## Providers

- Brave Search
- Exa Deep and Exa Instant
- Firecrawl Search
- Linkup Fast and Linkup Standard
- Parallel Basic and Parallel Advanced
- Seltz Companies
- Google SERP through RapidAPI
- Tavily Advanced

A configuration is a row because endpoints or modes from the same vendor can
behave like different products.

## Methodology and safety

- The model receives only the selected vendor-backed `web_search` and optional
  `web_fetch` tools; OpenAI-native web tools are not enabled.
- The default agent is GPT-5.6 Sol at medium reasoning effort, with a maximum of
  eight turns, fourteen searches, and—only in search + fetch—fourteen fetches.
- The research model is called through the standard OpenAI API using
  `OPENAI_API_KEY`.
- Questions run sequentially per provider, providers run concurrently, and the
  three trials for a question run concurrently.
- If every trial for one question fails, that provider stops to avoid spending
  through the remaining question set.
- Search + fetch validates Chromium before paid calls when a selected provider
  uses the local fallback. The fetcher rejects private/local targets and bounds
  redirects, response sizes, navigation time, and browser concurrency.
- Authorization headers are redacted before artifacts are written.

## Known limitations

- The benchmark targets multi-constraint company discovery, not general web
  search, news, coding, academic research, or consumer navigation.
- Three trials reveal obvious instability but are too few for a strong
  inferential uncertainty claim.
- Search + fetch evaluates the combined search and retrieval stack; use the
  search-only board to isolate search-result quality.
- Costs use a dated list-price snapshot and exclude local browser compute.

No vendor sponsors or controls this benchmark.

## License

[MIT](LICENSE)
