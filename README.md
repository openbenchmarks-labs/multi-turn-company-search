# OpenBenchmarks Multi Turn Company Search Benchmark

Open head-to-head benchmark runner for search APIs used by company-research agents.

Published and maintained by **[OpenBenchmarks Labs](https://openbenchmarks.com)**.

**Live benchmark:** https://openbenchmarks.com/multi-turn-company-search

This repo is the open runner + judge behind that page. It contains the fixed agent,
all measured vendor adapters, the local run-artifact contract, and the deterministic
offline scorer. It does not require a hosted database: the runner reads one frozen
dataset JSON file and writes one immutable JSON artifact per trial under `runs/`.

For every question, the agent must find the complete set of companies satisfying
three or four constraints at once—for example geography, founding period, investor,
accelerator, or funding history. The model, prompt, budgets, questions, output
schema, and three independent trials stay fixed while only the search configuration
changes.

Two boards are reported separately:

- **Search only.** The agent can issue focused searches and read normalized titles,
  URLs, and snippets, but cannot fetch page text.
- **Search + fetch.** The same agent can fetch an exact URL returned by search,
  using the provider's native extraction endpoint where available and a bounded
  local HTTP/Playwright fetcher otherwise.

The boards remain separate because search-result quality and page-retrieval quality
are different products.

## What counts as a correct company set

One rule, applied identically to every provider:

1. **Every constraint must hold.** A company is correct only when it satisfies all
   parts of the question. Partial matches are false positives.
2. **The set must be complete.** A correct company omitted by the agent is a false
   negative. Returning no unsupported companies is not enough if valid companies
   were missed.
3. **Identity is canonical.** Predictions are matched to the frozen answer set by
   canonical domain, canonical name, or an explicitly declared alias.

That produces four headline metrics: **precision**, **recall**, **F1**, and
**exact-set accuracy**. Exact-set accuracy is the strictest measure: the returned
set must contain every gold company and no extras.

## Endpoints

- **Live benchmark UI** — https://openbenchmarks.com/multi-turn-company-search
- **GitHub repository** — https://github.com/openbenchmarks-labs/multi-turn-company-search
- **Markdown agent docs** — https://openbenchmarks.com/llms.txt
- **OpenAPI 3.1 spec** — https://openbenchmarks.com/openapi.json
- **MCP server discovery** — https://openbenchmarks.com/.well-known/mcp.json

## Configurations measured

The benchmark compares the following concrete endpoints and configurations. A row
is a configuration—not merely a vendor name—because two modes from the same vendor
can behave like different products.

Each provider receives three independent trials per question. For every quality
metric, the judge first averages each trial over the complete question set, then
reports the **mean ± sample standard deviation** of those three trial-level values.
Standard deviation is expressed in percentage points and is descriptive, not a
confidence interval.

| Provider | Configuration |
|---|---|
| Brave Search | Brave web search |
| Exa | Deep, Instant |
| Firecrawl | Search |
| Linkup | Fast, Standard |
| Parallel | Turbo, Fast, Basic, Advanced |
| Perplexity | Search (`search_context_size` low on search-only, high on search + fetch) |
| Seltz | Companies |
| Google SERP | RapidAPI |
| Tavily | Advanced |
| TinyFish | Search |
| You | Web search |

Parallel maps `site:` hosts into `source_policy.include_domains` and strips those
operators from the query; other vendors receive the agent query unchanged.

The live page publishes the current search-only and search + fetch rankings, ordered
by mean F1 and then mean precision. This repository contains the runner rather than
a database export; locally generated rankings are written by the offline judge.

## What's in this repo

| path | purpose |
|---|---|
| `scripts/run_company_research_benchmark.py` | Offline planner and explicitly paid multi-provider orchestrator. |
| `scripts/judge_company_research.py` | Builds leaderboard JSON from saved trial artifacts without network calls. |
| `scripts/company_research/agent.py` | Fixed OpenAI Responses API tool loop, prompt, budgets, and full search/fetch transcript. |
| `scripts/company_research/vendors.py` | Search and native-fetch adapters for every measured configuration. |
| `scripts/company_research/custom_fetch.py` | SSRF-hardened bounded HTTP fetch with a concurrency-limited Playwright fallback. |
| `scripts/company_research/dataset.py` | Frozen dataset schema validation and SHA-256 content receipts. |
| `scripts/company_research/parser.py` | Strict structured company-set parser and normalization. |
| `scripts/company_research/runner.py` | Concurrent execution, resume rules, and atomic local artifact writes. |
| `scripts/company_research/judge.py` | Canonical matching, completeness gates, metric calculation, and mean ± SD aggregation. |
| `scripts/company_research/test_*.py` | Offline runner, model-client, and judge contract tests. |
| `.env.example` | OpenAI and provider credential names; no secrets or database configuration. |

## Reproducing a run

Pick any `(mode, provider, question, trial)` tuple. Its artifact lives at:

```text
runs/<dataset>/<mode>/<provider>/<case>/trial-<n>.json
```

The file contains:

- **`case`** — the exact question and frozen canonical answer set used by the
  offline judge. Gold answers are never included in the model prompt.
- **`result.searches[]`** — each literal provider request, redacted response
  envelope, normalized hits, latency, attempts, and cost receipt.
- **`result.fetches[]`** — each requested URL plus the native or bounded-local
  fetch response when fetch is enabled.
- **`result.raw_model_response`** — every model turn and token-usage receipt.
- **`parsed_companies`** — the strict structured set extracted from the final
  response.
- **`dataset.content_sha256`** — the frozen input receipt that prevents results
  from different datasets being mixed silently.

Existing successful trials are skipped on rerun. Failed trials remain available for
inspection and are retried only with `--retry-errors`.

## Running the benchmark yourself

```bash
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m playwright install chromium
cp .env.example .env && $EDITOR .env

# Offline plan: validates the dataset and prints calls, budgets, and required keys.
PYTHONPATH=scripts python scripts/run_company_research_benchmark.py \
  --dataset /absolute/path/to/dataset.json \
  --research-mode search_only

# Paid execution requires both explicit confirmation flags.
PYTHONPATH=scripts python scripts/run_company_research_benchmark.py \
  --dataset /absolute/path/to/dataset.json \
  --research-mode search_only \
  --execute --confirm-paid

PYTHONPATH=scripts python scripts/run_company_research_benchmark.py \
  --dataset /absolute/path/to/dataset.json \
  --research-mode search_and_fetch \
  --execute --confirm-paid
```

The runner uses the standard OpenAI API through `OPENAI_API_KEY`. Fill only the
provider keys selected with `--vendors`. Useful controls include `--trials`,
`--query-offset`, `--query-limit`, `--vendor-concurrency`, `--trial-concurrency`,
`--max-turns`, `--max-searches`, and `--max-fetches`.

A valid input uses the `company-research-dataset-v1` contract:

```text
dataset.json
├── schema_version
├── dataset { slug, name, ... }
├── cases[]
│   ├── case_key, question, constraints[]
│   └── gold[] { entity_key, name, domain, aliases[] }
└── content_sha256
```

Run the offline contract suite:

```bash
PYTHONPATH=scripts python -m unittest discover \
  -s scripts/company_research -p 'test_*.py'
```

## Judging saved runs

Judging is deterministic and makes no provider or model calls:

```bash
PYTHONPATH=scripts python scripts/judge_company_research.py \
  --runs runs/multi-constraint-company-research-v1 \
  --output leaderboard.json \
  --trials 3
```

The judge rejects incomplete comparisons: every provider in a mode must contain the
same questions and every expected trial index. Failed or unparseable completed runs
score as empty predictions rather than disappearing from the denominator.

## Contributing a new provider

1. Add a `VendorSpec` entry to `scripts/company_research/vendors.py` with the
   endpoint, credential environment variable, public request configuration, fetch
   capability, and dated unit-cost assumption.
2. Add the provider's request branch and normalize its response to the shared
   `{url, title, snippet, metadata}` hit shape.
3. If the provider exposes native page extraction, add its fetch request and page
   parser. Otherwise opt into the bounded custom fetcher explicitly.
4. Add offline adapter fixtures/tests, run the contract suite, then run one isolated
   question with `--vendors <your-provider> --trials 1 --query-limit 1` before a full
   benchmark.

Do not add OpenAI-native web tools or provider-specific prompting. The provider
adapter is the comparison variable; the research agent must remain fixed.

## Methodology

- **Frozen questions and gold sets.** Each dataset artifact is schema-validated and
  content-addressed before execution. Every case contains one multi-constraint
  question and a canonical company set. Gold data is available only to the offline
  judge, never to the agent.
- **Fixed agent.** The default agent is `gpt-5.6-sol` at medium reasoning effort,
  with a maximum of eight model turns, fourteen searches, two searches per turn,
  and ten results per search. Search + fetch additionally permits fourteen fetches,
  at most two per turn.
- **Provider swap.** The question, system prompt, model, reasoning effort, budgets,
  output schema, and trial count are identical across rows. Only the provider-backed
  search and fetch implementations change.
- **No native model search.** The model receives only the selected provider's
  `web_search` tool and, in search + fetch, `web_fetch`. OpenAI-native web search and
  fetch are not enabled.
- **Set scoring.** True positives are canonical identities present in both predicted
  and gold sets. Extra predictions are false positives; missed gold identities are
  false negatives. Precision, recall, F1, and exact-set accuracy are calculated per
  agent run.
- **Three-trial aggregation.** Each metric is first averaged across all questions
  within trial 1, trial 2, and trial 3. The leaderboard reports the mean and sample
  standard deviation of those three values rather than selecting the best attempt.
- **Operational metrics.** Trial artifacts retain end-to-end latency, provider
  latency, model turns, token usage, cited-return rate, and estimated vendor/model
  cost. Costs use dated list-price assumptions rather than invoice reconciliation.
- **Concurrency.** Questions run sequentially within one provider so a systematic
  failure can stop further spend. Providers run concurrently, as do the three trials
  for a question.
- **Fetch safety.** Custom fetch rejects local/private network targets and bounds
  redirects, response size, navigation time, and browser concurrency. Search + fetch
  validates Chromium before paid calls begin.
- **Credential safety.** Authorization headers are redacted before artifacts are
  written. `.env`, `.env.local`, `runs/`, and local virtual environments are ignored
  by Git.
- **Known limitations.** This benchmark measures multi-constraint company discovery,
  not general web search, news, coding, academic research, or consumer navigation.
  Three trials expose obvious stochastic instability but are too few for a strong
  inferential uncertainty claim. Search + fetch measures the combined search and
  retrieval stack; use search only to isolate result quality. Local browser compute
  is excluded from reported cost.

No vendor sponsors or controls this benchmark.

## License

[MIT](LICENSE)
