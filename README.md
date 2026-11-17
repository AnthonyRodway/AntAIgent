# AntAIgent

A learner agent that runs completely locally. Point it at a folder of code or data, let it
build an understanding of what's there, then ask it questions to help you understand or
develop the system further. Nothing leaves your machine: models run in
[Ollama](https://ollama.com) and the index is a SQLite file on disk.

## Requirements

- [Ollama](https://ollama.com/download) with two models:
  ```
  ollama pull gpt-oss:20b
  ollama pull nomic-embed-text
  ```
  `gpt-oss:20b` needs about 13 GB of memory. It runs well on a 16 GB GPU. On CPU it works but is slow.
- Python 3.10+ and [uv](https://docs.astral.sh/uv/) (or plain `pip`)

## Install

```
uv venv
uv pip install -e ".[dev]"
.venv\Scripts\antai setup      # one-time: downloads code parsers (needs internet)
.venv\Scripts\antai doctor     # checks Ollama, models, GPU use and parsers
```

On macOS/Linux use `.venv/bin/antai`.

## Use

```
antai learn C:\path\to\project          # index a folder (re-run any time; only changes are processed)
antai ask "How does the reporter build its summary?"
antai search "spatial validation"       # see what retrieval finds, without the model
antai status                            # what's indexed, what was skipped and why
antai projects                          # every folder you've learned
```

`ask`, `search` and `status` work on the learned project that contains your current folder,
otherwise the one used most recently. Pass `-p <folder>` to choose explicitly.

Useful `ask` options: `--think medium|high` for harder questions, `--show-thinking` to see
the model's reasoning, `-k 20` to retrieve more excerpts, `-m <model>` to use another Ollama
model.

### Controlling what gets learned

`.gitignore` files are honoured at every level, and common noise is skipped by default
(`.git`, `node_modules`, virtual environments, build output, lock files, binaries, images,
Office files, file geodatabases). For anything else, add a `.antaiignore` file with
gitignore syntax, for example to skip generated docs:

```
docs/
```

## Configuration

Every setting can be overridden with an environment variable:

| Variable | Default | Meaning |
|---|---|---|
| `ANTAI_CHAT_MODEL` | `gpt-oss:20b` | Model that writes answers |
| `ANTAI_EMBED_MODEL` | `nomic-embed-text` | Model that embeds chunks (changing it rebuilds the index) |
| `ANTAI_THINK` | `low` | Reasoning effort: `low`, `medium`, `high` |
| `ANTAI_NUM_CTX` | `16384` | Context window requested from Ollama |
| `ANTAI_KEEP_ALIVE` | `30m` | How long Ollama keeps models loaded between questions |
| `ANTAI_TOP_K` | `12` | Excerpts retrieved per question |
| `ANTAI_CONTEXT_CHARS` | `30000` | Upper bound on excerpt text sent to the model |
| `ANTAI_CHUNK_CHARS` | `2400` | Target chunk size (changing it rebuilds the index) |
| `ANTAI_MAX_FILE_BYTES` | `1000000` | Larger files are skipped (CSV/TSV are previewed instead) |
| `ANTAI_HOME` | `~/.antai` | Where indexes are stored |
| `ANTAI_OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama server |

## How it works

1. **Walk** the folder, applying ignore rules and skipping binary or oversized files.
2. **Chunk** each file so that it keeps its line numbers. Code is split along functions,
   classes and methods using tree-sitter. Markdown is split by heading, notebooks by cell,
   and CSV/TSV files become a single preview of their columns and first rows. Anything
   else is split into overlapping line windows.
3. **Embed** every chunk with `nomic-embed-text`, and store it in
   `~/.antai/projects/<name>-<hash>/index.db` along with a full-text index. The learned
   folder itself is never written to.
4. **Retrieve** with a hybrid search: vector similarity and BM25 keyword search
   (camelCase and snake_case aware), merged with reciprocal rank fusion.
5. **Answer** by passing the best excerpts that fit the context budget to `gpt-oss`, which
   is told to cite `path:start-end` for what it uses.

Re-running `learn` compares each file's size, modification time and content hash, so only
new or changed files are re-embedded. Deleted files are removed from the index.

## Project layout

All the code is in `src/antaigent/`, grouped by pipeline stage:

```
src/antaigent/
  cli.py, config.py, projects.py, __main__.py   app wiring and settings, used everywhere
  ingest/     reading a folder and splitting files into chunks
  index/      building, updating and storing the index
  llm/        talking to models through Ollama
  query/      searching the index and building the prompt
  commands/   one file per antai command
```

| File | What it does |
|---|---|
| `cli.py` | Creates the `antai` app and registers each command from `commands/`. |
| `config.py` | Holds every setting and its default, and reads `ANTAI_*` environment variable overrides. |
| `projects.py` | Decides where each project's index is stored under `~/.antai`, and remembers learned folders so `ask` knows which project you mean. |
| `__main__.py` | Lets you run the tool as `python -m antaigent` as well as `antai`. |
| `ingest/walker.py` | Finds the files worth indexing in a folder, applying `.gitignore`, `.antaiignore` and the built-in skip lists. |
| `ingest/chunking.py` | Splits each file into chunks that keep their line numbers: code by function and class (tree-sitter), Markdown by heading, notebooks by cell, CSV/TSV as a preview, and other text in line windows. |
| `index/indexer.py` | Runs `learn`: walks the folder, works out which files changed, then chunks them, embeds them in batches and saves them to the store. |
| `index/store.py` | The per-project SQLite database: file records, chunks, embeddings and the keyword index, plus the search queries that read them. |
| `llm/ollama.py` | A small client for the Ollama API. It lists models, creates embeddings, and streams chat replies, keeping the reasoning separate from the answer. |
| `llm/embed.py` | Turns chunks and questions into normalized vectors, adding the task prefixes some embedding models expect (such as nomic's `search_document:` and `search_query:`). |
| `query/retrieval.py` | Runs a meaning-based search and a keyword search, and merges the two rankings into one list of best matches. |
| `query/answer.py` | Builds the prompt for the chat model: the instructions, plus as many of the top excerpts as fit the context budget. |

**`commands/`**, one file per command:

| File | What it does |
|---|---|
| `learn.py` | `antai learn`: runs the indexer with a progress bar and prints a summary of what changed. |
| `ask.py` | `antai ask`: retrieves excerpts, streams gpt-oss's answer as rendered Markdown, then lists the sources and generation speed. |
| `search.py` | `antai search`: shows what retrieval finds for a query, with each result's rank in the vector and keyword searches, without calling the chat model. |
| `status.py` | `antai status`: summarizes a project's index (file and chunk counts, embedding model, last run) and lists skipped files with the reason. |
| `projects.py` | `antai projects`: lists every learned folder and whether its index still exists. |
| `doctor.py` | `antai doctor`: checks that Ollama is reachable, the models are downloaded, loaded models are on the GPU, and the code parsers are installed. |
| `setup.py` | `antai setup`: one-time download of the code parsers, plus a check for the Ollama models. |
| `_common.py` | Helpers the commands share: console output, error messages, the `--path` option, and opening the right project's index. |

The tests in `tests/` cover file discovery (`test_walker.py`), chunking (`test_chunking.py`),
indexing, incremental updates and search (`test_index_and_search.py`), and command
registration (`test_cli.py`). `conftest.py` provides the fake embedder that lets them run
without Ollama.

## Development

```
.venv\Scripts\python -m pytest
```

The tests use a fake embedder, so they run without Ollama.

## Roadmap

- **Phase 1 (done): question answering.** `learn`, `ask`, `search`, incremental indexing.
- **Phase 2: understanding.** A symbol and import graph ("what calls this?"), file, module
  and project summaries, profiling of data files (Excel, Parquet, schemas, column stats),
  and a watch mode.
- **Phase 3: agent.** A tool-using loop (search, read file, grep, query data with DuckDB),
  persistent memory and corrections, and proposed changes as diffs.
- **Phase 4: integration.** An MCP server so editors and other agents can use the index, and an optional web UI.
