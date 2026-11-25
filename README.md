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
uv pip install -e ".[gui,dev]"
.venv\Scripts\antai setup      # one-time: downloads code parsers (needs internet)
.venv\Scripts\antai doctor     # checks Ollama, models, GPU use, parsers and extras
```

On macOS/Linux use `.venv/bin/antai`. Leave out `gui` if you only want the command line.

## Use

```
antai learn C:\path\to\project          # index a folder (re-run any time; only changes are processed)
antai ask "How does the reporter build its summary?"
antai summarize                         # optional: file, folder and project summaries (see below)
antai gui                               # the desktop app
```

Other commands:

```
antai search "spatial validation"       # see what retrieval finds, without the model
antai symbol Logger.log                 # where it's defined, what calls it, what it calls
antai deps Logger.py                    # what a file imports, and which files import it
antai overview                          # the project overview and top-level folder summaries
antai watch                             # re-learn automatically while you work (Ctrl+C to stop)
antai status                            # what's indexed, what was skipped and why
antai projects                          # every folder you've learned
```

Commands work on the learned project that contains your current folder, otherwise the one
used most recently. Pass `-p <folder>` to choose explicitly.

Useful `ask` options: `--think medium|high` for harder questions, `--show-thinking` to see
the model's reasoning, `-k 20` to retrieve more excerpts, `-m <model>` to use another Ollama
model.

### The desktop app

`antai gui` opens a native window (Qt). On Windows, `antai-gui` starts it without a console
window, so you can pin it to the taskbar. It has:

- **Ask**: a chat about the project. Answers stream in as formatted Markdown, follow-up
  questions remember the conversation, and every `path:start-end` citation is a link that
  opens the file at those lines.
- **Search**: hybrid search results with a preview of each excerpt.
- **File**: the source with syntax colouring and the cited lines highlighted, next to the
  file's summary, outline, imports and the files that import it.
- **Symbol**: where a function or class is defined, every call site, and what it calls.
- **Overview**: the project overview, index statistics, folder summaries and skipped files.
- A sidebar with the file tree, a symbol finder, and **Chats**: every conversation is saved
  per project, so you can reopen one and carry on where you left off. Right-click a chat to
  rename or delete it.
- Toolbar buttons to learn a new folder, update the index, summarize, or watch for changes.

Chats are stored in `chats.db` next to the project's index, not inside it, so rebuilding
the index (`learn --full`, or changing the embedding model) never deletes them.

### Summaries

`antai summarize` (or **Summarize** in the app) has the chat model write a short summary of
every file, then every folder, then an overview of the whole project. `ask` uses them for
broad questions such as "what does this project do?" or "where is validation handled?",
which no single excerpt answers. The first run takes a few seconds per file. After that,
only files that changed since their summary, and the folders above them, are redone.

### Controlling what gets learned

`.gitignore` files are honoured at every level, and common noise is skipped by default
(`.git`, `node_modules`, virtual environments, build output, lock files, binaries, images,
Office documents, file geodatabases). HTML pages produced by API documentation generators
(pdoc, Sphinx, Doxygen, Javadoc, TypeDoc and others) are skipped too, because they repeat the
source code that is already indexed.

Common secret files are never learned, because the index keeps a copy of everything it reads
and answers can quote it: `.env` and `.env.*` (but not `.env.example`, `.env.sample` or
`.env.template`), private keys and certificates (`*.pem`, `*.key`, `*.p12`, `*.pfx`,
`id_rsa` and friends), and `.netrc`, `.pypirc` and `.npmrc`. Symlinks and junctions that
point outside the folder are not followed, so learning never reads files from elsewhere.

For anything else, add a `.antaiignore` file with gitignore syntax, for example:

```
archive/
*.log
```

### Data files

CSV, TSV, Parquet and Excel files are profiled instead of read as text: row counts, each
column's type, value range, mean, most common values and how often it's empty, plus a few
sample rows (one profile per Excel sheet). SQLite databases and GeoPackages get a description
of each table's columns. That lets `ask` answer questions like "which file has the biomass
column?" or "what are the possible values of status?". Profiling uses DuckDB and runs
locally; files over 200 MB are profiled from a 500,000-row sample.

## Configuration

Every setting can be overridden with an environment variable:

| Variable | Default | Meaning |
|---|---|---|
| `ANTAI_CHAT_MODEL` | `gpt-oss:20b` | Model that writes answers |
| `ANTAI_SUMMARY_MODEL` | the chat model | Model that writes summaries |
| `ANTAI_EMBED_MODEL` | `nomic-embed-text` | Model that embeds chunks (changing it rebuilds the index) |
| `ANTAI_THINK` | `low` | Reasoning effort: `low`, `medium`, `high` |
| `ANTAI_NUM_CTX` | `16384` | Context window requested from Ollama |
| `ANTAI_KEEP_ALIVE` | `30m` | How long Ollama keeps models loaded between questions. The app starts loading them as you type a question, so the first answer doesn't wait for them. |
| `ANTAI_TOP_K` | `12` | Excerpts retrieved per question |
| `ANTAI_CONTEXT_CHARS` | `30000` | Upper bound on excerpt text sent to the model |
| `ANTAI_CHUNK_CHARS` | `2400` | Target chunk size (changing it rebuilds the index) |
| `ANTAI_MAX_FILE_BYTES` | `1000000` | Larger files are skipped (CSV/TSV are previewed instead) |
| `ANTAI_HOME` | `~/.antai` | Where indexes are stored |
| `ANTAI_OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama server. Proxy settings are ignored for a local address, so requests can't detour through a proxy. Pointing this at another machine sends your files and questions there; `antai doctor` warns when it does. |

## How it works

1. **Walk** the folder, applying ignore rules and skipping binary or oversized files.
2. **Chunk** each file so that it keeps its line numbers. Code is split along functions,
   classes and methods using tree-sitter. Markdown is split by heading and notebooks by
   cell, data files become profiles, and anything else is split into overlapping line
   windows.
3. **Map the code**: tree-sitter's tags queries find each definition (with its enclosing
   class, e.g. `Store.put_file`) and each call site (with the function it's in). Imports
   are found per language and resolved to project files where possible.
4. **Embed** every chunk with `nomic-embed-text`, and store everything in
   `~/.antai/projects/<name>-<hash>/index.db` along with a full-text index. The learned
   folder itself is never written to.
5. **Retrieve** with a hybrid search: vector similarity and BM25 keyword search
   (camelCase and snake_case aware), merged with reciprocal rank fusion.
6. **Answer** by passing `gpt-oss` the project overview, the summaries of the most relevant
   files and folders, code-graph facts about any function, class or file the question
   names (where it's defined, what calls it, what imports it), and as many of the best
   excerpts as fit the context budget. It is told to cite `path:start-end` for what it uses.

Re-running `learn` compares each file's size, modification time and content hash, so only
new or changed files are re-processed. Deleted files are removed from the index, and so are
their summaries and graph entries.

Learning keeps Ollama busy: files are read several at a time (antivirus scanning makes
each file slow to open on Windows), and each batch of chunks is embedded while the next
files are chunked and mapped, so a large folder learns at about the speed Ollama embeds.

## Project layout

All the code is in `src/antaigent/`, grouped by pipeline stage:

```
src/antaigent/
  cli.py, config.py, projects.py, __main__.py   app wiring and settings, used everywhere
  ingest/     reading a folder: files, chunks, data profiles and code facts
  index/      building, updating and storing the index, summaries, watching for changes
  llm/        talking to models through Ollama
  query/      gathering context for a question and building the prompt
  commands/   one file per antai command
  gui/        the desktop app
```

| File | What it does |
|---|---|
| `cli.py` | Creates the `antai` app and registers each command from `commands/`. |
| `config.py` | Holds every setting and its default, and reads `ANTAI_*` environment variable overrides. |
| `projects.py` | Decides where each project's index is stored under `~/.antai`, and remembers learned folders so commands know which project you mean. |
| `chats.py` | Saves the app's conversations per project in `chats.db`, beside the index so rebuilds don't erase them. |
| `__main__.py` | Lets you run the tool as `python -m antaigent` as well as `antai`. |
| `ingest/walker.py` | Finds the files worth indexing in a folder, applying `.gitignore`, `.antaiignore` and the built-in skip lists. |
| `ingest/chunking.py` | Splits each file into chunks that keep their line numbers: code by function and class (tree-sitter), Markdown by heading, notebooks by cell, and other text in line windows. |
| `ingest/profiling.py` | Profiles CSV, TSV, Parquet and Excel files with DuckDB (types, ranges, common values, empties, sample rows) and describes SQLite and GeoPackage tables. |
| `ingest/graph.py` | Finds definitions and call sites with tree-sitter tags queries, finds imports for each language, and resolves imports to project files. |
| `index/indexer.py` | Runs `learn`: walks the folder, works out which files changed, reads them a few at a time, then chunks, profiles and maps them. Each batch of chunks is embedded on a background thread while the next is prepared, and saved to the store in one transaction. |
| `index/store.py` | The per-project SQLite database: file records, chunks, embeddings, the keyword index, the code graph and summaries, plus the queries that read them. |
| `index/summarize.py` | Has the chat model summarize each file, then each folder from its contents, then the project; fingerprints each summary so only stale ones are redone. |
| `index/watch.py` | Watches a folder for changes (with `watchfiles`) so the index can be brought up to date. |
| `llm/ollama.py` | A small client for the Ollama API. It lists models, creates embeddings, and streams chat replies, keeping the reasoning separate from the answer. |
| `llm/embed.py` | Turns chunks and questions into normalized vectors, adding the task prefixes some embedding models expect (such as nomic's `search_document:` and `search_query:`). |
| `query/retrieval.py` | Runs a meaning-based search and a keyword search, and merges the two rankings into one list of best matches. |
| `query/context.py` | Gathers everything for one question: the search results, the project overview, relevant summaries, and graph facts about the symbols and files the question names. Used by both `ask` and the app. |
| `query/answer.py` | Builds the prompt for the chat model: the instructions, the gathered notes, earlier turns of the conversation, and as many excerpts as fit the context budget. |

**`commands/`**, one file per command:

| File | What it does |
|---|---|
| `learn.py` | `antai learn`: runs the indexer with a progress bar and prints a summary of what changed. |
| `ask.py` | `antai ask`: gathers context, streams gpt-oss's answer as rendered Markdown, then lists the sources and generation speed. |
| `search.py` | `antai search`: shows what retrieval finds for a query, with each result's rank in the vector and keyword searches, without calling the chat model. |
| `symbol.py` | `antai symbol`: where a symbol is defined, every place that calls it, and what it calls. |
| `deps.py` | `antai deps`: a file's imports (and the project files they resolve to) and the files that import it. |
| `summarize.py` | `antai summarize`: writes or refreshes file, folder and project summaries with a progress bar. |
| `overview.py` | `antai overview`: prints the project overview and the top-level folder summaries. |
| `watch.py` | `antai watch`: re-learns the project after each burst of file changes, optionally refreshing summaries too. |
| `gui.py` | `antai gui`: opens the desktop app. |
| `status.py` | `antai status`: summarizes a project's index (files, chunks, code graph, summaries, embedding model, last run) and lists skipped files with the reason. |
| `projects.py` | `antai projects`: lists every learned folder and whether its index still exists. |
| `doctor.py` | `antai doctor`: checks Ollama, the models, GPU use, the code parsers, and the optional extras (DuckDB, openpyxl, watchfiles, PySide6). |
| `setup.py` | `antai setup`: one-time download of the code parsers, plus a check for the Ollama models. |
| `_common.py` | Helpers the commands share: console output, error messages, the `--path` option, and opening the right project's index. |

**`gui/`**, the desktop app:

| File | What it does |
|---|---|
| `app.py` | Starts the app (`antai gui` and `antai-gui`) and draws its icon. |
| `window.py` | The main window: toolbar, sidebar, tabs and status bar; runs learn, summarize and watch in the background and keeps the views up to date. |
| `panels.py` | The Ask, Search, File, Symbol and Overview tabs, and the sidebar's file tree, symbol finder and saved chats. |
| `widgets.py` | A code view with line numbers, syntax colouring and highlighted lines, and a browser whose links navigate the app. |
| `render.py` | Turns Markdown answers and code into the HTML Qt can display, and makes citations into links. |
| `workers.py` | Background jobs (learn, summarize, ask, search, watch, Ollama status), each on its own thread with its own database connection, and the warm-up that loads the models while a question is typed. |

The tests in `tests/` cover file discovery (`test_walker.py`), chunking (`test_chunking.py`),
data profiling (`test_profiling.py`), the code graph (`test_graph.py`), indexing, incremental
updates, graph upkeep and search (`test_index_and_search.py`), learning when Ollama fails or is
cancelled part way (`test_learn_pipeline.py`), summaries (`test_summarize.py`), the Ollama
client (`test_ollama.py`), saved chats (`test_chats.py`), watching (`test_watch.py`), the
commands (`test_cli.py`) and the app (`test_gui.py`, run without a screen). `conftest.py`
provides the fake embedder that lets them run without Ollama.

## Development

```
.venv\Scripts\python -m pytest
```

The tests use a fake embedder, so they run without Ollama.

## Roadmap

- **Phase 1 (done): question answering.** `learn`, `ask`, `search`, incremental indexing.
- **Phase 2 (done): understanding.** A symbol and import graph (`symbol`, `deps`, and graph
  facts in `ask`), file, folder and project summaries, profiling of data files (CSV,
  Parquet, Excel, SQLite and GeoPackage schemas), watch mode, and the desktop app.
- **Phase 3: agent.** A tool-using loop (search, read file, grep, query data with DuckDB),
  persistent memory and corrections, and proposed changes as diffs.
- **Phase 4: integration.** An MCP server so editors and other agents can use the index.
