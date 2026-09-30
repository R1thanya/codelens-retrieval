# CodeLens

CodeLens retrieves and ranks source-code blocks for natural-language queries. It returns code snippets with symbols, locations, and indexed revisions. It does not generate answers from retrieved code.

**Team:** Rithanya M and Rithika R · SRMIST Kattankulathur · B.Tech CSE (CTech), 3rd year  
**Challenge:** Agentic Code Intelligence  
**Repository:** https://github.com/R1thanya/codelens-retrieval

## Retrieval approach

- **Dense retrieval:** `sentence-transformers/all-MiniLM-L6-v2`, run locally on CPU. The first indexing run downloads the model; later runs reuse its cache. Code-specific encoder adapters are available, but the final benchmark uses MiniLM.
- **Code chunks:** Python definitions come from the AST, including nested definitions and decorators. The AppsRetrieval evaluation further divides long code documents into 128-token windows with 24-token overlap and keeps the highest chunk score for each original document.
- **Query chunks:** The benchmark splits long problem statements into 128-token windows, encodes each window, and uses the highest similarity for each document. This addresses MiniLM's input-length limit.
- **Lexical retrieval:** BM25 retains content terms such as `input`, `output`, `data`, `value`, `code`, `file`, `function`, and `main`. Repeated query words receive capped logarithmic weight.
- **Fusion:** A train-only search selected per-query min-max fusion with 25% BM25 and 75% semantic weight. The configuration is saved in `codelens_fusion.json` and is used by both evaluation and interactive search.
- **Versions:** Each Git commit has a separate index snapshot. Content/model-keyed embeddings are reused for unchanged blocks. The search server caches its BM25 index and embedding matrix and invalidates them when the SQLite database changes.
- **All-version results:** Identical normalized blocks with the same path and symbol appear once, with a revision history. Changed code stays separate. Near-duplicate evolutionary lineage is not implemented.

## Requirements and setup

- Python 3.10 or newer
- Git for indexing Git commits; plain folders also work
- Internet access for first-time model and MTEB dataset downloads
- CPU-only PyTorch is pinned in `requirements-eval.txt`; no GPU or API credentials are required

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-eval.txt
```

## Run the local demo

Index the included sample and start the local page:

```powershell
python codelens.py index examples/mini_repo --revision workspace --db .codelens/demo.sqlite
python codelens.py serve --db .codelens/demo.sqlite
```

Open `http://127.0.0.1:8765`. Search results show the matching code, source path, line range, relative score, query latency, and number of revisions searched. Scores are relative to the top result in that query, not calibrated confidence values.

The included authentication demo database already contains two snapshots. Run:

```powershell
python codelens.py serve --db demo_data/auth_versions.sqlite --port 8766
```

Open `http://127.0.0.1:8766`, ask `Where are bearer credentials checked?`, select **All versions**, and expand the history for `parse_bearer_token`. The exact parser block appears once with both revisions. This is a four-block fixture, not large-repository evidence.

For your own source folder:

```powershell
python codelens.py index C:\path\to\source --revision workspace
python codelens.py serve
```

For a Git repository:

```powershell
python codelens.py index C:\path\to\repo --revision HEAD
python codelens.py index C:\path\to\repo --all
python codelens.py search "Where are access tokens validated?" --all-versions --top-k 10
```

## AppsRetrieval benchmark

Run all three configurations on the pinned CoIR AppsRetrieval **test** split:

```powershell
python evaluate.py --mode all
```

To generate only the required hybrid artifact, with the same default windows and checked-in fusion settings:

```powershell
python evaluate.py --mode hybrid
```

The harness uses `mteb.get_task("AppsRetrieval")` and `mteb.evaluate(...)`. It downloads the pinned dataset revision automatically when needed. `appsretrieval_results.json` is native MTEB output suitable for a GitHub Release artifact. The independent training tune uses only AppsRetrieval's train split.

The completed MTEB 2.21.8 evaluation on CoIR revision `f22508f96b7a36c2415181ed8bb76f76e04ae2d5` measured:

| Configuration | NDCG@10 | MRR@10 |
| --- | ---: | ---: |
| BM25 control, same chunk settings | 0.02032 | 0.01586 |
| MiniLM dense, same chunk settings | 0.08050 | 0.06768 |
| CodeLens hybrid, train-tuned min-max | **0.08903** | **0.07496** |

The new hybrid result is 0.02228 NDCG@10 above the previously reproduced unchunked hybrid baseline (0.06675), a 33.4% relative increase. The previous baseline was reproduced at 0.06675 NDCG@10 / 0.05494 MRR@10 before these changes. These are measured comparisons on the same test split, not claims of state-of-the-art performance.

The final benchmark artifact and matched BM25/dense controls are at the project root. `benchmark_runs/` keeps the reproduced baseline and separate ablation outputs.

The default 128-token code/query windows and 24-token code overlap produced 10,622 code windows from 8,765 original code documents and 10,961 query windows from 3,765 test queries. In the final standalone hybrid run, cold encoding took 899.5 seconds for code and 751.3 seconds for queries; total evaluation time was 1,733.2 seconds (28 minutes 53 seconds). The retrieval-scoring component measured 15.00 ms p50 / 19.54 ms p95 across the 3,765 queries. That component includes chunk aggregation, ranking and top-k selection, and excludes query embedding. It measures the AppsRetrieval corpus windows, not interactive indexing of a real Git repository.

The test-config tuning used 500 held-out training queries and a deterministic sample of 1,500 training code documents. It selected min-max fusion with 25% lexical and 75% semantic weight. The selected configuration was then evaluated on the test split. The configuration file is `codelens_fusion.json`.

Run a fresh tuning report without using test labels:

```powershell
python evaluate.py --mode dense --tune-train --tune-train-queries 500 --tune-train-documents 1500
```

The tune command writes `fusion_tuning_sentence-transformers-all-minilm-l6-v2.json` and `fusion_sentence-transformers-all-minilm-l6-v2.json`. To evaluate that generated configuration explicitly:

```powershell
python evaluate.py --mode hybrid --fusion-config fusion_sentence-transformers-all-minilm-l6-v2.json
```

To compare against an unchunked run or the legacy BM25 stop list:

```powershell
python evaluate.py --mode all --chunk-tokens 0 --query-chunk-tokens 0
python evaluate.py --mode all --drop-content-terms
```

`--chunk-tokens`, `--chunk-overlap`, and `--query-chunk-tokens` control the windows. The default outputs are `appsretrieval_results_bm25.json`, `appsretrieval_results_dense.json`, and `appsretrieval_results.json`. `CODELENS_ENCODE_BATCH_SIZE` (default 64) and `CODELENS_TORCH_THREADS` (default 4) adjust CPU memory and thread use.

## Real repository speed check

On public repository [pytest-dev/pytest](https://github.com/pytest-dev/pytest), CodeLens indexed 7,830 blocks from 267 Python files. The first index took 429.57 s. A later commit changing three paths (two Python files) indexed in 1.49 s; it read the two changed Python files and reused all 7,830 cached vectors (zero embedding misses). With the index warm, ten queries repeated five times measured 145.4 ms p50 / 366.2 ms p95. The query measurement excludes first model load/download and indexing. The benchmark revision details are in benchmark_runs/pytest_speed_summary.json.

## Latency query file

`queries.txt` contains ten natural-language code-search queries. Measure warm interactive latency for an indexed local project with:

```powershell
python benchmark_latency.py --db .codelens/index.sqlite --revision HEAD --queries queries.txt --repeat 5
```

The report includes warm-query p50/p95, block count, operating system, and Python version. It excludes first model load/download and index construction. The AppsRetrieval per-query scoring measurements above provide a separate result at over ten thousand code windows.

## Scope and limitations

- The interactive application indexes function/class blocks. Benchmark windows are additional ranking units used for the long AppsRetrieval documents and queries.
- The real-repository timing uses one pytest history edge and ten representative queries; it is a measured demonstration, not a broad latency distribution across many repositories.
- The final encoder is a general-purpose MiniLM model. The code-specific DistilRoBERTa CPU run did not finish within the available run budget, so no score is claimed for it.
- All-version history merges exact duplicate blocks only. Near-duplicate evolutionary lineages and cross-encoder reranking are not implemented.
- Paired significance testing was not run. The score differences are reported as point estimates.
