# CodeLens update notes

## Final AppsRetrieval result

The final benchmark uses MTEB 2.21.8 and CoIR AppsRetrieval revision `f22508f96b7a36c2415181ed8bb76f76e04ae2d5`, test split:

| Configuration | NDCG@10 | MRR@10 |
| --- | ---: | ---: |
| BM25 control, chunk settings matched | 0.02032 | 0.01586 |
| MiniLM dense, chunk settings matched | 0.08050 | 0.06768 |
| Hybrid, train-tuned min-max fusion | 0.08903 | 0.07496 |

The final hybrid beats the previously measured 0.06675 / 0.05494 unchunked hybrid baseline. A fresh `python evaluate.py --mode hybrid` reproduction of the prior code produced exactly 0.06675 / 0.05494 before the BM25 and window changes. The final run used the explicit generated fusion file and default 128-token code/query windows with 24-token code overlap. That selected configuration is now also in `codelens_fusion.json`; the bare `--mode hybrid` command reads it by default.

BM25 now keeps `input`, `output`, `data`, `value`, `code`, `file`, `function`, and `main`, and repeated query terms have capped logarithmic weight. The unchunked BM25 run measured 0.02090 / 0.01697. With the final chunk settings, BM25 measured 0.02032 / 0.01586. The chunked dense pass reached 0.08050 / 0.06768. Equal-weight chunked RRF reached 0.08441 / 0.07034. A deterministic train-only sample of 500 held-out queries and 1,500 code documents selected 25% BM25 / 75% semantic min-max fusion; its full test evaluation reached 0.08903 / 0.07496. No training selection used test labels.

The final standalone evaluation took 1,733.2 seconds. It encoded 10,622 code windows in 899.5 seconds and 10,961 query windows in 751.3 seconds. Across the 3,765 test queries, the retrieval-scoring component measured 15.00 ms p50 / 19.54 ms p95, excluding query embedding. These measurements are for the benchmark corpus and windows, not a real repository indexing run.

## Real repository speed check

A public pytest-dev/pytest checkout indexed 7,830 blocks from 267 Python files in 429.57 s. A revision changing three paths, including two Python files, indexed in 1.49 s; two source files were read, with 7,830 vector cache hits and zero misses. Warm search over ten queries x five repeats measured p50 145.4 ms / p95 366.2 ms, excluding initial model load and index build. Details: benchmark_runs/pytest_speed_summary.json.

## App and version demo

- The local interface shows query latency and the number of versions searched.
- `demo_data/auth_versions.sqlite` contains the two authentication snapshots. An all-version API query returned the exact `parse_bearer_token` block once with both revisions in its history; the measured single warm request was 49.8 ms.
- Indexing the small folder snapshots reported 27.11 s for the first two-block snapshot and 29.23 s for the second. The second snapshot reused one of two embeddings. Startup/model overhead dominates these fixture timings; the separate pytest benchmark below measures a real repository.
- `queries.txt` now contains ten natural-language code-search examples for `benchmark_latency.py`.

## Presentation and docs

- The final deck is `deliverables/CodeLens_SRMIST_Hackathon_Final.pptx`. Slides 5, 8, 9, 10, and 11 now describe the demo, final scores, train-only tuning, and exact-duplicate revision history. The root deck copy is synchronized with the final deck.
- The README includes setup, run commands, final benchmark settings, metrics, latency scope, and limitations.
- The package integrity, 12-slide count, expected slide size, and first-party PPTX import passed validation.

## Work not completed

- GitHub shows `R1thanya/codelens-retrieval` as public, but the repository has no commits; the release page explicitly disables publishing for an empty repository. No release or uploaded artifact was created. The local `appsretrieval_results.json` is ready for upload once repository content has been pushed.
- A clean clone from GitHub could not be run because the public repository is empty. A separate clean local copy passed `pip install -r requirements-eval.txt`, automatically downloaded MiniLM on first indexing, indexed five sample blocks in 72.00 s, started the UI, and returned five search results (39.5 ms). The UI used port 8767 because the README port 8765 was already occupied. This verifies local setup but cannot substitute for a GitHub clone.
- Warm p50/p95 and changed-commit indexing were measured on pytest at 7,830 blocks; only one repository and one revision edge were measured. The AppsRetrieval scoring p50/p95 separately covers 10,622 code windows.
- The CPU run for `flax-sentence-embeddings/st-codesearch-distilroberta-base` did not finish in the earlier attempt. No code-specialized encoder score is claimed.
- Near-duplicate evolutionary lineage, cross-encoder reranking, and paired significance testing are not implemented.

The project and official MTEB outputs do not include credentials or API keys.
