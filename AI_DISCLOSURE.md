# AI Disclosure

CodeLens uses artificial intelligence and machine-learning components for source-code retrieval. This file describes where they are used and what they do.

## What uses AI

- **Semantic retrieval:** CodeLens uses the pretrained `sentence-transformers/all-MiniLM-L6-v2` text embedding model to represent code blocks and natural-language queries as vectors. The model is run locally on CPU. Its weights are downloaded on first use and then reused from the local cache.
- **Ranking:** CodeLens combines semantic similarity with BM25 lexical search using the checked-in fusion settings. The model contributes to ranking; BM25 is a conventional lexical retrieval method.
- **Benchmarking:** The AppsRetrieval evaluation uses the MTEB/CoIR benchmark data and evaluates retrieval rankings. Benchmark figures describe retrieval performance on that dataset; they do not establish correctness or quality for every repository or query.

## What CodeLens does not do

- It does not send source code or queries to a hosted AI API as part of the documented local workflow. No API credentials are required.
- It does not generate explanations, summaries, or answers from retrieved code. It returns matching source snippets with symbols, paths, line locations, scores, and revision history.
- It does not train or fine-tune the embedding model. Fusion weights were selected using the benchmark's training split; the held-out test split was used for evaluation.

## User data and limitations

CodeLens indexes source files supplied by the user and stores its index locally in SQLite. Treat that index as a copy of indexed source content and protect it accordingly. The documented workflow requires internet access to download the model on first use and benchmark data when running evaluation; this does not mean source code is uploaded to those services.

Semantic similarity and retrieval scores are not calibrated confidence or proof that a result is relevant. Results can miss relevant code or rank irrelevant code highly. Review retrieved code in its original context before relying on it. The model and benchmark may also reflect limitations or biases in their training and evaluation data.

For model provenance, licensing, and training-data details, consult the model publisher's documentation for `sentence-transformers/all-MiniLM-L6-v2`. The project does not independently verify the model's training data.
