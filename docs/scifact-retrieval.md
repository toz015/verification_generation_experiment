# SciFact — BM25 retrieval

BM25Okapi over title + abstract of all 5183 abstracts.
Lowercase alphanumeric tokens, no stemming, no stopword list.

## Recall of gold evidence abstracts (full dev)

NEI claims are excluded: they have no evidence abstract, so retrieval
recall is undefined for them and including them would deflate every row.

n claims = 188, gold abstracts = 209

| k | hit rate | micro recall |
|---|---|---|
| 1 | 70.2% | 63.2% |
| 2 | 79.8% | 73.2% |
| 3 | 84.0% | 77.5% |
| 5 | 89.9% | 83.7% |
| 10 | 93.6% | 88.5% |
| 20 | 95.7% | 90.9% |
| 50 | 95.7% | 92.3% |
| 100 | 96.3% | 92.8% |

## Recall of cited abstracts (all dev claims)

A different question: can BM25 surface the abstract an annotator judged
relevant, including the NEI cases where it turned out not to support the claim.

n claims = 300, gold abstracts = 339

| k | hit rate | micro recall |
|---|---|---|
| 1 | 52.3% | 46.3% |
| 2 | 63.7% | 57.5% |
| 3 | 69.3% | 63.1% |
| 5 | 75.3% | 69.6% |
| 10 | 79.7% | 75.5% |
| 20 | 85.0% | 81.1% |
| 50 | 87.7% | 86.1% |
| 100 | 88.3% | 86.7% |

## Pinned sample (n=50)

Reported to confirm the sample is not retrieval-anomalous relative to full dev.

n claims = 31, gold abstracts = 31

| k | hit rate | micro recall |
|---|---|---|
| 1 | 67.7% | 67.7% |
| 2 | 87.1% | 87.1% |
| 3 | 87.1% | 87.1% |
| 5 | 93.5% | 93.5% |
| 10 | 96.8% | 96.8% |
| 20 | 96.8% | 96.8% |
| 50 | 96.8% | 96.8% |
| 100 | 96.8% | 96.8% |

## Context budget

Longest abstracts: doc 86217760 (367 sentences), doc 12207167 (335 sentences), doc 36271512 (169 sentences), doc 26067999 (73 sentences), doc 10749308 (58 sentences).
Median is 8 sentences.

Relevant to S3: a top-k prompt concatenates k abstracts, so k and the
context limit have to be chosen against the tail, not the median.
