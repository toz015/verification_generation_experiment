# SciFact — dataset anatomy

Source: https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz

## Splits

| split | claims |
|---|---|
| train | 809 |
| dev | 300 |
| test | 300 (labels withheld) |

Corpus: **5183** abstracts, 3–367 sentences each (mean 8.9), 977 structured.

## Dev label distribution

| label | claims | share |
|---|---|---|
| SUPPORT | 124 | 41.3% |
| CONTRADICT | 64 | 21.3% |
| NEI | 112 | 37.3% |

## Pinned sample (n=50, scifact_sample_50.json)

| label | claims | share |
|---|---|---|
| SUPPORT | 20 | 40.0% |
| CONTRADICT | 11 | 22.0% |
| NEI | 19 | 38.0% |

## Schema

```
corpus.jsonl   doc_id:int  title:str  abstract:list[str]  structured:bool
claims_*.jsonl id:int  claim:str  cited_doc_ids:list[int]
               evidence: {doc_id_as_STRING: [{sentences:list[int], label:str}]}
```

Note the asymmetry: `evidence` keys are strings while `cited_doc_ids` are ints.
A claim with an empty `evidence` object is NEI — the abstention ground truth.
`cited_doc_ids` is populated even for NEI claims, so it is not a label proxy.
