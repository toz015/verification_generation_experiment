# PubMedQA — dataset anatomy

Items: `qiaojin/PubMedQA` config `pqa_labeled` (1000 expert-annotated).
Split: official 500-item test set, from the authors' `test_ground_truth.json`.

No retrieval stage — the context ships with each item.

## Label distribution

| label | PQA-L (1000) | official test (500) |
|---|---|---|
| yes | 552 (55.2%) | 276 (55.2%) |
| no | 338 (33.8%) | 169 (33.8%) |
| maybe | 110 (11.0%) | 55 (11.0%) |

## Pinned sample (n=50)

| label | items |
|---|---|
| yes | 28 |
| no | 17 |
| maybe | 5 |

`maybe` has only 5 items here. That is the class this dataset
exists to test, so its recall carries a very wide interval; treat it as
directional. Raising `n` to 500 in the config gives 55.

## Structure

Context sections per item: 2–9 (mean 3.4), labelled BACKGROUND / METHODS / RESULTS.

```
pubid          str
question       str
context        {contexts: list[str], labels: list[str], meshes: list[str]}
long_answer    str   <- the withheld CONCLUSIONS section
final_decision yes | no | maybe
```

**`long_answer` must never enter the prompt.** It is the abstract's
conclusion and states the answer outright; including it turns the task into
copying. The loader does not even read the field.
