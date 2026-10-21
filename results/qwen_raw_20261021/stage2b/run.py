# Stage 2b runner: only the judge jobs listed in new_job_ids.json (candidates first seen in samples 6-10).
import json, sys
from vgx import qwen_raw
R = 'results/qwen_raw_20261021'
qwen_raw._long_read_timeout()
ids = {tuple(x) for x in json.load(open(f'{R}/stage2b/new_job_ids.json'))}
jobs = qwen_raw.stage2(f'{R}/samples/collected.jsonl,{R}/samples_nolp/collected.jsonl,{R}/retry1/collected.jsonl',
                       f'{R}/stage2b/candidates', 'tmp/math_grader_venv_20261020/bin/python')
jobs = [j for j in jobs if (j[0], j[1], j[2]) in ids]
print(json.dumps(qwen_raw.run(f'{R}/stage2b', jobs, project='llm-applications-490420', limit_usd=5.0,
                              allow_api=True, workers=8), default=str))
