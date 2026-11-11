"""Five-verifier scores for every distinct raw candidate (Weaver / SSV inputs), no labels opened.

Prompts and model settings are exactly those of each dataset's main verification: GPQA uses the
fixed-candidate probability prompt and screen models of the GPQA studies (every option A-D of every
question is a candidate); MATH-500 and QuantumKatas use their verifier prompts and the
QuantumKatas/MATH verifier specs. Candidates are the union of the generators' raw-sample candidate sets
plus the frozen main-run answers. Each (verifier, candidate) is asked once; uncertain outcomes are
settled as service failures. Output: results/raw_verify_20261026/<dataset>/signals.jsonl.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
import threading
from types import SimpleNamespace

from vgx.common.api import OfflineCacheMiss, ProviderRejectedError, UncertainRequestError
from vgx.common.budget import BudgetLimitError, BudgetedRequestCache
from vgx.common.llm import Request
from vgx.common.storage import append_jsonl, read_jsonl
from vgx.gpqa import signal_study
from vgx.math_trial import data, prompt as mprompt
from vgx.quantum_katas import prompt as qprompt, runners
from vgx.quantum_katas.load import sha256_text
from vgx.quantum_katas.prompt import parse_verifier_response
from vgx.quantum_katas.run_pilot import AbandonedRequests, Run

OUT = Path('results/raw_verify_20261026')
CAND_DIRS = ['results/qwen_raw_20261021/stage2b/candidates', 'results/raw_gptoss_20261022/stage2/candidates',
             'results/raw_sonnet55_20261026/stage2/candidates']
QK_CFG = json.loads(Path('configs/quantum_katas_full_qwen3_20261019.json').read_text())
GPQA_SCREEN = json.loads(Path('configs/gpqa_signal_study_4096.json').read_text())


@dataclass(frozen=True)
class GPQACand:
    item_id: str
    subject: str
    question: str
    choices: list
    answer: str


def jobs_for(dataset):
    out = []
    if dataset == 'gpqa':
        b = json.loads(Path('results/gpqa_staged_generator_expansion_20261005/frozen_candidates.json').read_text())
        for r in b['groups']['qwen']:
            for letter in 'ABCD':
                c = GPQACand(r['item_id'], r['subject'], r['question'], list(r['choices']), letter)
                text = signal_study.prompt(c, 'probability')
                for m in GPQA_SCREEN['models']:
                    out.append((m['id'], signal_study.runner_for(m, GPQA_SCREEN['project'] if 'project' in GPQA_SCREEN else QK_CFG['budget']['project']),
                                Request(f"{m['id']}|gpqa|{r['item_id']}|{letter}", text, signal_study.SYSTEM,
                                        {'role': 'verifier', 'item_id': r['item_id'], 'candidate': letter}), r['item_id'], letter))
        return out
    specs = [runners.ModelSpec(**v) for v in QK_CFG['models']['verifiers']]
    rs = {s.id: s.runner() for s in specs}
    if dataset == 'math500':
        probs = {it['item_id']: it['problem'] for it in data.load('math500')[0]}
        cands = {}
        for d in CAND_DIRS:
            p = Path(d)/'math500_candidates.json'
            if p.exists():
                for item, cs in json.loads(p.read_text()).items():
                    for c in cs:
                        for a in c['members']:
                            cands.setdefault(item, set()).add(a)
        for item, answers in sorted(cands.items()):
            for a in sorted(answers):
                text = mprompt.build_verifier_prompt(probs[item], a)
                for s in specs:
                    out.append((s.id, rs[s.id], Request(f'{s.id}|math500|{item}|{sha256_text(a)[:16]}', text, mprompt.VERIFIER_SYSTEM,
                                                        {'role': 'verifier', 'item_id': item}), item, a))
        return out
    tasks = {t.task_id: t for t in Run('results/quantum_katas_full_qwen3_20261019').tasks}
    codes = {}
    for d in CAND_DIRS:
        p = Path(d)/'qkatas_candidates.json'
        if p.exists():
            for tid, cs in json.loads(p.read_text()).items():
                for c in cs:
                    codes[(tid, sha256_text(c['code']))] = c['code']
    for (tid, h), code in sorted(codes.items()):
        t = tasks[tid]
        view = {'task_id': tid, 'prompt': t.prompt, 'entry_point': t.entry_point,
                'required_functions': list(t.required_functions), 'code': code}
        text = qprompt.build_verifier_prompt(view, qprompt.PROMPT_VERSION_V2)
        for s in specs:
            out.append((s.id, rs[s.id], Request(f'{s.id}|qkatas|{tid}|{h}', text, qprompt.verifier_system(qprompt.PROMPT_VERSION_V2),
                                                {'role': 'verifier', 'item_id': tid}), tid, h))
    return out


def run(dataset, limit_usd, allow_api=False, workers=10):
    out = OUT/dataset; out.mkdir(parents=True, exist_ok=True)
    pricing = (GPQA_SCREEN['pricing'] if dataset == 'gpqa' else QK_CFG['budget']['pricing'])
    cache = BudgetedRequestCache(out/'cache', project=QK_CFG['budget']['project'],
                                 pricing={'usd_per_million_tokens': pricing['usd_per_million_tokens']},
                                 limit_usd=limit_usd, gemini_output_reserve=QK_CFG['budget']['gemini_output_reserve'])
    jobs = jobs_for(dataset)
    keyed = {r.cache_key(q): {'role': 'verifier', 'key': q.key} for _, r, q, _, _ in jobs}
    abandoned = AbandonedRequests(SimpleNamespace(root=out)); abandoned.settle_prior(cache, keyed)
    done = {x['request_key'] for x in read_jsonl(out/'signals.jsonl', tolerate_tail=True)} if (out/'signals.jsonl').exists() else set()
    stop = threading.Event(); lock = threading.Lock(); errors = []

    def one(job):
        vid, r, q, item, cand = job; key = r.cache_key(q)
        if stop.is_set() or key in done:
            return
        row = {'dataset': dataset, 'verifier': vid, 'item_id': item, 'candidate': cand, 'request_key': key}
        if key in abandoned.keys():
            row.update(score=None, failure='verifier_service_failure')
        else:
            try:
                call, _ = cache.get(r, q, allow_api=allow_api)
            except OfflineCacheMiss:
                return
            except UncertainRequestError:
                with lock:
                    abandoned.abandon(cache, key, keyed[key])
                row.update(score=None, failure='verifier_service_failure')
            except (ProviderRejectedError, BudgetLimitError) as exc:
                stop.set()
                with lock:
                    errors.append({'job': q.key, 'type': type(exc).__name__, 'http_status': getattr(exc, 'status', None), 'message': str(exc)[:200]})
                return
            else:
                if dataset == 'gpqa':
                    score, failure = signal_study.parse_signal(call.response, 'probability')
                else:
                    p = parse_verifier_response(call.response, truncated=runners.finish_status(call) == 'truncated')
                    score, failure = p.p_correct, p.failure
                row.update(score=score, failure=failure)
        with lock:
            append_jsonl(out/'signals.jsonl', row)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, jobs))
    have = {x['request_key'] for x in read_jsonl(out/'signals.jsonl', tolerate_tail=True)} if (out/'signals.jsonl').exists() else set()
    return {'jobs': len(jobs), 'collected': sum(r.cache_key(q) in have for _, r, q, _, _ in jobs), 'errors': errors[:3], 'budget': cache.summary()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset', choices=['gpqa', 'math500', 'qkatas'])
    p.add_argument('--limit-usd', type=float, required=True)
    p.add_argument('--allow-api', action='store_true')
    p.add_argument('--workers', type=int, default=10)
    a = p.parse_args()
    print(json.dumps(run(a.dataset, a.limit_usd, a.allow_api, a.workers), default=str))


if __name__ == '__main__':
    main()
