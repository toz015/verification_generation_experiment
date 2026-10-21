"""Raw Qwen3-235B data for future baselines, collected before the endpoint is retired (2026-10-21).

Nothing here is a baseline implementation; it only stores what baselines need from Qwen itself.
Every provider response is kept whole in the request cache (including token logprobs when
requested); collected.jsonl adds one parsed row per job. Jobs (stage 1 unless noted):

  gpqa_option_probs   per-option probability for A-D (JSON), T=0, top-5 logprobs      (PEG etc.)
  gpqa_letter         single answer letter, max 1 token, T=0, top-5 logprobs          (letter distribution)
  gpqa_samples        N independent samples of the GPQA generator prompt, T=0.7      (Weaver, self-consistency)
  math500_samples     N samples of the MATH prompt, T=0.7, token logprobs
  qkatas_samples      N samples of the QuantumKatas v2 prompt, T=0.7, token logprobs
  stage 2 (after stage 1): Qwen as judge on every distinct candidate
  {gpqa,math500,qkatas}_judge_p      verbalized p_correct for a fixed candidate
  {gpqa,math500,qkatas}_judge_yesno  "Yes"/"No" single token, top-5 logprobs (P(True))

Samples differ by a `seed` request parameter (part of request identity). Uncertain outcomes are
settled as service failures; rejections and budget stops halt the run for resume.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
from types import SimpleNamespace

from vgx.common.api import OfflineCacheMiss, ProviderRejectedError, UncertainRequestError
from vgx.common.budget import BudgetLimitError, BudgetedRequestCache
from vgx.common.llm import Request
from vgx.common.storage import append_jsonl, read_jsonl
from vgx.gpqa.prompt import SYSTEM as GPQA_SYSTEM
from vgx.math_trial import prompt as mprompt
from vgx.quantum_katas import prompt as qprompt
from vgx.quantum_katas.run_pilot import AbandonedRequests, Run
from vgx.quantum_katas.runners import JsonModeVertexRunner

MODEL = 'qwen/qwen3-235b-a22b-instruct-2507-maas'
LETTERS = 'ABCD'
GPQA_BUNDLE = 'results/gpqa_staged_generator_expansion_20261005/frozen_candidates.json'
QK_RUN = 'results/quantum_katas_full_qwen3_20261019'


class RawRunner(JsonModeVertexRunner):
    """Qwen chat completions with optional JSON mode and extra body params (seed, logprobs)."""

    def __init__(self, project, *, max_tokens, temperature, json_mode, extra):
        super().__init__(MODEL, 'us-south1', project, {'type': 'json_object'}, max_tokens=max_tokens,
                         temperature=temperature, top_p=1.0)
        self.json_mode, self.extra = json_mode, dict(extra)

    @property
    def params(self):
        value = dict(super().params)
        if not self.json_mode:
            value.pop('response_format')
        return {**value, **self.extra}

    @property
    def identity(self):
        value = dict(super().identity)
        if not self.json_mode:
            value.pop('response_format')
        return {**value, 'extra': self.extra, 'raw_collection': 'qwen_raw_v1'}


def _gpqa_items():
    b = json.loads(Path(GPQA_BUNDLE).read_text())
    return [{'item_id': r['item_id'], 'subject': r['subject'], 'question': r['question'], 'choices': r['choices']}
            for r in b['groups']['qwen']]


def _gpqa_body(it):
    options = '\n'.join(f'{l}. {c}' for l, c in zip(LETTERS, it['choices']))
    return f"Subject: {it['subject']}\n\nQuestion: {it['question']}\n\n{options}\n\n"


def _math_items():
    plan = json.loads(Path('results/math500_full_qwen3_20261021/plan.json').read_text())
    return plan['sample']['math500']


def _qk_tasks():
    return Run(QK_RUN).tasks


def stage1(project, n):
    jobs = []
    for it in _gpqa_items():
        body = _gpqa_body(it)
        jobs.append(('gpqa_option_probs', it['item_id'], 0, body +
                     'For each option, report your probability that it is the correct answer. The four probabilities '
                     'must sum to 1.\nReply with JSON only: {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}', GPQA_SYSTEM,
                     dict(max_tokens=512, temperature=0.0, json_mode=True, extra={'logprobs': True, 'top_logprobs': 5})))
        jobs.append(('gpqa_letter', it['item_id'], 0, body + 'Reply with the letter of the best answer only.', GPQA_SYSTEM,
                     dict(max_tokens=1, temperature=0.0, json_mode=False, extra={'logprobs': True, 'top_logprobs': 5})))
        gen = body + ('Choose the best answer. Report your probability that your selected answer is correct.\n'
                      'Reply with JSON only: {"answer": "A"|"B"|"C"|"D", "p_correct": 0.0}')
        for s in range(n):
            jobs.append(('gpqa_samples', it['item_id'], s, gen, GPQA_SYSTEM,
                         dict(max_tokens=4096, temperature=0.7, json_mode=True,
                              extra={'seed': s, 'logprobs': True, 'top_logprobs': 5})))
    for it in _math_items():
        for s in range(n):
            jobs.append(('math500_samples', it['item_id'], s, mprompt.build_generator_prompt(it['problem']),
                         mprompt.GENERATOR_SYSTEM, dict(max_tokens=16384, temperature=0.7, json_mode=True,
                                                        extra={'seed': s, 'logprobs': True})))
    for t in _qk_tasks():
        for s in range(n):
            jobs.append(('qkatas_samples', t.task_id, s, qprompt.build_generator_prompt(t, qprompt.PROMPT_VERSION_V2),
                         qprompt.generator_system(qprompt.PROMPT_VERSION_V2),
                         dict(max_tokens=8192, temperature=0.7, json_mode=True, extra={'seed': s, 'logprobs': True})))
    return jobs


def parse(kind, text):
    try:
        if kind == 'gpqa_letter':
            return {'letter': text.strip()[:1] or None}
        if kind.endswith('_yesno'):
            return {'token': text.strip()}
        value = json.loads(text)
        if kind == 'gpqa_option_probs':
            return {'probs': {l: value.get(l) for l in LETTERS}}
        if kind == 'qkatas_samples':
            p = qprompt.parse_generator_response(text)
            return {'code': p.code, 'p_correct': p.p_correct, 'failure': p.failure}
        if kind.endswith('_judge_p'):
            return {'p_correct': value.get('p_correct')}
        return {'answer': value.get('answer'), 'p_correct': value.get('p_correct')}
    except Exception as exc:
        return {'parse_error': type(exc).__name__}


def run(out, jobs, *, project, limit_usd, allow_api=False, workers=8, kinds=None):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    pricing = {'usd_per_million_tokens': {MODEL: {'input': 0.22, 'output': 0.88, 'cached_input': 0.22}}}
    cache = BudgetedRequestCache(out/'cache', project=project, pricing=pricing, limit_usd=limit_usd)
    jobs = [j for j in jobs if kinds is None or j[0] in kinds]
    # Priority under a retirement deadline: short GPQA jobs first, then sample seeds in rounds
    # (all datasets get seed 0 before any gets seed 1), so a cut-off leaves balanced partial data.
    rank = {'gpqa_option_probs': 0, 'gpqa_letter': 1}
    jobs.sort(key=lambda j: (rank.get(j[0], 2), j[2], j[0]))
    built = []
    for kind, item, s, text, system, cfg in jobs:
        r = RawRunner(project, **cfg)
        req = Request(f'{kind}|{item}|{s}', text, system, {'kind': kind, 'item_id': item, 'sample': s})
        built.append((kind, item, s, r, req, r.cache_key(req)))
    abandoned = AbandonedRequests(SimpleNamespace(root=out))
    abandoned.settle_prior(cache, {k: {'kind': kind, 'item_id': item, 'sample': s} for kind, item, s, _, _, k in built})
    done = {row['request_key'] for row in read_jsonl(out/'collected.jsonl', tolerate_tail=True)} if (out/'collected.jsonl').exists() else set()
    stop = threading.Event(); lock = threading.Lock(); errors = []

    def one(job):
        kind, item, s, r, req, key = job
        if stop.is_set() or key in done:
            return
        row = {'kind': kind, 'item_id': item, 'sample': s, 'request_key': key}
        if key in abandoned.keys():
            row['status'] = 'service_failure'
        else:
            try:
                call, _ = cache.get(r, req, allow_api=allow_api)
            except OfflineCacheMiss:
                return
            except UncertainRequestError:
                with lock:
                    abandoned.abandon(cache, key, {'kind': kind, 'item_id': item, 'sample': s})
                row['status'] = 'service_failure'
            except (ProviderRejectedError, BudgetLimitError) as exc:
                stop.set()
                with lock:
                    errors.append({'job': req.key, 'type': type(exc).__name__, 'http_status': getattr(exc, 'status', None)})
                return
            else:
                pr = (call.meta or {}).get('provider_response') or {}
                fr = (pr.get('choices') or [{}])[0].get('finish_reason')
                row.update(status='ok', finish_reason=fr, **parse(kind, call.response))
        with lock:
            append_jsonl(out/'collected.jsonl', row)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, built))
    have = {row['request_key'] for row in read_jsonl(out/'collected.jsonl', tolerate_tail=True)} if (out/'collected.jsonl').exists() else set()
    return {'jobs': len(built), 'collected': sum(b[5] in have for b in built), 'errors': errors[:3], 'budget': cache.summary()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, default=Path('results/qwen_raw_20261021'))
    p.add_argument('--project', default='llm-applications-490420')
    p.add_argument('--n', type=int, default=10)
    p.add_argument('--limit-usd', type=float, default=30.0)
    p.add_argument('--kinds', nargs='*')
    p.add_argument('--allow-api', action='store_true')
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    jobs = stage1(a.project, a.n)
    print(json.dumps(run(a.out, jobs, project=a.project, limit_usd=a.limit_usd, allow_api=a.allow_api,
                         workers=a.workers, kinds=a.kinds), indent=2, default=str))


if __name__ == '__main__':
    main()
