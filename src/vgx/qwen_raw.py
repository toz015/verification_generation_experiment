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
from vgx.quantum_katas.runners import AnthropicVertexRunner, JsonModeVertexRunner

MODEL = 'qwen/qwen3-235b-a22b-instruct-2507-maas'
# Other generators (2026-10-22, owner: "start working on the other model's response generation").
# Settings follow each model's main experiments; token-only jobs get room for gpt-oss reasoning and
# keep logprobs, so the answer token's distribution is read after the final-channel marker.
PROFILES = {
    'qwen3': {'model': MODEL, 'location': 'us-south1', 'effort': 'low', 'short_max': 1, 'json_max': 512,
              'gpqa_max': 4096, 'math_max': 16384, 'qk_max': 8192,
              'top_k': 20, 'price': {'input': 0.22, 'output': 0.88, 'cached_input': 0.22}},
    'gptoss': {'model': 'openai/gpt-oss-120b-maas', 'location': 'us-central1', 'effort': 'medium', 'short_max': 4096,
               'json_max': 8192, 'gpqa_max': 32768, 'math_max': 32768, 'qk_max': 32768,
               'top_k': 5, 'price': {'input': 0.09, 'output': 0.36, 'cached_input': 0.09}},
    # Sonnet 5.5: main-experiment settings (adaptive thinking, effort high, 32768, structured output).
    # No token logprobs on this API and no reasoning in the reply (reasoning_extraction refusal), so only
    # stated probabilities are stored; PEG's token prompts are replaced by verbal option distributions.
    'sonnet55': {'model': 'anthropic/claude-sonnet-5-5', 'location': 'global', 'anthropic': True,
                 'price': {'input': 2.0, 'output': 10.0, 'cached_input': 2.0}},
}
P = PROFILES['qwen3']


def _schema(props):
    return {'type': 'object', 'properties': {k: {'type': t} for k, t in props.items()},
            'required': list(props), 'additionalProperties': False}


def sonnet_cfg(props, sample=0):
    return {'anthropic': {'max_tokens': 32768, 'thinking': {'type': 'adaptive'},
                          'output_config': {'effort': 'high', 'format': {'type': 'json_schema', 'schema': _schema(props)}}},
            'sample': sample}


class SonnetRawRunner(AnthropicVertexRunner):
    """Sonnet with the main-experiment body; `sample` and `attempt` only enter request identity."""

    def __init__(self, project, generation, sample=0, attempt=1):
        super().__init__(P['model'], P['location'], project, generation)
        self.sample, self.attempt = sample, attempt

    @property
    def identity(self):
        value = {**super().identity, 'raw_collection': 'raw_v1', 'sample': self.sample}
        return value if self.attempt == 1 else {**value, 'attempt': self.attempt}


def stage1_sonnet(n):
    jobs = []
    for it in _gpqa_items():
        body = _gpqa_body(it)
        jobs.append(('gpqa_option_probs', it['item_id'], 0, body +
                     'For each option, report your probability that it is the correct answer. The four probabilities '
                     'must sum to 1.\nReply with JSON only: {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}', GPQA_SYSTEM,
                     sonnet_cfg({l: 'number' for l in LETTERS})))
        jobs.append(('gpqa_option_probs_incorrect', it['item_id'], 0, body +
                     'For each option, report your probability of choosing it if you were asked to name an incorrect '
                     'answer. The four probabilities must sum to 1.\nReply with JSON only: {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}',
                     GPQA_SYSTEM, sonnet_cfg({l: 'number' for l in LETTERS})))
        gen = body + ('Choose the best answer. Report your probability that your selected answer is correct.\n'
                      'Reply with JSON only: {"answer": "A"|"B"|"C"|"D", "p_correct": 0.0}')
        for s in range(n):
            jobs.append(('gpqa_samples', it['item_id'], s, gen, GPQA_SYSTEM, sonnet_cfg({'answer': 'string', 'p_correct': 'number'}, s)))
    for it in _math_items():
        text = mprompt.build_generator_prompt(it['problem'], mprompt.PROMPT_VERSION_ANSWER_ONLY)
        for s in range(n):
            jobs.append(('math500_samples', it['item_id'], s, text, mprompt.GENERATOR_SYSTEM,
                         sonnet_cfg({'answer': 'string', 'p_correct': 'number'}, s)))
    for t in _qk_tasks():
        text = qprompt.build_generator_prompt(t, qprompt.PROMPT_VERSION_V2)
        for s in range(n):
            jobs.append(('qkatas_samples', t.task_id, s, text, qprompt.generator_system(qprompt.PROMPT_VERSION_V2),
                         sonnet_cfg({'code': 'string', 'p_correct': 'number'}, s)))
    return jobs


def configure(name):
    global P, PEG_TOKEN, JUDGE_JSON
    P = PROFILES[name]
    if P.get('anthropic'):
        return
    PEG_TOKEN = dict(max_tokens=P['short_max'], temperature=0.0, json_mode=False, extra={'logprobs': True, 'top_logprobs': P['top_k']})
    JUDGE_JSON = dict(max_tokens=P['json_max'], temperature=0.0, json_mode=True, extra={'logprobs': True, 'top_logprobs': 5})
LETTERS = 'ABCD'
GPQA_BUNDLE = 'results/gpqa_staged_generator_expansion_20261005/frozen_candidates.json'
QK_RUN = 'results/quantum_katas_full_qwen3_20261019'


class RawRunner(JsonModeVertexRunner):
    """Qwen chat completions with optional JSON mode and extra body params (seed, logprobs)."""

    def __init__(self, project, *, max_tokens, temperature, json_mode, extra, attempt=1):
        super().__init__(P['model'], P['location'], project, {'type': 'json_object'}, max_tokens=max_tokens,
                         temperature=temperature, top_p=1.0, reasoning_effort=P['effort'])
        self.json_mode, self.extra, self.attempt = json_mode, dict(extra), attempt

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
        value = {**value, 'extra': self.extra, 'raw_collection': 'qwen_raw_v1'}
        # attempt > 1 only re-sends a job whose first attempt was a service failure (owner request
        # 2026-10-21); it changes request identity, never the body sent to the provider.
        return value if self.attempt == 1 else {**value, 'attempt': self.attempt}


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


def _peg_subject(subject):
    return ''.join(' ' + e for e in subject.split('_'))  # PEG format_subject, verbatim


def peg_generator_prompt(subject, question, choices, get_correct):
    # PEG initial_policy_diffModel_GPQA.ipynb build_generator_prompt (toz015/neurips2025-repo@0ade29f), verbatim.
    prompt = 'The following are multiple choice questions (with answers) about {}.\n\n'.format(_peg_subject(subject))
    prompt += f'{question}'
    for i, c in enumerate(choices):
        prompt += '\n{}'.format(f'{chr(65+i)}. {c}')
    return prompt + ('\nAnswer:' if get_correct else '\nIncorrect Answer:')


def peg_discriminator_prompt(subject, question, proposed_answer):
    # PEG build_discriminator_prompt, verbatim.
    return f"""You are an expert evaluator of questions about {_peg_subject(subject)}. 
Determine if the proposed answer is correct. Output ONLY 'A' or 'B'.
Question: {question}
Proposed Answer: {proposed_answer}

Is this answer correct? Respond ONLY with:
A. Correct
B. Incorrect

Answer:
"""


PEG_TOKEN = dict(max_tokens=1, temperature=0.0, json_mode=False, extra={'logprobs': True, 'top_logprobs': 20})


def peg_gpqa():
    jobs = []
    for it in _gpqa_items():
        for flag, kind in ((True, 'gpqa_peg_gen_correct'), (False, 'gpqa_peg_gen_incorrect')):
            jobs.append((kind, it['item_id'], 0, peg_generator_prompt(it['subject'], it['question'], it['choices'], flag),
                         None, PEG_TOKEN))
        for i, c in enumerate(it['choices']):
            jobs.append(('gpqa_peg_disc', it['item_id'], i, peg_discriminator_prompt(it['subject'], it['question'], c),
                         None, PEG_TOKEN))
    return jobs


def stage1(project, n):
    jobs = peg_gpqa()
    for it in _gpqa_items():
        body = _gpqa_body(it)
        jobs.append(('gpqa_option_probs', it['item_id'], 0, body +
                     'For each option, report your probability that it is the correct answer. The four probabilities '
                     'must sum to 1.\nReply with JSON only: {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}', GPQA_SYSTEM,
                     dict(max_tokens=P['json_max'], temperature=0.0, json_mode=True, extra={'logprobs': True, 'top_logprobs': 5})))
        jobs.append(('gpqa_letter', it['item_id'], 0, body + 'Reply with the letter of the best answer only.', GPQA_SYSTEM,
                     dict(max_tokens=P['short_max'], temperature=0.0, json_mode=False, extra={'logprobs': True, 'top_logprobs': 5})))
        gen = body + ('Choose the best answer. Report your probability that your selected answer is correct.\n'
                      'Reply with JSON only: {"answer": "A"|"B"|"C"|"D", "p_correct": 0.0}')
        for s in range(n):
            # nolp variant for reasoning generators: logprobs would cover the whole reasoning trace.
            jobs.append(('gpqa_samples_nolp', it['item_id'], s, gen, GPQA_SYSTEM,
                         dict(max_tokens=P['gpqa_max'], temperature=0.7, json_mode=True, extra={'seed': s})))
            jobs.append(('gpqa_samples', it['item_id'], s, gen, GPQA_SYSTEM,
                         dict(max_tokens=P['gpqa_max'], temperature=0.7, json_mode=True,
                              extra={'seed': s, 'logprobs': True, 'top_logprobs': 5})))
    for it in _math_items():
        for s in range(n):
            # nolp: same sample without token logprobs. Long logprob replies stalled past 900 s on
            # 2026-10-21 (most recorded as uncertain), so completeness takes priority (owner).
            jobs.append(('math500_samples_nolp', it['item_id'], s, mprompt.build_generator_prompt(it['problem']),
                         mprompt.GENERATOR_SYSTEM, dict(max_tokens=P['math_max'], temperature=0.7, json_mode=True,
                                                        extra={'seed': s})))
            jobs.append(('math500_samples', it['item_id'], s, mprompt.build_generator_prompt(it['problem']),
                         mprompt.GENERATOR_SYSTEM, dict(max_tokens=P['math_max'], temperature=0.7, json_mode=True,
                                                        extra={'seed': s, 'logprobs': True})))
    for t in _qk_tasks():
        for s in range(n):
            jobs.append(('qkatas_samples_nolp', t.task_id, s, qprompt.build_generator_prompt(t, qprompt.PROMPT_VERSION_V2),
                         qprompt.generator_system(qprompt.PROMPT_VERSION_V2),
                         dict(max_tokens=P['qk_max'], temperature=0.7, json_mode=True, extra={'seed': s})))
            jobs.append(('qkatas_samples', t.task_id, s, qprompt.build_generator_prompt(t, qprompt.PROMPT_VERSION_V2),
                         qprompt.generator_system(qprompt.PROMPT_VERSION_V2),
                         dict(max_tokens=P['qk_max'], temperature=0.7, json_mode=True, extra={'seed': s, 'logprobs': True})))
    return jobs


# ---------------------------------------------------------------- stage 2: Qwen as judge
JUDGE_JSON = dict(max_tokens=512, temperature=0.0, json_mode=True, extra={'logprobs': True, 'top_logprobs': 5})
MATH_RUNS = {g: f'results/math500_full_{g}_20261021' for g in ('qwen3', 'gptoss', 'sonnet55')}
QK_RUNS = {g: f'results/quantum_katas_full_{g}_20261019' for g in ('qwen3', 'gptoss', 'sonnet55')}


def _samples(paths, kind):
    """OK sample rows of `kind` and its no-logprob variant from every collected.jsonl in paths (comma-separated)."""
    rows = []
    for path in str(paths).split(','):
        if Path(path).exists():
            rows += [json.loads(l) for l in open(path)]
    return [r for r in rows if r['kind'] in (kind, kind + '_nolp') and r.get('status') == 'ok']


def math_candidates(samples_path, interpreter):
    """Distinct answers per MATH problem (Qwen samples + frozen answers of three generators), clustered
    with the official grader; candidates.json records every source so nothing is lost."""
    import subprocess
    from vgx.math_trial import data
    by = {it['item_id']: [] for it in _math_items()}
    for r in _samples(samples_path, 'math500_samples'):
        if isinstance(r.get('answer'), str) and r['answer'].strip():
            by[r['item_id']].append({'answer': r['answer'], 'source': f"qwen_{r['kind']}_{r['sample']}"})
    for g, d in MATH_RUNS.items():
        for r in json.loads((Path(d)/'frozen_candidates.json').read_text())['rows']:
            if r.get('answer') is not None:
                by[r['item_id']].append({'answer': r['answer'], 'source': f'frozen_{g}'})
    payload = ''.join(json.dumps({'item_id': k, 'answers': [c['answer'] for c in v]})+'\n' for k, v in by.items())
    grader_dir = str(Path(data.SOURCES['omnimath_rule']['path']).parent/'evaluation')
    proc = subprocess.run([interpreter, '-I', str(Path(__file__).parent/'math_trial'/'cluster_worker.py'), grader_dir],
                          input=payload, capture_output=True, text=True, timeout=7200)
    if proc.returncode:
        raise RuntimeError(proc.stderr[-2000:])
    out = {}
    for line in proc.stdout.splitlines():
        r = json.loads(line); cands = by[r['item_id']]
        out[r['item_id']] = [{'answer': cands[c[0]]['answer'], 'sources': [cands[i]['source'] for i in c],
                              'members': [cands[i]['answer'] for i in c]} for c in r['clusters']]
    return out


def qk_candidates(samples_path):
    tasks = {t.task_id: t for t in _qk_tasks()}
    by = {t: {} for t in tasks}
    for r in _samples(samples_path, 'qkatas_samples'):
        if r.get('code'):
            by[r['item_id']].setdefault(r['code'], []).append(f"qwen_{r['kind']}_{r['sample']}")
    for g, d in QK_RUNS.items():
        for c in Run(d).candidates.all():
            if c.has_code:
                by[c.task_id].setdefault(c.raw_code, []).append(f'frozen_{g}')
    return {t: [{'code': code, 'sources': src} for code, src in v.items()] for t, v in by.items()}


def _judge_cfg():
    return sonnet_cfg({'p_correct': 'number'}) if P.get('anthropic') else JUDGE_JSON


def stage2(samples_path, candidates_dir, interpreter):
    from vgx.gpqa.prompt import build_verifier_prompt as gpqa_verifier_text
    from vgx.math_trial import data
    jobs = []
    cdir = Path(candidates_dir); cdir.mkdir(parents=True, exist_ok=True)
    # GPQA: Qwen as verifier on each option with the GPQA verifier prompt (PEG A/B already in stage 1).
    for it in _gpqa_items():
        body = _gpqa_body(it)
        for i, letter in enumerate(LETTERS):
            text = (body + f"Fixed candidate answer: {letter}. {it['choices'][i]}\n\n"
                    'Assess whether this fixed answer is correct. Do not suggest a replacement answer.\n'
                    'Reply with JSON only: {"p_correct": 0.0}')
            jobs.append(('gpqa_judge_p', it['item_id'], i, text, 'You evaluate whether a proposed answer to a '
                         'graduate-level multiple-choice science question is correct.', _judge_cfg()))
    # MATH: clustered candidates -> PEG discriminator, verbal judge, and (>=2 candidates) option distributions.
    path = cdir/'math500_candidates.json'
    if path.exists():
        mc = json.loads(path.read_text())
    else:
        mc = math_candidates(samples_path, interpreter); path.write_text(json.dumps(mc))
    meta = {it['item_id']: it for it in data.load('math500')[0]}
    for item_id, cands in mc.items():
        prob, subject = meta[item_id]['problem'], meta[item_id]['topic']
        for i, c in enumerate(cands):
            jobs.append(('math500_peg_disc', item_id, i, peg_discriminator_prompt(subject, prob, c['answer']), None, PEG_TOKEN))
            jobs.append(('math500_judge_p', item_id, i, mprompt.build_verifier_prompt(prob, c['answer']),
                         mprompt.VERIFIER_SYSTEM, _judge_cfg()))
        if 2 <= len(cands) <= 20:
            answers = [c['answer'] for c in cands]
            for flag, kind in ((True, 'math500_peg_gen_correct'), (False, 'math500_peg_gen_incorrect')):
                jobs.append((kind, item_id, 0, peg_generator_prompt(subject, prob, answers, flag), None, PEG_TOKEN))
    # QuantumKatas: distinct code candidates -> PEG discriminator and the QK verifier prompt (v2).
    qc = qk_candidates(samples_path)
    (cdir/'qkatas_candidates.json').write_text(json.dumps(qc))
    tasks = {t.task_id: t for t in _qk_tasks()}
    for tid, cands in qc.items():
        t = tasks[tid]
        for i, c in enumerate(cands):
            view = {'task_id': tid, 'prompt': t.prompt, 'entry_point': t.entry_point,
                    'required_functions': list(t.required_functions), 'code': c['code']}
            jobs.append(('qkatas_peg_disc', tid, i, peg_discriminator_prompt('quantum computing', t.prompt, c['code']),
                         None, PEG_TOKEN))
            jobs.append(('qkatas_judge_p', tid, i, qprompt.build_verifier_prompt(view, qprompt.PROMPT_VERSION_V2),
                         qprompt.verifier_system(qprompt.PROMPT_VERSION_V2), _judge_cfg()))
    return jobs


def retry_jobs(jobs, collected_paths, attempt=2):
    """Second attempt for every job whose only row is a service failure (same prompt and body)."""
    rows = []
    for path in collected_paths:
        if Path(path).exists():
            rows += [json.loads(l) for l in open(path)]
    ok = {(r['kind'], r['item_id'], r['sample']) for r in rows if r.get('status') == 'ok'}
    failed = {(r['kind'], r['item_id'], r['sample']) for r in rows if r.get('status') == 'service_failure'} - ok
    return [(k, i, s, t, sy, dict(cfg, attempt=attempt)) for k, i, s, t, sy, cfg in jobs if (k, i, s) in failed]


def parse(kind, text):
    try:
        if kind == 'gpqa_letter' or '_peg_' in kind:
            return {'letter': text.strip()[:1] or None}
        if kind.endswith('_yesno'):
            return {'token': text.strip()}
        value = json.loads(text)
        if isinstance(value, dict) and set(value) == {'final'} and isinstance(value['final'], str):
            # gpt-oss sometimes wraps its JSON in {"final": "<json>"}; unwrap for raw-data parsing only
            # (the main experiments' strict parser is unchanged; the response itself is stored as is).
            value = dict(json.loads(value['final']), _unwrapped_final=True)
        if kind.startswith('gpqa_option_probs'):
            return {'probs': {l: value.get(l) for l in LETTERS}}
        if kind.startswith('qkatas_samples'):
            p = qprompt.parse_generator_response(text)
            return {'code': p.code, 'p_correct': p.p_correct, 'failure': p.failure}
        if kind.endswith('_judge_p'):
            return {'p_correct': value.get('p_correct')}
        return {'answer': value.get('answer'), 'p_correct': value.get('p_correct'), **({'unwrapped_final': True} if value.get('_unwrapped_final') else {})}
    except Exception as exc:
        return {'parse_error': type(exc).__name__}


def run(out, jobs, *, project, limit_usd, allow_api=False, workers=8, kinds=None):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    pricing = {'usd_per_million_tokens': {P['model']: P['price']}}
    if P.get('anthropic'):
        pricing['usage_semantics'] = {P['model']: 'completion_includes_reasoning'}
    cache = BudgetedRequestCache(out/'cache', project=project, pricing=pricing, limit_usd=limit_usd)
    jobs = [j for j in jobs if kinds is None or j[0] in kinds]
    # Priority under a retirement deadline: short GPQA jobs first, then sample seeds in rounds
    # (all datasets get seed 0 before any gets seed 1), so a cut-off leaves balanced partial data.
    rank = {'gpqa_option_probs': 0, 'gpqa_letter': 1, 'gpqa_peg_gen_correct': 1, 'gpqa_peg_gen_incorrect': 1, 'gpqa_peg_disc': 1}
    jobs.sort(key=lambda j: (rank.get(j[0], 2), j[2], j[0]))
    built = []
    for kind, item, s, text, system, cfg in jobs:
        if 'anthropic' in cfg:
            r = SonnetRawRunner(project, cfg['anthropic'], cfg.get('sample', 0), cfg.get('attempt', 1))
        else:
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
                    errors.append({'job': req.key, 'type': type(exc).__name__, 'http_status': getattr(exc, 'status', None), 'message': str(exc)[:200]})
                return
            else:
                pr = (call.meta or {}).get('provider_response') or {}
                fr = (pr.get('choices') or [{}])[0].get('finish_reason') if 'choices' in pr else pr.get('stop_reason')
                row.update(status='ok', finish_reason=fr, **parse(kind, call.response))
        with lock:
            append_jsonl(out/'collected.jsonl', row)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, built))
    have = {row['request_key'] for row in read_jsonl(out/'collected.jsonl', tolerate_tail=True)} if (out/'collected.jsonl').exists() else set()
    return {'jobs': len(built), 'collected': sum(b[5] in have for b in built), 'errors': errors[:3], 'budget': cache.summary()}


def _long_read_timeout():
    """This CLI only: 900 s read timeout instead of the runners' 300 s. Long MATH samples with logprobs
    exceeded 300 s while the endpoint was congested (2026-10-21) and were recorded as uncertain."""
    import requests as _requests
    original = _requests.Session.post
    def post(session, *args, **kwargs):
        kwargs['timeout'] = 900
        return original(session, *args, **kwargs)
    _requests.Session.post = post


def main():
    import sys
    configure(sys.argv[sys.argv.index('--model')+1] if '--model' in sys.argv else 'qwen3')
    _long_read_timeout()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, default=Path('results/qwen_raw_20261021'))
    p.add_argument('--project', default='llm-applications-490420')
    p.add_argument('--n', type=int, default=10)
    p.add_argument('--limit-usd', type=float, default=30.0)
    p.add_argument('--kinds', nargs='*')
    p.add_argument('--allow-api', action='store_true')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--stage', type=int, default=1)
    p.add_argument('--model', default='qwen3', choices=sorted(PROFILES))
    p.add_argument('--retry-from', help='comma-separated collected.jsonl files whose service failures to re-send once')
    p.add_argument('--samples', default='results/qwen_raw_20261021/samples/collected.jsonl,'
                                       'results/qwen_raw_20261021/samples_nolp/collected.jsonl')
    p.add_argument('--interpreter', default='tmp/math_grader_venv_20261020/bin/python')
    a = p.parse_args()
    if a.stage == 1:
        jobs = stage1_sonnet(a.n) if P.get('anthropic') else stage1(a.project, a.n)
    else:
        jobs = stage2(a.samples, a.out/'candidates', a.interpreter)
        if P.get('anthropic'):
            jobs = [j for j in jobs if 'judge_p' in j[0]]
    if a.retry_from:
        jobs = retry_jobs(jobs, a.retry_from.split(','))
    print(json.dumps(run(a.out, jobs, project=a.project, limit_usd=a.limit_usd, allow_api=a.allow_api,
                         workers=a.workers, kinds=a.kinds), indent=2, default=str))


if __name__ == '__main__':
    main()
