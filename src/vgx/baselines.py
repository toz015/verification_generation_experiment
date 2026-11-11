"""Baselines for releasing a fixed generator answer, on GPQA, MATH-500 and QuantumKatas (no API calls).

Every method decides, for the main experiments' frozen answer of each generator, whether to release
it or abstain, on the same splits as our method: GPQA fits on the 249 calibration items and is scored
on the 196 Diamond items; MATH-500 and QuantumKatas use the same 5-fold cross-fitting. Utility is the
main experiments' (+1 correct release, -L wrong release, L=19, minus 10 utility units per expected
USD, expected cost = training-fold mean cost of each call made). Anything a method fits (likelihoods,
thresholds, label-model class balance, the choice of verifiers) uses training folds only.

  ours                adaptive sequential verification (unchanged planner, selected triple)
  generator_only      release iff the stated p_correct >= L/(1+L)
  fixed_k             always ask the first k verifiers of the selected triple (k=1,2,3) or all five,
                      same belief update and release rule as ours
  self_consistency    share of the generator's 10 raw samples that agree with the frozen answer (GPQA
                      letter, MATH pinned math_equal); release iff share >= a training-fit threshold;
                      cost = the 10 samples. Not defined for code (QuantumKatas)
  weaver              Weaver's weak-supervision label model (binary votes at 0.5, conditionally
                      independent, class balance from training labels, accuracies fit without labels
                      by EM on all candidates) over the selected triple (main) or all five (appendix);
                      release iff the posterior >= a training-fit threshold
  ssv_oracle          SSV (Kiyani et al.): one weak verifier (the verifier with the best single-verifier
                      training J0) with accept/reject thresholds updated online
                      from strong-verifier feedback; strong = ground truth, as in the paper (reported
                      with its number of oracle queries; oracle cost not priced)
  ssv_model           the same with the strongest other verifier (training AUROC) as strong verifier
  peg                 Peer Elicitation Game (Chen et al.) with three LLM discriminators (Qwen3, gpt-oss,
                      Sonnet) on the frozen answer; release iff enough discriminators end the game
                      agreeing with it (training-fit threshold)
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from vgx.common.storage import atomic_json, read_jsonl
from vgx.gpqa.dependence_triples import bin_of, execute
from vgx.gpqa.raw_prior_analysis import baseline_decision
from vgx.gpqa.score import binary_forecast_metrics
from vgx.gpqa.sequential import unseal
from vgx.quantum_katas import crossfit, crossfit_extended

OUT = Path('results/baselines_20261027')
EXPORT = Path('reports/raw_generator_export_20261026')
VERIFIERS = ['gemini_flash', 'gemini_lite', 'mistral_small', 'mistral_medium', 'claude_haiku']
EDGES = (0.0, 1/3, 2/3, 1.0)
LOSS = 19.0
# Gemini 3.8 Flash is not used as a generator: too close to the Gemini verifiers.
GPQA_STUDIES = {'qwen3': 'results/gpqa_qwen_verification_20261020',
                'gptoss': 'results/gpqa_gptoss_verification_regen_20261020',
                'sonnet55': 'results/gpqa_sonnet55_verification_20261016'}
CONFIGS = {'gpqa': 'configs/gpqa_gptoss_verification_20261020.json', 'math500': 'configs/math500_crossfit_complete_20261021.json',
           'qkatas': 'configs/quantum_katas_crossfit_3gen_20261019.json'}
RAW = {'qwen3': 'results/qwen_raw_20261021', 'gptoss': 'results/raw_gptoss_20261022',
       'sonnet55': 'results/raw_sonnet55_20261026'}
GRADER_PY = 'tmp/math_grader_venv_20261020/bin/python'


# ----------------------------------------------------------------------------- data

def load(dataset):
    """{generator: rows}; each row as in the main analyses plus item_id and the frozen answer key."""
    cfg = json.loads(Path(CONFIGS[dataset]).read_text())
    out = {}
    if dataset == 'gpqa':
        from vgx.gpqa.verification_summary import load_rows
        for g, study in GPQA_STUDIES.items():
            rows, _ = load_rows(study)
            answers = {}
            for part in ('calibration', 'evaluation'):
                v = json.loads((Path(study)/part/'signals.json').read_text())
                answers.update({r['item_id']: r['answer'] for r in v['rows']})
            out[g] = [dict(r, answer=answers[r['item_id']]) for r in rows]
    elif dataset == 'math500':
        from vgx.math_trial.analysis import _ledger, load_rows
        split = json.loads(Path(cfg['split']).read_text())
        signals = json.loads((Path(cfg['verification'])/'signals.json').read_text()); unseal(signals, 'signals_id')
        vl = _ledger(cfg['verification'])
        for g, key in (('qwen3', 'qwen3complete'), ('gptoss', 'gptoss'), ('sonnet55', 'sonnet55')):
            d = cfg['generators'][key]
            rows = load_rows(d, key, signals, vl, split, cfg['verifiers'])
            bundle = json.loads((Path(d)/'frozen_candidates.json').read_text())['rows']
            out[g] = [dict(r, item_id=r['task_id'], answer=c.get('answer')) for r, c in zip(rows, bundle)]
    else:
        from vgx.common.storage import file_digest
        from vgx.quantum_katas.run_pilot import Run
        split = {'value': json.loads(Path(cfg['split']).read_text()), 'sha256': file_digest(cfg['split'])}
        for g, root in cfg['runs'].items():
            rows = crossfit.load_rows(root, split, cfg['verifiers'])
            cands = Run(root).candidates.by_task()
            out[g] = [dict(r, item_id=r['task_id'], answer=(cands[r['task_id']].code_sha256
                      if r['task_id'] in cands and cands[r['task_id']].has_code else None)) for r in rows]
    return out, cfg


def splits(dataset, rows):
    if dataset == 'gpqa':
        return [([i for i, r in enumerate(rows) if r['fold'] == 1], [i for i, r in enumerate(rows) if r['fold'] == 0])]
    folds = sorted({r['fold'] for r in rows})
    return [([i for i, r in enumerate(rows) if r['fold'] != k], [i for i, r in enumerate(rows) if r['fold'] == k]) for k in folds]


def decision(action, used=0, expected=0.0, usage=0.0, score=None, **extra):
    return {'action': action, 'verifiers_used': used, 'expected_cost_usd': float(expected),
            'usage_estimated_cost_usd': float(usage), 'failure': None, 'posterior': score, **extra}


def fit_threshold(scores, ys, loss=LOSS):
    """Largest-utility release threshold on training scores (ties -> the higher threshold)."""
    pairs = sorted(((s, y) for s, y in zip(scores, ys) if s is not None), key=lambda p: -p[0])
    best, best_t, gain, i = 0.0, math.inf, 0.0, 0
    while i < len(pairs):
        s = pairs[i][0]
        while i < len(pairs) and pairs[i][0] == s:
            gain += 1.0 if pairs[i][1] else -loss; i += 1
        if gain > best + 1e-12:
            best, best_t = gain, s
    return best_t


def release_by_score(rows, idx, scores, tau, expected, usage, **extra):
    return [decision('assert' if rows[i]['valid'] and scores[i] is not None and scores[i] >= tau else 'abstain',
                     extra.get('used', 0), expected, usage[i], scores[i]) for i in idx]


# ----------------------------------------------------------------------------- ours + fixed budgets

def ours_and_fixed(rows, train_idx, test_idx, settings):
    train = [rows[i] for i in train_idx]
    fitted = crossfit.fit_fold(train, VERIFIERS, settings)
    triple = crossfit.select_order(train, fitted, VERIFIERS, LOSS, settings)[0]['order']
    out = {'generator_only': [baseline_decision(rows[i]['prior'] if rows[i]['valid'] else 0., rows[i]['valid'], LOSS,
                                                settings['correct_reward']) for i in test_idx]}
    planner, costs = crossfit.planner_for(fitted, triple, LOSS, settings)
    out['ours'] = [crossfit.decide(rows[i], planner, triple, costs, EDGES) for i in test_idx]
    for name, order in (('fixed_1', triple[:1]), ('fixed_2', triple[:2]), ('fixed_3', triple), ('fixed_5', VERIFIERS)):
        planner, costs = crossfit.planner_for(fitted, order, LOSS, settings)
        out[name] = [execute(rows[i]['prior'] if rows[i]['valid'] else 0., rows[i]['valid'], planner, order, costs,
                             {v: rows[i]['signals'][v] for v in order}, EDGES, force_all=True) for i in test_idx]
    return out, {'triple': triple, 'fitted': fitted}


# ----------------------------------------------------------------------------- self-consistency

def _raw_samples(dataset):
    """{(generator, item_id): [(sample, answer, usd)]} from the raw collections (no-logprob variant first)."""
    from vgx.raw_generator_export import GENERATORS, collected, recover, response
    prefix = {'gpqa': 'gpqa_samples', 'math500': 'math500_samples'}[dataset]
    out = defaultdict(dict)
    for g, root in GENERATORS.items():
        ledgers = {}
        for r in collected(root):
            if not r['kind'].startswith(prefix) or r.get('status') != 'ok':
                continue
            key = (g, r['item_id']); s = int(r['sample'])
            if s in out[key] and not r['kind'].endswith('_nolp'):
                continue
            if r['_dir'] not in ledgers:
                p = r['_dir']/'cache'/'budget.json'
                ledgers[r['_dir']] = json.loads(p.read_text())['requests'] if p.exists() else {}
            ans = r.get('answer')
            if ans is None and (text := response(r)):
                ans = recover(text, dataset)[0]
            out[key][s] = (ans, (ledgers[r['_dir']].get(r['request_key']) or {}).get('estimated_usd') or 0.0)
    return out


def _math_equivalence(pairs):
    path = OUT/'math500_sc_equivalence.json'
    known = json.loads(path.read_text()) if path.exists() else {}
    todo = sorted({p for p in pairs if '\t'.join(p) not in known})
    if todo:
        from vgx.math_trial import data
        grader_dir = str(Path(data.SOURCES['omnimath_rule']['path']).parent/'evaluation')
        payload = ''.join(json.dumps({'item_id': '\t'.join(p), 'answer': p[2], 'reference': p[1]}) + '\n' for p in todo)
        proc = subprocess.run([GRADER_PY, '-I', 'src/vgx/math_trial/grade_worker.py', grader_dir], input=payload,
                              capture_output=True, text=True, timeout=36000)
        if proc.returncode:
            raise RuntimeError(proc.stderr[-2000:])
        for g in map(json.loads, proc.stdout.splitlines()):
            known[g['item_id']] = bool(g['correct'])
        OUT.mkdir(parents=True, exist_ok=True); atomic_json(path, known)
    return {tuple(k.split('\t')): v for k, v in known.items()}


def self_consistency_scores(dataset, gen, rows, samples):
    scores, usage = [], []
    if dataset == 'math500':
        pairs = {(r['item_id'], r['answer'], a) for r in rows if r['answer'] is not None
                 for a, _ in samples.get((gen, r['item_id']), {}).values() if a is not None}
        equiv = _math_equivalence(pairs)
    for r in rows:
        ss = samples.get((gen, r['item_id']), {})
        usage.append(sum(u for _, u in ss.values()))
        if r['answer'] is None or not ss:
            scores.append(None); continue
        if dataset == 'gpqa':
            agree = [a == r['answer'] for a, _ in ss.values()]
        else:
            agree = [a is not None and (a == r['answer'] or equiv.get((r['item_id'], r['answer'], a), False)) for a, _ in ss.values()]
        scores.append(float(np.mean(agree)))
    return scores, usage


def self_consistency(rows, train_idx, test_idx, scores, usage):
    tau = fit_threshold([scores[i] for i in train_idx if rows[i]['valid']], [rows[i]['outcome'] for i in train_idx if rows[i]['valid']])
    expected = float(np.mean([usage[i] for i in train_idx]))
    return release_by_score(rows, test_idx, scores, tau, expected, usage), {'threshold': None if math.isinf(tau) else tau}


def vote_rows(dataset, gen, rows, samples):
    """Self-consistency as answer selection: the majority answer of the 10 raw samples, with the
    generator's own confidence (mean stated p_correct of the samples in the majority) as its prior.
    Returns rows with answer / prior / outcome / valid replaced, and the per-item cost of the 10 samples.
    MATH answers are clustered by the pinned math_equal (exact string first); ties go to the cluster
    with the higher total p_correct."""
    export = defaultdict(list)
    with gzip.open(EXPORT/f'{dataset}.jsonl.gz', 'rt') as f:
        for line in f:
            r = json.loads(line)
            if r['generator'] == gen and isinstance(r.get('answer'), str) and r['answer'].strip():
                export[r['item_id']].append(r)
    if dataset == 'math500':
        pairs = set()
        for item, ss in export.items():
            distinct = sorted({s['answer'] for s in ss})
            pairs |= {(item, a, b) for k, a in enumerate(distinct) for b in distinct[k+1:]}
        equiv = _math_equivalence(pairs)
    out, usage = [], []
    for r in rows:
        ss = export.get(r['item_id'], [])
        usage.append(sum(u for _, u in samples.get((gen, r['item_id']), {}).values()))
        clusters = []
        for s in ss:
            for c in clusters:
                a, b = sorted((c[0]['answer'], s['answer']))
                if a == b or (dataset == 'math500' and equiv.get((r['item_id'], a, b), False)):
                    c.append(s); break
            else:
                clusters.append([s])
        if not clusters:
            out.append(dict(r, answer=None, prior=0.0, outcome=0, valid=False)); continue
        best = max(clusters, key=lambda c: (len(c), sum(s['p_correct'] or 0.0 for s in c)))
        ps = [s['p_correct'] for s in best if s['p_correct'] is not None]
        labels = [s['correct'] for s in best if s['correct'] is not None]
        out.append(dict(r, answer=best[0]['answer'], prior=float(np.mean(ps)) if ps else 0.0,
                        outcome=int(labels[0]) if labels else 0, valid=bool(ps), vote_share=len(best)/len(ss)))
    return out, usage


def self_consistency_vote(vrows, test_idx, usage, train_idx, settings):
    """Release the majority answer iff its confidence >= L/(1+L), the generator-only rule."""
    expected = float(np.mean([usage[i] for i in train_idx]))
    return [dict(baseline_decision(vrows[i]['prior'], vrows[i]['valid'], LOSS, settings['correct_reward']),
                 expected_cost_usd=expected, usage_estimated_cost_usd=usage[i]) for i in test_idx]


# ----------------------------------------------------------------------------- Weaver

def weaver_posterior(votes, class_balance, iters=500):
    """Conditionally independent binary label model; class balance fixed, accuracies by EM (no labels)."""
    x = np.asarray(votes, float); m = x.shape[1]
    tpr, tnr = np.full(m, .7), np.full(m, .7)
    pi = min(max(class_balance, 1e-3), 1-1e-3)
    for _ in range(iters):
        l1 = math.log(pi) + (x*np.log(tpr) + (1-x)*np.log(1-tpr)).sum(1)
        l0 = math.log(1-pi) + (x*np.log(1-tnr) + (1-x)*np.log(tnr)).sum(1)
        q = 1/(1+np.exp(l0-l1))
        new_tpr = np.clip((q[:, None]*x).sum(0)/max(q.sum(), 1e-9), .01, .99)
        new_tnr = np.clip(((1-q)[:, None]*(1-x)).sum(0)/max((1-q).sum(), 1e-9), .01, .99)
        if np.allclose(new_tpr, tpr, atol=1e-8) and np.allclose(new_tnr, tnr, atol=1e-8):
            break
        tpr, tnr = new_tpr, new_tnr
    return q, tpr, tnr


def weaver_binned_posterior(bins, class_balance, k, iters=500):
    """Weaver's label model with each verifier's score in k bins (missing = -1, skipped): class-conditional
    bin distributions per verifier, conditionally independent, fit by EM without labels. class_balance is
    the prior P(correct): one fixed number, or one per item (the generator's stated p_correct)."""
    x = np.asarray(bins, int); n, m = x.shape
    onehot = np.stack([(x == b) for b in range(k)], -1).astype(float)          # n, m, k
    ramp = np.linspace(1, 2, k)
    p1 = np.tile(ramp/ramp.sum(), (m, 1)); p0 = np.tile(ramp[::-1]/ramp.sum(), (m, 1))
    pi = np.clip(np.broadcast_to(np.asarray(class_balance, float), (n,)), 1e-3, 1-1e-3)
    for _ in range(iters):
        l1 = np.log(pi) + (onehot*np.log(p1)).sum((1, 2))
        l0 = np.log(1-pi) + (onehot*np.log(p0)).sum((1, 2))
        q = 1/(1+np.exp(l0-l1))
        new1 = (q[:, None, None]*onehot).sum(0) + 1e-2; new1 /= new1.sum(1, keepdims=True)
        new0 = ((1-q)[:, None, None]*onehot).sum(0) + 1e-2; new0 /= new0.sum(1, keepdims=True)
        if np.allclose(new1, p1, atol=1e-8) and np.allclose(new0, p0, atol=1e-8):
            break
        p1, p0 = new1, new0
    return q, p1, p0


def weaver(rows, train_idx, test_idx, pool, fitted, binned=True, with_prior=False, fit_tau=False):
    """binned: scores in the main experiments' three probability bins (main table); else binary votes at 0.5.
    with_prior: the generator's stated p_correct is each item's prior in place of the training class balance.
    Release iff the posterior >= L/(1+L), the rule of ours; fit_tau: a training-fit threshold instead (appendix)."""
    idx = sorted(set(train_idx) | set(test_idx))
    use = [i for i in idx if rows[i]['valid']]
    cb = float(np.mean([rows[i]['outcome'] for i in train_idx if rows[i]['valid']]))
    if binned:
        bins = [[-1 if rows[i]['signals'][v]['score'] is None else bin_of(rows[i]['signals'][v]['score'], EDGES)
                 for v in pool] for i in use]
        prior = [rows[i]['prior'] for i in use] if with_prior else cb
        q, tpr, tnr = weaver_binned_posterior(bins, prior, len(EDGES)-1)
        tpr, tnr = tpr.tolist(), tnr.tolist()
    else:
        votes = [[1.0 if (rows[i]['signals'][v]['score'] or 0.0) >= 0.5 else 0.0 for v in pool] for i in use]
        q, tpr, tnr = weaver_posterior(votes, cb)
        tpr, tnr = tpr.tolist(), tnr.tolist()
    scores = [None]*len(rows)
    for i, s in zip(use, q):
        scores[i] = float(s)
    tau = fit_threshold([scores[i] for i in train_idx if rows[i]['valid']], [rows[i]['outcome'] for i in train_idx if rows[i]['valid']]) \
        if fit_tau else LOSS/(1+LOSS)
    expected = sum(fitted['expected_cost_usd'][v] for v in pool)
    usage = [sum(r['signals'][v]['estimated_usd'] for v in pool) for r in rows]
    ds = [dict(d, verifiers_used=len(pool) if rows[i]['valid'] else 0) for d, i in
          zip(release_by_score(rows, test_idx, scores, tau, expected, usage), test_idx)]
    return ds, {'threshold': None if math.isinf(tau) else tau, 'class_balance': cb, 'tpr': dict(zip(pool, tpr)), 'tnr': dict(zip(pool, tnr)),
                'scores': scores}


# ----------------------------------------------------------------------------- SSV

class SSV:
    """SimulatedAdaptiveRunner from weak-and-strong-verification (src/algorithms/adaptive.py), one attempt
    per problem; defaults from the repository except alpha, set to the target error 1/(1+L)."""

    def __init__(self, alpha, beta=0.10, eta=0.1, eta_R=0.1, P_a_init=0.3, P_r_init=0.3, P_min=0.05, seed=0):
        self.alpha, self.beta, self.eta, self.eta_R = alpha, beta, eta, eta_R
        self.tau_A, self.tau_R = 1-alpha, beta
        self.P_a_init, self.P_r_init, self.P_min = P_a_init, P_r_init, P_min
        self.t = 0; self.rng = np.random.default_rng(seed)

    def P_a(self, t):
        return max(self.P_min, self.P_a_init/np.sqrt(t+1))

    def P_r(self, t):
        return max(self.P_min, self.P_r_init/np.sqrt(t+1))

    def update(self, w, H):
        t = self.t; bad, good = (0 if H else 1), (1 if H else 0)
        if w > self.tau_A:
            g, gb = bad*(1-self.alpha)/self.P_a(t), good*self.beta/self.P_a(t)
        elif w > self.tau_R:
            g, gb = bad*(-self.alpha), good*self.beta
        else:
            g, gb = bad*(-self.alpha)/self.P_r(t), good*(-(1-self.beta))/self.P_r(t)
        self.tau_A = max(self.tau_R+0.05, min(1.0, self.tau_A + self.eta*g))
        self.tau_R = max(0.0, min(self.tau_A-0.05, self.tau_R + self.eta_R*gb))

    def step(self, w, strong):
        """Returns (accept, queried). strong() gives the strong verifier's verdict when queried."""
        self.t += 1
        if w is None:  # no weak score: ask the strong verifier, no threshold update
            return strong(), True
        if w > self.tau_A:
            accept, queried = True, self.rng.random() < self.P_a(self.t)
        elif w > self.tau_R:
            H = strong(); self.update(w, H)
            return H, True
        else:
            accept, queried = False, self.rng.random() < self.P_r(self.t)
        if queried:
            self.update(w, strong())
        return accept, queried


def ssv(rows, train_idx, test_idx, weak, strong, fitted, seed=0):
    """Warm up online on the training stream, then continue online on the test stream."""
    runner = SSV(alpha=1/(1+LOSS), seed=seed)
    rng = np.random.default_rng(seed)
    out = {}
    for phase, idx in (('train', list(rng.permutation(train_idx))), ('test', list(rng.permutation(test_idx)))):
        for i in idx:
            r = rows[i]
            if not r['valid']:
                if phase == 'test':
                    out[i] = decision('abstain')
                continue
            if strong == 'oracle':
                verdict = lambda r=r: bool(r['outcome'])
            else:
                verdict = lambda r=r: (r['signals'][strong]['score'] or 0.0) >= 0.5
            accept, queried = runner.step(r['signals'][weak]['score'], verdict)
            if phase == 'test':
                calls = [weak] + ([strong] if queried and strong != 'oracle' else [])
                out[i] = decision('assert' if accept else 'abstain', len(calls), sum(fitted['expected_cost_usd'][v] for v in calls),
                                  sum(r['signals'][v]['estimated_usd'] for v in calls), oracle_queries=int(queried and strong == 'oracle'))
    return [out[i] for i in test_idx], {'weak': weak, 'strong': strong, 'final_tau_A': runner.tau_A, 'final_tau_R': runner.tau_R}


def auroc(rows, idx, v):
    pairs = [(rows[i]['outcome'], rows[i]['signals'][v]['score']) for i in idx if rows[i]['valid'] and rows[i]['signals'][v]['score'] is not None]
    return binary_forecast_metrics([a for a, _ in pairs], [b for _, b in pairs]).auroc



# ----------------------------------------------------------------------------- PEG

PEG_DISCRIMINATORS = ('qwen3', 'gptoss', 'sonnet55')
PEG_CANDIDATES = {'qwen3': 'results/qwen_raw_20261021/stage2/candidates', 'gptoss': 'results/raw_gptoss_20261022/stage2/candidates',
                  'sonnet55': 'results/raw_sonnet55_20261026/stage2/candidates'}


def _ab_probs(record):
    """P(A), P(B) at the answer token: Qwen's single token, or gpt-oss's last A/B token in its final reply."""
    content = ((((record.get('meta') or {}).get('provider_response') or {}).get('choices') or [{}])[0].get('logprobs') or {}).get('content') or []
    for tok in reversed(content):
        if tok['token'].strip().strip('*:. ') in ('A', 'B'):
            mass = {'A': 0.0, 'B': 0.0}
            for alt in tok.get('top_logprobs') or [tok]:
                t = alt['token'].strip().strip('*:. ')
                if t in mass:
                    mass[t] += math.exp(alt['logprob'])
            if mass['A'] + mass['B'] > 0:
                return mass['A'], mass['B']
    return None


def peg_policies(dataset):
    """{discriminator: {item_id: [(option_key, p_correct, p_incorrect, usd)]}}; option_key is the GPQA letter,
    the MATH cluster's member answers, or the QuantumKatas code sha256."""
    from vgx.common.api import RequestCache
    from vgx.quantum_katas.load import sha256_text
    from vgx.raw_generator_export import collected
    kind = {'qwen3': 'peg_disc', 'gptoss': 'peg_disc', 'sonnet55': 'judge_p'}
    out = {}
    for d in PEG_DISCRIMINATORS:
        want = f'{dataset}_{kind[d]}'
        cands = None
        if dataset != 'gpqa':
            cands = json.loads((Path(PEG_CANDIDATES[d])/f'{dataset}_candidates.json').read_text())
        pol, caches, ledgers = defaultdict(dict), {}, {}
        for r in collected(RAW[d]):
            if r['kind'] != want or r.get('status') != 'ok':
                continue
            if dataset != 'gpqa' and r['_dir'].name not in ('stage2', 'stage2b'):
                continue
            if r['_dir'] not in caches:
                caches[r['_dir']] = RequestCache(r['_dir']/'cache')
                bp = r['_dir']/'cache'/'budget.json'
                ledgers[r['_dir']] = json.loads(bp.read_text())['requests'] if bp.exists() else {}
            usd = (ledgers[r['_dir']].get(r['request_key']) or {}).get('estimated_usd') or 0.0
            if kind[d] == 'judge_p':
                p = r.get('p_correct'); pc = (float(p), 1-float(p)) if isinstance(p, (int, float)) and 0 <= p <= 1 else None
            else:
                recs = [x for x in caches[r['_dir']].log(r['request_key']).records() if x.get('response') is not None]
                ab = _ab_probs(recs[0]) if recs else None
                pc = (ab[0]/(ab[0]+ab[1]), ab[1]/(ab[0]+ab[1])) if ab else None
            pc = pc or (0.5, 0.5)
            i = int(r['sample'])
            if dataset == 'gpqa':
                key = 'ABCD'[i]
            elif kind[d] == 'peg_disc':
                # Candidate files can be rewritten by a later stage; the PEG prompt itself names the candidate.
                recs = [x for x in caches[r['_dir']].log(r['request_key']).records() if x.get('response') is not None]
                prompt = recs[0]['prompt']
                text = prompt.split('Proposed Answer: ', 1)[1].rsplit('\n\nIs this answer correct?', 1)[0]
                if dataset == 'math500':
                    c = next((c for c in cands.get(r['item_id'], []) if text == c['answer'] or text in (c.get('members') or [])), None)
                    key = frozenset([text, *((c or {}).get('members') or [])])
                else:
                    key = sha256_text(text)
            elif dataset == 'math500':
                c = cands[r['item_id']][i]; key = frozenset([c['answer'], *(c.get('members') or [])])
            else:
                key = sha256_text(cands[r['item_id']][i]['code'])
            pol[r['item_id']][key] = (key, *pc, usd)
        out[d] = {item: list(opts.values()) for item, opts in pol.items()}
    return out


def _softmax(x):
    e = np.exp(np.asarray(x, float) - np.max(x)); return e/e.sum()


def _match(key, answer, dataset):
    return answer in key if dataset == 'math500' else key == answer


def _dmi(responses):
    """DMI payments (PEG_mechanism.ipynb compute_dmi_payments), normalized across discriminators."""
    n, m = len(responses), len(responses[0])
    halves = (responses[:(n+1)//2], responses[(n+1)//2:])
    pay = np.zeros(m)
    for i in range(m):
        for j in range(i+1, m):
            dets = []
            for part in halves:
                mat = np.zeros((2, 2))
                for row in part:
                    mat[0 if row[i] else 1][0 if row[j] else 1] += 1
                dets.append(np.linalg.det(mat))
            pay[i] += dets[0]*dets[1]; pay[j] += dets[0]*dets[1]
    return (pay - pay.mean())/(pay.std() + 1e-8)


def peg_game(items, batch_size=8, T=10, lr=0.1):
    """batch_update_discriminators_flexible with the frozen answer as every discriminator's choice.

    items: list of per-item lists of discriminator policies {option: [p_correct, p_incorrect]} (softmaxed
    across options as in apply_softmax) plus the chosen option per discriminator. The notebook leaves the
    last partial batch untouched; here it joins the previous batch so every item is updated."""
    starts = list(range(0, max(len(items) - batch_size, 0) + 1, batch_size)) or [0]
    bounds = [(s, starts[k+1] if k+1 < len(starts) else len(items)) for k, s in enumerate(starts)]
    for lo, hi in bounds:
        batch = items[lo:hi]
        for _ in range(T):
            resp = [[max(pol, key=lambda o: pol[o][0]) == ch for pol, ch in it] for it in batch]
            reward = _dmi(resp) if len(batch) >= 2 else np.zeros(len(batch[0]))
            for it, rs in zip(batch, resp):
                for d, ((pol, ch), correct) in enumerate(zip(it, rs)):
                    pc, pi = pol[ch]
                    g = (1/pc, -1/pc) if correct else (-1/pi, 1/pi)
                    lc, li = math.log(pc) + lr*reward[d]*g[0], math.log(pi) + lr*reward[d]*g[1]
                    mx = max(lc, li); ec, ei = math.exp(lc-mx), math.exp(li-mx)
                    pc = min(max(ec/(ec+ei), 1e-6), 1-1e-6)
                    pol[ch] = [pc, 1-pc]
    return items


def peg_scores(dataset, rows, policies):
    """Score = discriminators whose final argmax-correct option is the frozen answer, plus a small
    tie-break by their mean final P(correct | frozen answer)."""
    items, where, usage = [], [], [0.0]*len(rows)
    for i, r in enumerate(rows):
        if r['answer'] is None:
            continue
        it, ok = [], True
        for d in PEG_DISCRIMINATORS:
            opts = policies[d].get(r['item_id'], [])
            usage[i] += sum(o[3] for o in opts)
            sc, si = _softmax([o[1] for o in opts]) if opts else [], _softmax([o[2] for o in opts]) if opts else []
            pol = {k: [float(a), float(b)] for k, (_, a, b, _) in enumerate(opts)}
            # apply_softmax normalizes each class across options; the game uses the pair for the chosen option.
            pol = {k: [sc[k]/(sc[k]+si[k]), si[k]/(sc[k]+si[k])] for k in pol}
            ch = next((k for k, o in enumerate(opts) if _match(o[0], r['answer'], dataset)), None)
            if ch is None:
                ok = False; break
            it.append((pol, ch))
        if ok:
            items.append(it); where.append(i)
    peg_game(items)
    scores = [None]*len(rows)
    for i, it in zip(where, items):
        votes = sum(max(pol, key=lambda o: pol[o][0]) == ch for pol, ch in it)
        scores[i] = votes + 0.1*float(np.mean([pol[ch][0] for pol, ch in it]))
    return scores, usage, len(where)


def peg(rows, train_idx, test_idx, scores, usage):
    tau = fit_threshold([scores[i] for i in train_idx if rows[i]['valid']], [rows[i]['outcome'] for i in train_idx if rows[i]['valid']])
    expected = float(np.mean([usage[i] for i in train_idx]))
    return release_by_score(rows, test_idx, scores, tau, expected, usage), {'threshold': None if math.isinf(tau) else tau}

# ----------------------------------------------------------------------------- driver

def summarize(rows, ds, settings):
    s, u = crossfit_extended._summary(rows, ds, settings, LOSS)
    keep = ('released', 'correct_released', 'wrong_released', 'coverage', 'output_error_rate', 'output_error_wilson95',
            'calls_per_task', 'expected_usd_per_task', 'measured_usd_per_task', 'mean_utility', 'mean_utility_ci95')
    out = {k: s[k] for k in keep}
    if any('oracle_queries' in d for d in ds):
        out['oracle_queries_per_task'] = float(np.mean([d.get('oracle_queries', 0) for d in ds]))
    return out, u


def risk_coverage(rows, test, scores):
    """(coverage, error rate) when releasing valid test answers in decreasing score order."""
    pts = sorted(((scores[i], rows[i]['outcome']) for i in test if rows[i]['valid'] and scores[i] is not None), key=lambda p: -p[0])
    out, wrong = [], 0
    for k, (s, y) in enumerate(pts, 1):
        wrong += 1-y
        if k == len(pts) or pts[k][0] != s:
            out.append([k/len(test), wrong/k])
    return out


def run(dataset):
    data, cfg = load(dataset)
    settings = cfg['settings']
    samples = _raw_samples(dataset) if dataset in ('gpqa', 'math500') else {}
    policies = peg_policies(dataset)
    result = {'dataset': dataset, 'loss': LOSS, 'generators': {}}
    for gen, rows in data.items():
        order = []; arms = defaultdict(dict); folds = []
        sc_scores, sc_usage = self_consistency_scores(dataset, gen, rows, samples) if samples else (None, None)
        vrows, v_usage = vote_rows(dataset, gen, rows, samples) if samples else (None, None)
        curves = defaultdict(dict)
        pg_scores, pg_usage, pg_n = peg_scores(dataset, rows, policies)
        for train_idx, test_idx in splits(dataset, rows):
            order += test_idx
            base, fit = ours_and_fixed(rows, train_idx, test_idx, settings)
            fitted, triple = fit['fitted'], fit['triple']
            info = {'triple': triple}
            for name, ds in base.items():
                arms[name].update(zip(test_idx, ds))
            if sc_scores is not None:
                ds, info['self_consistency'] = self_consistency(rows, train_idx, test_idx, sc_scores, sc_usage)
                arms['self_consistency'].update(zip(test_idx, ds))
                curves['self_consistency'].update({i: sc_scores[i] for i in test_idx})
            if vrows is not None:
                arms['sc_vote'].update(zip(test_idx, self_consistency_vote(vrows, test_idx, v_usage, train_idx, settings)))
            for name, pool, binned, wp, ft in (('weaver_3', triple, True, True, False), ('weaver_5', VERIFIERS, True, True, False),
                                               ('weaver_5_noprior', VERIFIERS, True, False, False),
                                               ('weaver_5_fit', VERIFIERS, True, False, True),
                                               ('weaver_5_binary', VERIFIERS, False, False, False)):
                ds, w = weaver(rows, train_idx, test_idx, pool, fitted, binned, wp, ft)
                ws = w.pop('scores')
                curves[name].update({i: ws[i] for i in test_idx})
                arms[name].update(zip(test_idx, ds)); info[name] = w
            # Weak verifier: the best single verifier for our planner on the training folds (training J0).
            priors = np.array([rows[i]['prior'] for i in train_idx if rows[i]['valid']])
            weak = max(VERIFIERS, key=lambda v: (float(crossfit.planner_for(fitted, [v], LOSS, settings)[0].j(0, priors).mean()), v))
            strong = max((v for v in VERIFIERS if v != weak), key=lambda v: (auroc(rows, train_idx, v), v))
            for name, s in (('ssv_oracle', 'oracle'), ('ssv_model', strong)):
                ds, info[name] = ssv(rows, train_idx, test_idx, weak, s, fitted)
                arms[name].update(zip(test_idx, ds))
            ds, info['peg'] = peg(rows, train_idx, test_idx, pg_scores, pg_usage)
            arms['peg'].update(zip(test_idx, ds))
            curves['peg'].update({i: pg_scores[i] for i in test_idx})
            folds.append(info)
        test_rows = [rows[i] for i in order]
        res = {'n_test': len(order), 'peg_items_with_all_discriminators': pg_n, 'valid_test': sum(r['valid'] for r in test_rows),
               'correct_test': sum(r['outcome'] for r in test_rows if r['valid']), 'folds': folds,
               'arms': {}, 'ours_minus': {},
               'risk_coverage': {name: risk_coverage(rows, order, sc) for name, sc in
                                 ((n, [c.get(i) for i in range(len(rows))]) for n, c in curves.items())}}
        util = {}
        for name, d in arms.items():
            base_rows = [vrows[i] for i in order] if name == 'sc_vote' else test_rows
            res['arms'][name], util[name] = summarize(base_rows, [d[i] for i in order], settings)
        from vgx.gpqa.diamond_score import boot
        for name in util:
            if name != 'ours':
                b = boot(util['ours']-util[name], settings['bootstrap_repeats'], 20261027)
                res['ours_minus'][name] = {'mean': b['mean'], 'ci95': b['ci95']}
        res['risk_coverage']['generator_only'] = risk_coverage(rows, order, [r['prior'] if r['valid'] else None for r in rows])
        if vrows is not None:
            res['risk_coverage']['sc_vote'] = risk_coverage(vrows, order, [r['prior'] if r['valid'] else None for r in vrows])
            res['sc_vote_answer_changed'] = sum(vrows[i]['valid'] and rows[i]['valid'] and vrows[i]['answer'] != rows[i]['answer']
                                                 for i in order)
        result['generators'][gen] = res
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT/f'{dataset}.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('datasets', nargs='+', choices=['gpqa', 'math500', 'qkatas'])
    a = p.parse_args()
    for ds in a.datasets:
        r = run(ds)
        for g, v in r['generators'].items():
            print(ds, g, {k: (a['released'], a['wrong_released'], round(a['mean_utility'], 3)) for k, a in v['arms'].items()})


if __name__ == '__main__':
    main()
