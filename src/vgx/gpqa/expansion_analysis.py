"""Offline calibration-only generator summaries; evaluation labels stay closed."""
import argparse
from itertools import combinations
import json
from pathlib import Path

from vgx.common.storage import atomic_json, file_digest, read_jsonl
from vgx.gpqa import generator_expansion as study
from vgx.gpqa.artifacts import load_candidates
from vgx.gpqa.reasoning_analysis import paired_summary
from vgx.gpqa.sequential import unseal


def analyze(root, *, staged=False):
    root = Path(root)
    if staged:
        from vgx.gpqa import staged_generator
        plan, _ = staged_generator.load(root)
        old, pilot_items, _ = study.previous.load(Path(plan['config']['previous_results']))
    else:
        plan, _ = study.load(root)
        pilot, _, _, _, _, _ = study.source(plan['config'])
        old, _, _ = study.previous.load(Path(pilot['config']['previous_results']))
        _, pilot_items = study.completion.load(Path(plan['config']['completion_root']))
    source = Path(old['config']['baseline_bundle'])
    manifest, baseline = load_candidates(source)
    path = source/'calibration_records.jsonl'
    if file_digest(path) != manifest['files'][path.name]: raise ValueError('calibration labels changed')
    labels = {r['item_id']: r['correct_index'] for r in read_jsonl(path)}
    bundle = json.loads((root/'frozen_candidates.json').read_text()); unseal(bundle, 'bundle_id')
    if bundle['study_id'] != plan['study_id']: raise ValueError('candidate study differs')
    groups = dict(bundle['groups'])
    groups['gemini_baseline'] = [{'item_id':c.item_id, 'partition':c.partition, 'answer':c.answer, 'p_correct':c.p_correct} for c in baseline]
    pilot_ids = {i.item_id for i in pilot_items}
    records = {}
    for name, candidates in groups.items():
        rows = []
        for c in sorted(candidates, key=lambda c:c['item_id']):
            if c['partition'] != 'calibration': continue
            outcome = int(c['answer'] is not None and 'ABCD'.index(c['answer']) == labels[c['item_id']])
            rows.append({'item_id':c['item_id'], 'generator_answer':c['answer'],
                         'generator_p_correct':c['p_correct'], 'outcome':outcome})
        records[name] = rows
    summaries = {}
    for name, rows in records.items():
        for cohort in ('all_calibration','pilot','remaining_calibration'):
            selected = [r for r in rows if cohort == 'all_calibration' or ((r['item_id'] in pilot_ids) == (cohort == 'pilot'))]
            valid = [r for r in selected if r['generator_answer'] is not None and r['generator_p_correct'] is not None]
            wrong = [r for r in valid if not r['outcome']]
            summaries[name+':'+cohort] = {'n':len(selected), 'correct':sum(r['outcome'] for r in selected),
                'valid_answers':sum(r['generator_answer'] is not None for r in selected),
                'valid_confidence':len(valid),
                'mean_confidence':sum(r['generator_p_correct'] for r in valid)/len(valid) if valid else None,
                'mean_wrong_confidence':sum(r['generator_p_correct'] for r in wrong)/len(wrong) if wrong else None,
                'raw_brier':sum((r['generator_p_correct']-r['outcome'])**2 for r in valid)/len(valid) if valid else None}
    paired = {}
    for a,b in combinations(records,2):
        paired[b+'_minus_'+a] = paired_summary(records[a],records[b])
        paired[b+'_minus_'+a]['direction'] = b+' minus '+a
    result = {'study_id':plan['study_id'], 'analysis_sha256':file_digest(Path(__file__)),
              'calibration_label_sha256':file_digest(path), 'summaries':summaries, 'paired':paired,
              'evaluation_labels_read':False, 'calibration_fitted':False,
              'limitations':['Different model/prompt/settings conditions; not a pure model effect.',
                  'Prompt completion selection used the pilot; remaining calibration is reported separately.',
                  'Raw confidence is a self report. No verifier likelihood, policy, or high-confidence guarantee established.']}
    atomic_json(root/'calibration_summary.json',result)
    atomic_json(root/'calibration_analysis_records.json',records)
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--staged',action='store_true');args=p.parse_args()
    v=analyze(args.root,staged=args.staged); print(json.dumps({k:x for k,x in v.items() if k!='paired'}))
