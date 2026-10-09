from dataclasses import asdict
import json
from pathlib import Path
import sys

import pytest

from vgx.gpqa import workflow
from vgx.gpqa.artifacts import load_candidates, validate_original
from vgx.gpqa.verifiers import load_specs
from vgx.gpqa.workflow import prepare_policy


def test_offline_cli_roundtrip_never_calls_models(frozen_run, monkeypatch):
    run = frozen_run
    monkeypatch.setattr('vgx.common.vertex.VertexBatchRunner.run',lambda *a,**k:pytest.fail('no inference'))
    spec_path = run.root/'specs.json'
    spec_path.write_text(json.dumps({'schema':1,'verifiers':[asdict(s) for s in run.specs],
                                    'arms':[{'name':'original','order':[s.id for s in run.specs]}]}))
    def command(name, *args):
        monkeypatch.setattr(sys,'argv',['workflow',name,'--bundle',str(run.bundle),'--cache',str(run.cache.root),*map(str,args)])
        workflow.main()
    command('prepare-comparison','--specs',spec_path,'--output',run.root/'comparison.json')
    command('collect','--specs',spec_path,'--output',run.root/'collected.json')
    usage = json.loads((run.root/'collected.json.usage.json').read_text())
    assert usage['successful_calls'] == 0  # cached originals are not incremental charges
    policy_path = run.root/'policy.json'
    command('prepare-policy','--specs',spec_path,'--normalized-costs','.01','.01','--loss','1','--output',policy_path)
    output = run.root/'execution'
    command('execute','--policy',policy_path,'--output',output)  # defaults to evaluation
    decision = json.loads((output/'decisions.json').read_text())
    assert decision['binding']['partition'] == 'evaluation'
    command('score','--policy',policy_path,'--decisions',output/'decisions.json','--output',run.root/'scored.json')
    assert json.loads((run.root/'scored.json').read_text())['n'] == len(run.split.evaluation)


def test_cli_live_billing_does_not_open_unselected_verifier_outputs(frozen_run, monkeypatch):
    from test_sequential import make_policy
    run = frozen_run
    policy = make_policy(run)
    path = run.root/'policy.json'
    path.write_text(json.dumps(policy))
    _, candidates = load_candidates(run.bundle)
    forbidden = set()
    for candidate in candidates:
        key = run.specs[1].runner().cache_key(run.specs[1].request(candidate))
        forbidden.add(run.cache.log(key).path)
    # CallLog.records uses Path.open, whereas candidate/checkpoint files use read_text.
    original_open = Path.open
    def guarded(path, *args, **kwargs):
        assert path not in forbidden, 'unselected future verifier output was opened'
        assert path.name not in ('calibration_records.jsonl','evaluation_labels.jsonl')
        return original_open(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guarded)
    monkeypatch.setattr(sys,'argv',['workflow','execute','--bundle',str(run.bundle),'--cache',str(run.cache.root),
                                  '--policy',str(path),'--output',str(run.root/'execution')])
    workflow.main()


def test_modified_original_metrics_are_rejected(original_run):
    run = original_run
    path = run.run_dir/'pilot_metrics.json'
    metrics = json.loads(path.read_text())
    metrics['evaluation_metrics']['generator_confidence']['brier'] += .1
    path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError,match='reported generator forecasts'):
        validate_original(run.source,run.sample,run.run_dir,run.run_id)


def test_checked_in_specs_validate_without_credentials():
    path = Path(__file__).resolve().parents[1]/'configs/gpqa_frozen_verifiers.json'
    specs = load_specs(json.loads(path.read_text()))
    assert [s.id for s in specs][-2:] == ['jev_choice_v1','jev_noul_v1']


def test_old_main_no_longer_exposes_generation(monkeypatch):
    from vgx.gpqa import run_pilot
    monkeypatch.setattr(sys,'argv',['run_pilot'])
    with pytest.raises(SystemExit) as result:
        run_pilot.main()
    assert result.value.code == 2
