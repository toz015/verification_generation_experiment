"""Continue the authorized completion pilot, then gated full generator collection.

Status is durable and suitable for a read-only completion monitor. No evaluation
scoring or additional verifier collection is performed by this pipeline.
"""
import argparse
import json
from pathlib import Path
import time

from vgx.common.storage import atomic_json, file_digest, file_lock
from vgx.gpqa import completion_study as controls, staged_generator as staged
from vgx.gpqa import generator_expansion as expansion
from vgx.gpqa.expansion_analysis import analyze


def run(root, *, allow_api=False):
    root=Path(root);status={'pipeline_sha256':file_digest(Path(__file__)), 'started_at':time.time()}
    def update(stage, **extra):
        status.update(stage=stage,updated_at=time.time(),**extra)
        atomic_json(root/'status.json',status);print(json.dumps(status),flush=True)
    control_root=Path('results/gpqa_qwen_completion_20261005')
    def config(name):return json.loads((Path('configs')/name).read_text())
    with file_lock(root/'pipeline.lock'):
        try:
            update('waiting_for_controls',control_root=str(control_root))
            # Wait for the currently running collector. Do not send another
            # request or terminate an in-flight provider execution.
            with file_lock(control_root/'collection.lock'):
                collection=json.loads((control_root/'collection_50.json').read_text())
                if collection['error'] or collection['completed']!=collection['target']:
                    raise RuntimeError('control collector incomplete; inspect its ledger before continuing')
            selection=controls.gate(control_root)
            update('controls_complete',control_selection=selection)
            if selection['selected_arm'] is None:
                pilot_root=Path('results/gpqa_staged_qwen_pilot_20261005')
                staged.prepare(config('gpqa_staged_qwen_pilot.json'),pilot_root)
                update('collecting_staged_pilot',pilot_root=str(pilot_root))
                staged.collect(pilot_root,allow_api=allow_api)
                result=staged.gate(pilot_root)
                if not result['passed']:raise RuntimeError('staged pilot completion gate failed; expansion not started')
                full_root=Path('results/gpqa_staged_generator_expansion_20261005')
                staged.prepare(config('gpqa_staged_generator_expansion.json'),full_root)
                update('collecting_full_generators',full_root=str(full_root),protocol='two_stage_qwen',pilot_gate=result)
                staged.collect(full_root,allow_api=allow_api)
                report=analyze(full_root,staged=True)
            else:
                full_root=Path('results/gpqa_generator_expansion_20261005')
                expansion.prepare(config('gpqa_generator_expansion.json'),full_root)
                update('collecting_full_generators',full_root=str(full_root),protocol='single_stage_qwen')
                expansion.collect(full_root,allow_api=allow_api)
                report=analyze(full_root)
            update('complete',calibration_summary=str(full_root/'calibration_summary.json'),
                   evaluation_labels_read=False,calibration_summaries=report['summaries'])
            return status
        except Exception as exc:
            update('blocked',error_type=type(exc).__name__,error=str(exc))
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--allow-api',action='store_true');a=p.parse_args();run(a.root,allow_api=a.allow_api)
