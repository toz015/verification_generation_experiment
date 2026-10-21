"""Runs inside the grader venv: official Omni-MATH-Rule answer comparison, one pair per line.

Mirrors evaluation/evaluate.py + parser.run_execute at the pinned commit: the reference is
parsed with parse_ground_truth(..., 'omni-math') (answer wrapped in \\boxed{}), the
prediction is wrapped in \\boxed{} and passed through extract_answer and strip_string, and
math_equal(pred, gt, timeout=True) decides. A comparison taking over 3 s counts as
incorrect, as in the official pool timeout. Standard library only, besides the grader.
"""
import json
import signal
import sys


def main(grader_dir):
    sys.path.insert(0, grader_dir); sys.path.insert(0, grader_dir + '/latex2sympy')
    from grader import math_equal
    from parser import extract_answer, parse_ground_truth, strip_string

    class Timeout(Exception):
        pass

    def alarm(*_):
        raise Timeout()
    signal.signal(signal.SIGALRM, alarm)
    for line in sys.stdin:
        r = json.loads(line)
        out = {'item_id': r['item_id']}
        try:
            signal.alarm(3)
            _, gt = parse_ground_truth({'gt_cot': r['reference'], 'gt': r['reference']}, 'omni-math')
            pred = strip_string(extract_answer('\\boxed{' + r['answer'] + '}', 'omni-math'))
            out.update(pred=pred, gt=gt, correct=bool(math_equal(pred, gt, timeout=True)), status='graded')
        except Timeout:
            out.update(correct=False, status='timeout')
        except Exception as exc:  # official evaluate.py aborts here; we record and count as incorrect
            out.update(correct=False, status='grader_error', error=f'{type(exc).__name__}: {exc}'[:300])
        finally:
            signal.alarm(0)
        print(json.dumps(out), flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
