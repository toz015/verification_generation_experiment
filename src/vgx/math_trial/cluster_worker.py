"""Runs inside the grader venv: group equivalent candidate answers per problem with the official math_equal.

Input lines: {"item_id", "answers": [...]}. Each answer is normalized exactly as grade_worker does
(\\boxed{} + extract_answer + strip_string), then greedily assigned to the first earlier cluster whose
representative is math_equal to it (3 s limit per comparison; a timeout means not equal).
Output lines: {"item_id", "clusters": [[answer indices], ...]}.
"""
import json
import signal
import sys


def main(grader_dir):
    sys.path.insert(0, grader_dir); sys.path.insert(0, grader_dir + '/latex2sympy')
    from grader import math_equal
    from parser import extract_answer, strip_string

    class Timeout(Exception):
        pass

    def alarm(*_):
        raise Timeout()
    signal.signal(signal.SIGALRM, alarm)

    def eq(a, b):
        try:
            signal.alarm(3)
            return bool(math_equal(a, b))
        except Exception:
            return False
        finally:
            signal.alarm(0)
    for line in sys.stdin:
        r = json.loads(line)
        norm = []
        for a in r['answers']:
            try:
                norm.append(strip_string(extract_answer('\\boxed{' + a + '}', 'omni-math')))
            except Exception:
                norm.append(a)
        clusters = []
        for i, a in enumerate(norm):
            for c in clusters:
                if norm[c[0]] == a or eq(a, norm[c[0]]):
                    c.append(i); break
            else:
                clusters.append([i])
        print(json.dumps({'item_id': r['item_id'], 'clusters': clusters}), flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
