"""Explicit CPU-only entry. No benchmark defaults are modified."""
import argparse
import json
import os
import platform
import sys
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

import torch

from . import test_correctness


class Result(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []

    def startTest(self, test):
        self.started = time.monotonic()
        super().startTest(test)

    def _record(self, test, outcome, error=None):
        self.records.append({'test': test.id(), 'outcome': outcome,
                             'seconds': time.monotonic() - self.started, 'error': error})

    def addSuccess(self, test):
        self._record(test, 'PASS')
        super().addSuccess(test)

    def addFailure(self, test, err):
        self._record(test, 'FAIL', self._exc_info_to_string(err, test))
        super().addFailure(test, err)

    def addError(self, test, err):
        self._record(test, 'ERROR', self._exc_info_to_string(err, test))
        super().addError(test, err)

    def addSubTest(self, test, subtest, err):
        if err:
            self._record(subtest, 'FAIL', self._exc_info_to_string(err, test))
        super().addSubTest(test, subtest, err)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'result.json').exists():
        raise SystemExit('Choose a new run directory; evidence is not overwritten.')
    os.environ['WEAVETP_CORRECTNESS_OUTPUT'] = str(out)
    torch.set_num_threads(1)
    torch.manual_seed(731)
    start = datetime.now(timezone.utc).isoformat()
    suite = unittest.defaultTestLoader.loadTestsFromModule(test_correctness)
    result = unittest.TextTestRunner(verbosity=2, resultclass=Result).run(suite)
    record = {'scope': 'CPU tiny tensor fixtures, real planner/transaction, mailbox transport double',
              'start_utc': start, 'end_utc': datetime.now(timezone.utc).isoformat(),
              'python': sys.version, 'torch': torch.__version__, 'platform': platform.platform(),
              'pid': os.getpid(), 'seed': 731, 'cuda_initialized': torch.cuda.is_initialized(),
              'GPU': 'NOT_RUN', 'real_checkpoint': 'NOT_RUN', 'tests_run': result.testsRun,
              'successful': result.wasSuccessful(), 'tests': result.records,
              'evidence': test_correctness.EVIDENCE}
    (out / 'result.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
