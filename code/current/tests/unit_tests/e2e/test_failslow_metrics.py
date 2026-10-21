"""Fail-slow figure metrics (tools/resharding/e2e/failslow_metrics.py) and the analyzer's fail-slow branch."""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

from tools.resharding.e2e import failslow_metrics as fs

SETTINGS = {'warmup_s': 10.0, 'slow_s': 20.0, 'tail_s': 10.0, 'detect_s': 2.0}


def row(seq, replica, admit, tokens, done=True):
    return {'seq': seq, 'replica': replica, 'input_len': 4, 'output_len': len(tokens), 't_arrive': 0.0, 't_admit': admit,
            't_first': tokens[0] if tokens else None, 't_done': tokens[-1] if done and tokens else None,
            'admits': [{'t': admit, 'step': 0, 'replica': replica, 'layout': 4, 'generated_at_admit': 0}], 't_tokens': tokens}


def result(method='weavetp', level='W', rate=1.0, launch='l', restart=False, gpu_sharing='exclusive'):
    """inject_start = 10 s, window [10, 40); EDP0 (replica 0) and EDP1 (replica 2) each commit `rate` tokens/s per request."""
    tokens = [0.25 + i / rate for i in range(int(40 * rate))]
    requests = [row(1, 0, 0.0, tokens, done=False), row(2, 2, 0.0, list(tokens), done=False)]
    worker = {'fail': 12.0, 'shrink_done': 15.0, 'rejoin_start': 30.5, 'expand_done': 33.0}
    return {'launch': launch, 'method': method, 'level': level, 'settings': SETTINGS, 'requests': requests, 'gpu_sharing': gpu_sharing,
            'events_s': {'inject_start': 10.0, 'detect': 12.0, 'deadline': 40.0}, 'worker_events_s': worker,
            'restart_loaded_s': {'shrink': 14.0, 'expand': 32.0} if restart else None, 'requests_unfinished': 2,
            'deadline_overrun_s': 0.0, 'check': {'passed': True},
            'edp0_decode_steps': [[t, 0.02] for t in range(0, 10)] + [[10.0 + t / 10, 0.05] for t in range(20)]}


class WindowTests(unittest.TestCase):
    def test_window_throughput_counts_only_the_window(self):
        metrics, series = fs.analyze(result())
        self.assertEqual(metrics['window_s'], [10.0, 40.0])
        self.assertAlmostEqual(metrics['throughput_tok_s'], 2.0)  # one token per second from each of two requests
        self.assertEqual(sum(series['global']), 80)
        self.assertEqual(series['EDP0'][20], 1)
        self.assertEqual(fs.window_throughput([(5.0, 0, 1), (10.0, 0, 1), (39.9, 2, 2), (40.0, 2, 2)], 10.0, 40.0), 2 / 30)

    def test_ttft_from_first_admission_and_tpot_intersecting_the_window(self):
        rows = [row(1, 0, 11.0, [11.5, 12.5, 13.5]), row(2, 0, 0.0, [0.5, 1.5]), row(3, 2, 39.0, [39.25])]
        latency = fs.window_latency(rows, 10.0, 40.0)
        self.assertEqual(latency['ttft_s'], {'p50': 0.25, 'p99': 0.5, 'n': 2})  # row 2's first token is before the window
        self.assertEqual(latency['tpot_s'], {'p50': 1.0, 'p99': 1.0, 'n': 1})  # one-token and out-of-window rows have none

    def test_slow_rate_compares_detection_window_with_the_baseline(self):
        self.assertAlmostEqual(fs.analyze(result())[0]['slow_rate_measured'], 2.5)
        self.assertIsNone(fs.slow_rate([[11.0, 0.05]], 10.0, 12.0))

    def test_restart_downtime_per_switch(self):
        downtime = fs.analyze(result('restart', restart=True))[0]['restart_downtime']
        self.assertEqual(downtime['shrink'], {'stop_s': 12.0, 'loaded_s': 14.0, 'resume_s': 15.25, 'downtime_s': 3.25})
        self.assertEqual(downtime['expand']['resume_s'], 33.25)
        self.assertAlmostEqual(downtime['total_s'], 3.25 + 2.75)
        self.assertIsNone(fs.analyze(result('weavetp'))[0]['restart_downtime'])


class Fig1Tests(unittest.TestCase):
    def launches(self):
        runs = [result('nofault', None, 1.0, 'n1'), result('nofault', None, 1.0, 'n2'), result('weavetp', 'W', 0.5, 'w1'),
                result('weavetp', 'W', 1.0, 'w2'), result('restart', 'S', 0.5, 'r1', restart=True)]
        return [fs.analyze(r)[0] for r in runs]

    def test_normalised_to_the_nofault_mean_of_the_same_timeline(self):
        rows = {(r['severity'], r['method']): r for r in fs.fig1(self.launches())}
        self.assertEqual(list(rows), [('W', 'weavetp'), ('S', 'restart'), ('-', 'nofault')])
        self.assertEqual({r['gpu_sharing'] for r in fs.fig1(self.launches())}, {'exclusive'})
        self.assertAlmostEqual(rows['-', 'nofault']['mean'], 1.0)
        self.assertEqual(rows['W', 'weavetp']['n'], 2)
        self.assertAlmostEqual(rows['W', 'weavetp']['mean'], (0.5 + 1.0) / 2)
        self.assertAlmostEqual(rows['W', 'weavetp']['std'], 0.3535533905932738)
        self.assertIsNone(rows['S', 'restart']['std'])
        self.assertIsNotNone(rows['S', 'restart']['restart_downtime_s'])
        self.assertIsNone(rows['W', 'weavetp']['restart_downtime_s'])

    def test_shared_gpu_launches_have_their_own_rows_and_nofault_reference(self):
        shared = [fs.analyze(result('nofault', None, 2.0, 'n3', gpu_sharing='shared'))[0],
                  fs.analyze(result('weavetp', 'W', 1.0, 'w3', gpu_sharing='shared'))[0]]
        rows = {(r['severity'], r['method'], r['gpu_sharing']): r for r in fs.fig1(self.launches() + shared)}
        self.assertAlmostEqual(rows['W', 'weavetp', 'exclusive']['mean'], (0.5 + 1.0) / 2)  # unchanged by the shared launches
        self.assertAlmostEqual(rows['W', 'weavetp', 'shared']['mean'], 0.5)  # against the shared nofault launch only
        self.assertEqual((rows['W', 'weavetp', 'shared']['n'], rows['-', 'nofault', 'shared']['n']), (1, 1))

    def test_no_nofault_launch_leaves_the_mean_empty(self):
        rows = fs.fig1([fs.analyze(result('weavetp'))[0]])
        self.assertEqual((rows[0]['n'], rows[0]['mean']), (0, None))

    def test_analyzer_cli_writes_fig1_and_timelines_and_rejects_failed_checks(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'tools' / 'resharding' / 'failstop'))
        import analyze_failstop_e2e

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, data in (('a', result('nofault', None, 1.0, 'n1')), ('b', result('weavetp', 'M', 0.5, 'w1')),
                               ('c', {**result('ignore', 'M', 0.5, 'i1'), 'check': {'passed': False}})):
                (root / 'runs' / name).mkdir(parents=True)
                (root / 'runs' / name / 'failslow_result.json').write_text(json.dumps(data))
            code = analyze_failstop_e2e.main([str(root / 'runs'), '--out', str(root / 'out')])
            self.assertEqual(code, 1)  # the failed launch is rejected
            with (root / 'out' / 'fig1.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(list(rows[0]), list(fs.FIG1))
            self.assertEqual([(r['severity'], r['method'], r['mean']) for r in rows], [('M', 'weavetp', '0.5'), ('-', 'nofault', '1')])
            self.assertTrue((root / 'out' / 'w1' / 'timeline.csv').is_file())
            self.assertFalse((root / 'out' / 'failslow_launch_metrics.csv').exists())
            rejected = json.loads((root / 'out' / 'rejected.json').read_text())
            self.assertEqual(len(rejected), 1)


if __name__ == '__main__':
    unittest.main()
