"""Recovery metrics and summary.csv for formal fail-stop launches (tools/resharding/e2e/metrics.py).

  python analyze_failstop_e2e.py --out DIR <launch directory or run_failstop_e2e output root>... [--degraded-reference TOK_S]

A launch directory holds result.json and events_actual.json (the SL3060 gate's output; run_case's own outcome is
launch_result.json); a root is searched for them. Per launch: `metrics_<launch>.json` and `per_second_<launch>.csv`;
over all launches: `launch_metrics.csv` (one row per launch) and `summary.csv` (mean / sample std / n per method x
scenario x mode x gpu_sharing and metric; shared-GPU timing launches, --shared-gpu-ok, are their own groups). A
launch whose request rows do not reconcile is reported and left out.

Fail-slow: when the paths hold failslow_result.json files (run_failstop_e2e --scenario failslow, SL3060 output), the
output is `<launch>/timeline.csv` + `<launch>/metrics.json` per launch and `fig1.csv` (e2e/failslow_metrics.py); a launch whose event-order check failed is rejected.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tools.resharding.e2e import failslow_metrics as fs  # noqa: E402
from tools.resharding.e2e import metrics as m  # noqa: E402


def launch_directories(paths):
    found = []
    for path in paths:
        path = Path(path)
        if (path / 'result.json').is_file() and (path / 'events_actual.json').is_file():
            found.append(path)
        else:
            found.extend(sorted(p.parent for p in path.rglob('events_actual.json') if (p.parent / 'result.json').is_file()))
    return found


def analyze_failslow(paths, out):
    launches, rejected = [], []
    for path in sorted({p for path in map(Path, paths) for p in ([path] if path.is_file() else path.rglob('failslow_result.json'))}):
        try:
            metrics, series = fs.analyze(fs.load(path))
            if not metrics['check_passed']:
                raise ValueError('the launch failed its event-order check')
        except (ValueError, KeyError, TypeError) as error:
            rejected.append({'file': str(path), 'error': f'{type(error).__name__}: {error}'})
            continue
        directory = out / metrics['launch']
        directory.mkdir(parents=True, exist_ok=True)
        m.write_series(directory / 'timeline.csv', series)
        (directory / 'metrics.json').write_text(json.dumps(metrics, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
        launches.append(metrics)
    rows = fs.write_fig1(out, launches)
    (out / 'rejected.json').write_text(json.dumps(rejected, indent=1) + '\n', encoding='utf-8')
    print(json.dumps({'failslow_launches': len(launches), 'rejected': rejected, 'fig1_rows': len(rows), 'out': str(out)}, indent=1))
    return 0 if launches and not rejected else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('paths', nargs='+')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--degraded-reference', type=float, default=None,
                        help='tokens/s reference for the shrink TTP (default: the second half of the degraded window)')
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if any(Path(p).name == 'failslow_result.json' or any(Path(p).rglob('failslow_result.json')) for p in args.paths):
        return analyze_failslow(args.paths, args.out)
    collected, rejected = [], []
    for directory in launch_directories(args.paths):
        try:
            launch = m.load_launch(directory)
            metrics, series = m.analyze(launch, args.degraded_reference)
        except (ValueError, KeyError) as error:
            rejected.append({'directory': str(directory), 'error': f'{type(error).__name__}: {error}'})
            continue
        (args.out / f"metrics_{metrics['launch']}.json").write_text(json.dumps(metrics, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
        m.write_series(args.out / f"per_second_{metrics['launch']}.csv", series)
        collected.append(metrics)
    flat = [m.flatten(x) for x in collected]
    fields = sorted({k for row in flat for k in row},
                    key=lambda k: (k not in ('launch', 'method', 'scenario', 'mode', 'run_mode', 'gpu_sharing'), k))
    m.write_csv(args.out / 'launch_metrics.csv', flat, fields)
    rows = m.summarize(collected)
    m.write_csv(args.out / 'summary.csv', rows, ('method', 'scenario', 'mode', 'gpu_sharing', 'metric', 'mean', 'sample_std', 'n_valid',
                                                 'n_censored_or_missing', 'n_launches'))
    (args.out / 'rejected.json').write_text(json.dumps(rejected, indent=1) + '\n', encoding='utf-8')
    print(json.dumps({'launches': len(collected), 'rejected': rejected, 'summary_rows': len(rows), 'out': str(args.out)}, indent=1))
    return 0 if collected and not rejected else 1


if __name__ == '__main__':
    sys.exit(main())
