"""T07: run the same CPU regressions on the real Linux interpreter."""

import argparse
import sys

from common import ROOT, command, read, require, run

TESTS = ("test_summarize_weavetp_formal.py", "test_compare_weavetp_16gpu.py",
         "test_profile_weavetp_16gpu.py", "test_server_acceptance.py",
         "test_weavetp_review_findings.py", "test_weavetp_observations.py")


def check(args, out, env):
    failed = []
    for name in TESTS:
        try:
            command([sys.executable, "-B", "-X", "utf8", str(ROOT / "tests/unit_tests/resharding" / name)],
                    out, name.removesuffix(".py"), cwd=out, env={**env, "CUDA_VISIBLE_DEVICES": ""})
        except ValueError:
            failed.append(name)
    require(not failed, f"CPU gates failed: {failed}; inspect individual logs, do not run GPUs")
    return {"suites": list(TESTS), "gpu_started": False,
            "exit_codes": {name: read(out / (name.removesuffix('.py') + '.json'))["exit_code"] for name in TESTS}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    raise SystemExit(run(parser.parse_args(), check))
