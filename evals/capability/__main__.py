"""CLI: python -m evals.capability --brain openai_compat --model <tag> --out reports/<model>/"""
from __future__ import annotations

import argparse

from evals.capability.runner import make_brain, run_suite, write_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure Medha's capability suite against a configured brain.")
    parser.add_argument("--brain", choices=("openai_compat", "medha", "stub"), default="openai_compat")
    parser.add_argument("--model", required=True, help="Local model tag, or a label for the experimental model")
    parser.add_argument("--out", required=True, help="Output directory, for example reports/qwen3-8b/")
    args = parser.parse_args()
    brain = make_brain(args.brain, args.model)
    brain.warmup()
    report = run_suite(brain)
    write_report(args.model, args.out, report)
    for category, scores in report["categories"].items():
        print(f"{category}: {scores['passed']}/{scores['total']} ({scores['rate']:.1%})")
    print(f"Wrote measured report to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
