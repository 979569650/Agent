import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare two eval reports.")
    p.add_argument("--baseline", required=True, help="baseline report json path")
    p.add_argument("--current", required=True, help="current report json path")
    p.add_argument("--out", default="eval/reports/compare.md", help="markdown output path")
    return p.parse_args()


def _load(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    base_path = Path(args.baseline)
    current_path = Path(args.current)
    out_path = Path(args.out)

    if not current_path.exists():
        print(f"[EvalCompare] current report missing: {current_path}")
        return 1

    if not base_path.exists():
        base_path.parent.mkdir(parents=True, exist_ok=True)
        base_path.write_text(current_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"[EvalCompare] baseline absent; bootstrapped from current: {base_path}")

    baseline = _load(str(base_path))
    current = _load(str(current_path))

    base_rate = float(baseline.get("pass_rate", 0.0))
    curr_rate = float(current.get("pass_rate", 0.0))
    delta = round(curr_rate - base_rate, 2)

    lines = [
        "# Eval 回归对比报告",
        "",
        f"- baseline: `{base_path}`",
        f"- current: `{current_path}`",
        "",
        "| 指标 | Baseline | Current | Delta |",
        "|---|---:|---:|---:|",
        f"| pass_rate | {base_rate:.2f}% | {curr_rate:.2f}% | {delta:+.2f}% |",
        f"| passed/total | {baseline.get('passed', 0)}/{baseline.get('total', 0)} | {current.get('passed', 0)}/{current.get('total', 0)} | - |",
        "",
        f"- baseline failed_ids: {baseline.get('failed_ids', [])}",
        f"- current failed_ids: {current.get('failed_ids', [])}",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[EvalCompare] report saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
