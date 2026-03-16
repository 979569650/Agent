import json
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.domain.policies import should_use_retrieval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval policy evaluation.")
    parser.add_argument("--cases", default="eval/eval_cases.json", help="Path to eval cases json")
    parser.add_argument("--report", default="", help="Optional output report json path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases_path = Path(args.cases)
    if not cases_path.exists():
        print(f"[Eval] 未找到评测集: {cases_path}")
        return 1

    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    total = len(cases)
    passed = 0

    print(f"[Eval] Loaded {total} cases")
    for c in cases:
        q = c["query"]
        expected = bool(c["expected_use_retrieval"])
        got = should_use_retrieval(q)
        ok = got == expected
        if ok:
            passed += 1
        print(f"- {c['id']}: expected={expected}, got={got}, pass={ok}")

    score = (passed / total * 100.0) if total else 0.0
    print(f"[Eval] pass={passed}/{total} ({score:.1f}%)")

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "cases": str(cases_path),
            "total": total,
            "passed": passed,
            "pass_rate": round(score, 2),
            "failed_ids": [c["id"] for c in cases if should_use_retrieval(c["query"]) != bool(c["expected_use_retrieval"])],
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[Eval] report saved: {report_path}")

    return 0 if passed == total else 2


if __name__ == "__main__":
    raise SystemExit(main())
