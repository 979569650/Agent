import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Show local observability dashboard from JSONL logs.")
    p.add_argument("--log", default="observability/events.jsonl", help="JSONL log path")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    path = Path(args.log)
    if not path.exists():
        print(f"[Dashboard] log file not found: {path}")
        return 1

    total = 0
    response_count = 0
    latency_sum = 0.0
    retrieval_count = 0
    retrieval_hit = 0
    error_count = 0

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except Exception:
            continue

        total += 1
        event = item.get("event", "")
        if event == "response_generated":
            response_count += 1
            latency_sum += float(item.get("latency_ms", 0) or 0)
        elif event == "retrieval_done":
            retrieval_count += 1
            if bool(item.get("hit", False)):
                retrieval_hit += 1
        elif event in {"response_failed", "security_audit"}:
            # security_audit 中 result=error 或 blocked 也可视作风控异常观测
            if event == "response_failed" or str(item.get("result", "")).lower() in {"error", "blocked"}:
                error_count += 1

    avg_latency = (latency_sum / response_count) if response_count else 0.0
    hit_rate = (retrieval_hit / retrieval_count * 100.0) if retrieval_count else 0.0
    err_rate = (error_count / max(total, 1) * 100.0)

    print("\n=== 智能笔记助手 Metrics Dashboard ===")
    print(f"Log File           : {path}")
    print(f"Total Events       : {total}")
    print(f"Avg Latency (ms)   : {avg_latency:.2f}")
    print(f"Retrieval Hit Rate : {hit_rate:.2f}% ({retrieval_hit}/{retrieval_count})")
    print(f"Error Rate         : {err_rate:.2f}% ({error_count}/{total})")
    print("===================================\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
