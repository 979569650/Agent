import subprocess


CHECKS = [
    ["python", "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"],
    ["python", "eval/run_eval.py"],
    [
        "python",
        "-m",
        "py_compile",
        "main.py",
        "rag_engine.py",
        "config/settings.py",
        "workflow/state.py",
        "workflow/graph.py",
        "workflow/nodes/agent_node.py",
        "infra/llm/openai_compatible.py",
        "infra/retrieval/model_reranker.py",
        "infra/retrieval/storage_utils.py",
        "infra/retrieval/search_pipeline.py",
        "core/domain/policies.py",
        "core/domain/retrieval_planner.py",
        "core/observability/logger.py",
        "core/observability/metrics.py",
        "core/observability/telemetry.py",
        "core/security/input_guard.py",
        "core/security/tool_guard.py",
        "core/security/audit.py",
        "core/security/auth.py",
        "core/security/rate_limit.py",
        "scripts/show_metrics_dashboard.py",
        "eval/compare_reports.py",
    ],
]


def run_cmd(cmd: list[str]) -> int:
    print(f"\n[QualityGate] Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd)
    return proc.returncode


def main() -> int:
    for cmd in CHECKS:
        code = run_cmd(cmd)
        if code != 0:
            print(f"[QualityGate] FAILED with code={code}")
            return code
    print("\n[QualityGate] ALL CHECKS PASSED ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
