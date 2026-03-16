from typing import Any

from core.observability.logger import log_event


def audit_security_event(action: str, result: str, reason: str = "", **fields: Any) -> None:
    log_event(
        "security_audit",
        trace_id="security",
        action=action,
        result=result,
        reason=reason,
        **fields,
    )
