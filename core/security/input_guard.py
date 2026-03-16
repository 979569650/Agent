from dataclasses import dataclass
import re


@dataclass
class InjectionCheckResult:
    blocked: bool
    reason: str = ""


INJECTION_PATTERNS = [
    (r"(?i)ignore\s+(all\s+)?previous\s+instructions", "override_system_instructions"),
    (r"(?i)(reveal|show|print).*(system\s+prompt|hidden\s+prompt)", "prompt_exfiltration"),
    (r"(?i)developer\s+message", "developer_message_probe"),
    (r"(?i)jailbreak|do anything now|dan", "jailbreak_attempt"),
]


def check_prompt_injection(query: str) -> InjectionCheckResult:
    text = (query or "").strip()
    if not text:
        return InjectionCheckResult(blocked=False)

    for pattern, reason in INJECTION_PATTERNS:
        if re.search(pattern, text):
            return InjectionCheckResult(blocked=True, reason=reason)
    return InjectionCheckResult(blocked=False)
