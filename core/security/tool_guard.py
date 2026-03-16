from config.settings import Settings


def ensure_tool_allowed(settings: Settings, tool_name: str) -> None:
    allowed = set(settings.allowed_tools)
    if "*" in allowed:
        return
    if tool_name not in allowed:
        raise PermissionError(f"工具未在白名单中: {tool_name}")
