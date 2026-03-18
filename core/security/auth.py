import hashlib

try:
    import bcrypt  # type: ignore
except Exception:  # pragma: no cover
    bcrypt = None


def hash_access_code_bcrypt(plain_key: str) -> str:
    """使用 bcrypt 生成访问口令哈希。"""
    if not bcrypt:
        raise RuntimeError("bcrypt 未安装，无法生成 bcrypt 哈希")
    return bcrypt.hashpw(plain_key.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_access_code(user_input: str, expected_hash: str) -> bool:
    """
    兼容两类哈希：
    1) bcrypt（推荐）: 以 $2a/$2b/$2y 开头
    2) 旧版 sha256 十六进制
    """
    if not expected_hash:
        return False

    if expected_hash.startswith(("$2a$", "$2b$", "$2y$")):
        if not bcrypt:
            return False
        try:
            return bool(bcrypt.checkpw(user_input.encode("utf-8"), expected_hash.encode("utf-8")))
        except Exception:
            return False

    user_hash = hashlib.sha256(user_input.encode("utf-8")).hexdigest()
    return user_hash == expected_hash

