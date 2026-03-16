import hashlib
import shutil
import json
import os
import tempfile
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator


@contextmanager
def index_lock(db_path: str, lock_file: str, timeout: int = 30, poll_interval: float = 0.2) -> Iterator[None]:
    """基础文件锁，避免并发重建索引导致脏写。"""
    os.makedirs(db_path, exist_ok=True)
    start = time.time()
    fd = None

    while True:
        try:
            fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.write(fd, str(os.getpid()).encode("utf-8"))
            break
        except FileExistsError:
            if (time.time() - start) >= timeout:
                raise RuntimeError("索引构建锁等待超时，请稍后重试")
            time.sleep(poll_interval)

    try:
        yield
    finally:
        try:
            if fd is not None:
                os.close(fd)
            if os.path.exists(lock_file):
                os.remove(lock_file)
        except Exception:
            pass


def atomic_write_json(target_path: str, payload: Any):
    """JSON 原子写：先写临时文件，再 replace。"""
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix="tmp_", suffix=".json", dir=os.path.dirname(target_path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, target_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def calculate_file_hash(filepath: str) -> str:
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def collect_hashes(data_dir: str, allowed_exts: set[str]) -> Dict[str, str]:
    current_hashes: Dict[str, str] = {}
    if not os.path.exists(data_dir):
        return current_hashes

    for root, _, files in os.walk(data_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext not in allowed_exts:
                continue
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, data_dir)
            try:
                current_hashes[rel_path] = calculate_file_hash(file_path)
            except Exception:
                continue
    return current_hashes


def write_recovery_marker(marker_file: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(marker_file), exist_ok=True)
    atomic_write_json(marker_file, payload)


def read_recovery_marker(marker_file: str) -> Dict[str, Any] | None:
    if not os.path.exists(marker_file):
        return None
    try:
        with open(marker_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else None
    except Exception:
        return None


def create_db_snapshot(db_path: str, snapshot_dir: str, exclude_names: set[str] | None = None) -> None:
    exclude = exclude_names or set()
    if os.path.exists(snapshot_dir):
        shutil.rmtree(snapshot_dir, ignore_errors=True)
    os.makedirs(snapshot_dir, exist_ok=True)

    if not os.path.exists(db_path):
        return

    for name in os.listdir(db_path):
        if name in exclude:
            continue
        src = os.path.join(db_path, name)
        dst = os.path.join(snapshot_dir, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)


def restore_db_snapshot(snapshot_dir: str, db_path: str, preserve_names: set[str] | None = None) -> bool:
    if not os.path.exists(snapshot_dir):
        return False

    preserve = preserve_names or set()
    os.makedirs(db_path, exist_ok=True)

    for name in os.listdir(db_path):
        if name in preserve:
            continue
        path = os.path.join(db_path, name)
        try:
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                os.remove(path)
        except Exception:
            pass

    for name in os.listdir(snapshot_dir):
        src = os.path.join(snapshot_dir, name)
        dst = os.path.join(db_path, name)
        try:
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
        except Exception:
            return False
    return True
