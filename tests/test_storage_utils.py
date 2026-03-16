import json
import tempfile
import unittest
from pathlib import Path

from infra.retrieval.storage_utils import (
    create_db_snapshot,
    read_recovery_marker,
    restore_db_snapshot,
    write_recovery_marker,
)


class TestStorageUtils(unittest.TestCase):
    def test_snapshot_restore_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "db"
            snap = Path(tmp) / "snap"
            db.mkdir(parents=True, exist_ok=True)
            (db / "a.txt").write_text("v1", encoding="utf-8")

            create_db_snapshot(str(db), str(snap))
            (db / "a.txt").write_text("v2", encoding="utf-8")

            ok = restore_db_snapshot(str(snap), str(db))
            self.assertTrue(ok)
            self.assertEqual((db / "a.txt").read_text(encoding="utf-8"), "v1")

    def test_recovery_marker_rw(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / ".recovery.json"
            payload = {"status": "in_progress"}
            write_recovery_marker(str(marker), payload)
            loaded = read_recovery_marker(str(marker))
            self.assertEqual(loaded, payload)


if __name__ == "__main__":
    unittest.main()
