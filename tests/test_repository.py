from __future__ import annotations

import ast
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = (ROOT / "src", ROOT / "scripts")
EXCLUDED_METHOD_IDENTIFIERS = (
    "fed" + "lease",
    "fed" + "it",
    "fed" + "sa",
    "ffa" + "_lora",
    "_base" + "line",
    '"' + "ours" + '"',
    "--" + "method",
)


class RepositoryTest(unittest.TestCase):
    def test_python_files_parse(self) -> None:
        python_files = sorted((ROOT / "src").rglob("*.py"))
        self.assertTrue(python_files)
        for path in python_files:
            with self.subTest(path=path.relative_to(ROOT)):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_public_source_contains_only_fedweave(self) -> None:
        checked_suffixes = {".py", ".sh"}
        for source_root in SOURCE_ROOTS:
            for path in sorted(source_root.rglob("*")):
                if not path.is_file() or path.suffix not in checked_suffixes:
                    continue
                text = path.read_text(encoding="utf-8").lower()
                for identifier in EXCLUDED_METHOD_IDENTIFIERS:
                    with self.subTest(path=path.relative_to(ROOT), identifier=identifier):
                        self.assertNotIn(identifier, text)

    def test_public_entrypoints_exist(self) -> None:
        expected = (
            ROOT / "src" / "train.py",
            ROOT / "src" / "predict.py",
            ROOT / "scripts" / "train" / "fedweave.sh",
            ROOT / "scripts" / "eval" / "predict.sh",
        )
        for path in expected:
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertTrue(path.is_file())

    def test_shell_files_parse(self) -> None:
        shell_files = sorted((ROOT / "scripts").rglob("*.sh"))
        self.assertTrue(shell_files)
        completed = subprocess.run(
            ["bash", "-n", *(str(path) for path in shell_files)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
