"""Dataset integrity validation — checksums, schema, and manifest."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger


@dataclass
class ValidationResult:
    """Outcome of a dataset validation pass."""

    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.passed = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        lines = [f"Validation {status} — {len(self.errors)} errors, {len(self.warnings)} warnings"]
        for e in self.errors:
            lines.append(f"  ERROR: {e}")
        for w in self.warnings:
            lines.append(f"  WARN : {w}")
        return "\n".join(lines)


class DatasetValidator:
    """Validates a downloaded dataset directory.

    Checks performed:
    - Manifest file exists and parses correctly.
    - All listed files are present on disk.
    - SHA-256 checksums match for every file.
    - Required splits (train, val, test) are present.

    Args:
        dataset_dir: Root directory of the downloaded dataset.
        manifest_name: Name of the manifest file inside dataset_dir.
        verify_checksums: Whether to verify file checksums (can be slow for large datasets).
        fail_on_error: If True, raises on validation failure.
    """

    REQUIRED_SPLITS = ("train.parquet", "val.parquet", "test.parquet")

    def __init__(
        self,
        dataset_dir: Path,
        manifest_name: str = "manifest.json",
        verify_checksums: bool = True,
        fail_on_error: bool = True,
    ) -> None:
        self._dir = Path(dataset_dir)
        self._manifest_name = manifest_name
        self._verify_checksums = verify_checksums
        self._fail_on_error = fail_on_error

    def validate(self) -> ValidationResult:
        """Run all validation checks.

        Returns:
            ValidationResult with pass/fail status and details.

        Raises:
            ValueError: If fail_on_error is True and validation fails.
        """
        result = ValidationResult(passed=True)

        self._check_directory_exists(result)
        if not result.passed:
            self._maybe_raise(result)
            return result

        manifest = self._load_manifest(result)
        self._check_required_splits(result)

        if manifest and self._verify_checksums:
            self._verify_checksums_from_manifest(manifest, result)

        logger.info("{}", result.summary())

        self._maybe_raise(result)
        return result

    def _check_directory_exists(self, result: ValidationResult) -> None:
        if not self._dir.exists():
            result.add_error(f"Dataset directory does not exist: {self._dir}")
        elif not self._dir.is_dir():
            result.add_error(f"Path exists but is not a directory: {self._dir}")

    def _load_manifest(self, result: ValidationResult) -> Optional[dict]:
        manifest_path = self._dir / self._manifest_name
        if not manifest_path.exists():
            result.add_warning(f"Manifest file not found: {manifest_path}")
            return None
        try:
            with manifest_path.open("r") as fh:
                return json.load(fh)
        except json.JSONDecodeError as exc:
            result.add_error(f"Manifest JSON parse error: {exc}")
            return None

    def _check_required_splits(self, result: ValidationResult) -> None:
        for split_file in self.REQUIRED_SPLITS:
            path = self._dir / split_file
            if not path.exists():
                result.add_error(f"Required split file missing: {split_file}")

    def _verify_checksums_from_manifest(self, manifest: dict, result: ValidationResult) -> None:
        files = manifest.get("files", [])
        if not files:
            result.add_warning("Manifest contains no file entries — skipping checksum verification")
            return

        for entry in files:
            rel_path = entry.get("path", "")
            expected_sha256 = entry.get("sha256", "")
            if not rel_path or not expected_sha256:
                result.add_warning(f"Incomplete manifest entry: {entry}")
                continue

            file_path = self._dir / rel_path
            if not file_path.exists():
                result.add_error(f"File listed in manifest not found: {rel_path}")
                continue

            actual_sha256 = self._sha256(file_path)
            if actual_sha256 != expected_sha256:
                result.add_error(
                    f"Checksum mismatch for {rel_path}: "
                    f"expected {expected_sha256}, got {actual_sha256}"
                )
            else:
                logger.debug("Checksum OK: {}", rel_path)

    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def _maybe_raise(self, result: ValidationResult) -> None:
        if not result.passed and self._fail_on_error:
            raise ValueError(f"Dataset validation failed:\n{result.summary()}")
