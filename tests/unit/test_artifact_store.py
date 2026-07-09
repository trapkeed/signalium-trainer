"""Unit tests for the S21A immutable, versioned on-disk artifact store."""

from __future__ import annotations

from pathlib import Path

import pytest

from trainer.research_lab.artifact_store import (
    ArtifactImmutabilityError,
    ArtifactKind,
    ArtifactStore,
)


@pytest.fixture()
def store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(root=tmp_path / "store")


class TestVersionLayout:
    def test_version_dir_is_root_slash_version(self, store: ArtifactStore, tmp_path: Path) -> None:
        assert store.version_dir("v1") == tmp_path / "store" / "v1"

    def test_path_for_without_kind(self, store: ArtifactStore) -> None:
        path = store.path_for("v1", "weights.pt")
        assert path == store.version_dir("v1") / "weights.pt"

    def test_path_for_with_kind_namespaces_under_subdir(self, store: ArtifactStore) -> None:
        path = store.path_for("v1", "report.json", kind=ArtifactKind.VALIDATION_REPORT)
        assert path == store.version_dir("v1") / "validation_report" / "report.json"

    def test_sanitizes_unsafe_version_segment(self, store: ArtifactStore) -> None:
        path = store.version_dir("../../etc")
        assert ".." not in path.parts
        assert store.root in path.parents

    def test_rejects_empty_or_dot_segment(self, store: ArtifactStore) -> None:
        with pytest.raises(ValueError):
            store.version_dir("..")
        with pytest.raises(ValueError):
            store.version_dir("")


class TestReserveVersion:
    def test_reserve_creates_directory_with_no_real_artifact_files(self, store: ArtifactStore) -> None:
        directory = store.reserve_version("v1")
        assert directory.exists()
        assert directory.is_dir()
        # Only the internal reservation sentinel exists — no artifact files.
        assert store.list_version("v1") == []

    def test_reserve_is_a_single_shot_atomic_claim(self, store: ArtifactStore) -> None:
        """reserve_version is not idempotent — the second call must fail even
        though the directory is still empty, since the sentinel it creates
        is the single-writer claim two concurrent exporters race on."""
        store.reserve_version("v1")
        with pytest.raises(ArtifactImmutabilityError):
            store.reserve_version("v1")

    def test_reserve_rejects_already_populated_version(self, store: ArtifactStore) -> None:
        store.write_bytes("v1", "weights.pt", b"abc")
        with pytest.raises(ArtifactImmutabilityError):
            store.reserve_version("v1")

    def test_is_populated(self, store: ArtifactStore) -> None:
        assert store.is_populated("v1") is False
        store.write_bytes("v1", "weights.pt", b"abc")
        assert store.is_populated("v1") is True


class TestWrites:
    def test_write_bytes_creates_file_with_matching_content(self, store: ArtifactStore) -> None:
        path = store.write_bytes("v1", "weights.pt", b"hello-weights")
        assert path.exists()
        assert path.read_bytes() == b"hello-weights"

    def test_write_bytes_rejects_overwrite_of_existing_file(self, store: ArtifactStore) -> None:
        store.write_bytes("v1", "weights.pt", b"first")
        with pytest.raises(ArtifactImmutabilityError):
            store.write_bytes("v1", "weights.pt", b"second")
        # original content must be untouched
        assert store.read_bytes("v1", "weights.pt") == b"first"

    def test_write_bytes_allows_different_filenames_in_same_version(self, store: ArtifactStore) -> None:
        store.write_bytes("v1", "weights.pt", b"w")
        store.write_bytes("v1", "config.json", b"{}")
        assert store.read_bytes("v1", "weights.pt") == b"w"
        assert store.read_bytes("v1", "config.json") == b"{}"

    def test_write_text(self, store: ArtifactStore) -> None:
        path = store.write_text("v1", "config.json", '{"a": 1}')
        assert path.read_text() == '{"a": 1}'

    def test_import_file_copies_bytes(self, store: ArtifactStore, tmp_path: Path) -> None:
        src = tmp_path / "external_weights.pt"
        src.write_bytes(b"external-bytes")
        path = store.import_file("v1", "weights.pt", src)
        assert path.read_bytes() == b"external-bytes"

    def test_written_file_is_read_only_on_disk(self, store: ArtifactStore) -> None:
        path = store.write_bytes("v1", "weights.pt", b"abc")
        mode = path.stat().st_mode & 0o777
        assert mode & 0o222 == 0, "file must not be writable after being stored"

    def test_write_into_different_versions_does_not_collide(self, store: ArtifactStore) -> None:
        store.write_bytes("v1", "weights.pt", b"v1-bytes")
        store.write_bytes("v2", "weights.pt", b"v2-bytes")
        assert store.read_bytes("v1", "weights.pt") == b"v1-bytes"
        assert store.read_bytes("v2", "weights.pt") == b"v2-bytes"


class TestConcurrency:
    def test_concurrent_reserve_version_only_one_writer_wins(self, store: ArtifactStore) -> None:
        """Two threads racing to reserve the same version: exactly one must
        succeed and exactly one must raise ArtifactImmutabilityError — never
        both succeeding (which would violate write-once) and never both
        failing."""
        import threading

        results: list[str] = []
        barrier = threading.Barrier(2)

        def attempt() -> None:
            barrier.wait()
            try:
                store.reserve_version("v-race")
                results.append("ok")
            except ArtifactImmutabilityError:
                results.append("rejected")

        threads = [threading.Thread(target=attempt) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert sorted(results) == ["ok", "rejected"]

    def test_concurrent_write_bytes_same_file_only_one_writer_wins(self, store: ArtifactStore) -> None:
        """Two threads racing to write the same (version, filename): exactly
        one succeeds, the other raises, and the surviving content is never a
        mix of both writers' bytes."""
        import threading

        results: list[str] = []
        barrier = threading.Barrier(2)

        def attempt(payload: bytes) -> None:
            barrier.wait()
            try:
                store.write_bytes("v-race", "weights.pt", payload)
                results.append("ok")
            except ArtifactImmutabilityError:
                results.append("rejected")

        threads = [
            threading.Thread(target=attempt, args=(b"writer-a",)),
            threading.Thread(target=attempt, args=(b"writer-b",)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert sorted(results) == ["ok", "rejected"]
        content = store.read_bytes("v-race", "weights.pt")
        assert content in (b"writer-a", b"writer-b")


class TestListing:
    def test_list_version_returns_all_files(self, store: ArtifactStore) -> None:
        store.write_bytes("v1", "weights.pt", b"w")
        store.write_bytes("v1", "config.json", b"{}")
        names = {p.name for p in store.list_version("v1")}
        assert names == {"weights.pt", "config.json"}

    def test_list_version_empty_for_unknown_version(self, store: ArtifactStore) -> None:
        assert store.list_version("does-not-exist") == []

    def test_list_versions(self, store: ArtifactStore) -> None:
        store.write_bytes("v1", "weights.pt", b"w")
        store.write_bytes("v2", "weights.pt", b"w")
        assert store.list_versions() == ["v1", "v2"]
