---
name: Immutable artifact store design
description: Enforcing write-once semantics for on-disk artifact stores keyed by version/hash, and keeping path resolution centralized.
---

When building a write-once (immutable) on-disk artifact store keyed by
version/hash:

- Enforce write-once at the file level with an atomic create-exclusive
  syscall (`os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)`), not a
  `path.exists()` check followed by a separate write. The latter is a
  check-then-act race: two concurrent writers can both pass the existence
  check before either writes, silently violating immutability.
  **Why:** code review caught this exact race in a first-pass implementation
  that used `if path.exists(): raise ... else: write()`.
  **How to apply:** any "write-once" or "immutable" store/cache layer with
  concurrent writers must use OS-level exclusive-create, not app-level
  existence checks.

- When a class both writes files (e.g. an export pipeline) and reports to a
  registry that also owns path-resolution policy, the writer must construct
  its storage handle *from the registry* (`registry.store`) rather than
  building its own separately-rooted store instance. Otherwise the two can
  silently diverge on where "the" canonical path for an artifact actually is.
  **Why:** flagged by review — a pipeline building its own
  `ArtifactStore(root=output_dir)` while the registry had its own default
  store meant `ArtifactRecord.path` and `registry.resolve_path()` could
  disagree.
  **How to apply:** prefer `getattr(registry, "store", None) or <fallback>`
  pattern so a caller-less/registry-less path still works for tests, but a
  registry-backed call is always authoritative.
