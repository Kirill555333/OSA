# Changelog

## 1.5.1.6.1 — 2026-10-17

### Added
- `OSA_HOME` (default `~/.osa`) for data: memory, settings, logs, screenshots.
- `OSA_WORKSPACE` (default cwd) for allowed files in SafeFilesystem and SafeShell.
- `osa.utils.paths`: `resolve_osa_home`, `resolve_workspace`, `OSAHomeError`, `OSAWorkspaceError`.

### Changed
- `main.py`, `gui/__main__.py`, `gui/worker.py`, `tools/desktop.py` no longer use `Path.cwd()` for data; screenshots move to `OSA_HOME/data/screenshots`.
- Single source of version: `osa.version.__version__` (currently `1.5.1.6`).
- `pyproject.toml` uses `dynamic = ["version"]` from `osa.version`.

### Notes
- If `OSA_HOME` or `OSA_WORKSPACE` is explicitly set but unusable, OSA raises a clear error instead of silently falling back.

## 0.3.10

## 0.3.10

### Added
- Integrated autonomous execution entry point into Agent.
- Added explicit AUTONOMOUS-mode guard for autonomous runs.
- Added package version export.
- Added Agent autonomous execution contract tests.

### Architecture
- AutonomousLoop remains responsible for orchestration.
- TaskExecutor remains responsible for task execution.
- Verification remains responsible for result validation.
- Recovery remains responsible for transient task recovery.
- PermissionPolicy and AutonomousSafetyGate remain responsible for authorization.
