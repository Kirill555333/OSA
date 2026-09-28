# Changelog

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
