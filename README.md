# Database-Agnostic AI Retrieval Prototype

This repository contains a minimal, vendor-neutral composition of the interactive and durable bulk paths. Production databases, model providers, brokers, object stores, and deployment infrastructure remain behind replaceable interfaces; the executable prototype uses deterministic in-memory fakes only.

Responsibility-focused packages include:

- `admission/` and `control_plane/`: exclusive routing, immutable configuration, policy, and budget binding.
- `relational/`, `model_routing/`, `security/`, and `validation/`: read-only data access, economical model admission, protected invocation, and deterministic validation.
- `interactive/`: absolute-deadline completion/fallback coordination.
- `bulk/` and `write_back/`: durable checkpoints/resume, terminal reporting, optional approval-gated effects, and recovery.
- `composition/` and `prototype/`: application wiring and local replaceable fake adapters.
- `cli/`: backward-compatible admission commands plus opt-in end-to-end smoke flows.

Legacy admission-only commands retain their existing JSON contract:

```powershell
python main.py interactive --request-id request-1 --query-plan customers --config-version v1
python main.py bulk --job-id job-1 --item item-1 --config-version v1
```

Exercise the composed prototype with `--execute`:

```powershell
python main.py interactive --request-id request-1 --query-plan customers --config-version v1 --execute
python main.py interactive --request-id request-2 --query-plan customers --config-version v1 --execute --fallback
python main.py bulk --job-id bulk-job --item item-1 --config-version v1 --execute --resume --write-back disabled
python main.py bulk --job-id bulk-job --item item-1 --config-version v1 --execute --write-back approved
```

The approved-write fixture intentionally uses the exact reviewed scope `bulk-job/item-1`; other scopes fail closed. Run all tests with `python -m pytest -q`.
