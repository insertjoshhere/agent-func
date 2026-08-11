"""Thread-safe current policy and approval resolution."""

from dataclasses import replace
from threading import RLock

from ai_retrieval.write_back.models import ApprovalRecord, AuthorizationSnapshot, WriteBackPolicy


class InMemoryWriteBackAuthorization:
    """Prototype authority whose resolve call returns one current atomic snapshot."""

    def __init__(
        self,
        policies: tuple[WriteBackPolicy, ...] = (),
        approvals: tuple[ApprovalRecord, ...] = (),
    ) -> None:
        self._policies = {policy.version: policy for policy in policies}
        self._approvals = {approval.reference: approval for approval in approvals}
        self._revoked: set[str] = set()
        self._lock = RLock()

    def resolve(self, policy_version: str, approval_reference: str) -> AuthorizationSnapshot:
        with self._lock:
            approval = self._approvals.get(approval_reference)
            if approval is not None and approval.reference in self._revoked and not approval.revoked:
                approval = replace(approval, revoked=True)
            return AuthorizationSnapshot(self._policies.get(policy_version), approval)

    def revoke(self, approval_reference: str) -> None:
        with self._lock:
            self._revoked.add(approval_reference)
