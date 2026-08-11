"""Vendor-neutral relational data-access contracts."""

from ai_retrieval.relational.data_access import DataAccessLayer, DataAccessRejection
from ai_retrieval.relational.models import (
    DatabaseAccessDecision,
    NormalizedColumn,
    NormalizedResult,
    NormalizedType,
    OperationKind,
    OperationNode,
    OrderTerm,
    ParameterSpec,
    QueryPlan,
    QueryPlanReference,
    RawColumn,
    RawRelationalResult,
    VendorNeutralContract,
)
from ai_retrieval.relational.plans import QueryPlanRegistry, classify_operation
from ai_retrieval.relational.ports import ApprovedEffectAdapter, ReadRelationalAdapter

__all__ = [
    "ApprovedEffectAdapter",
    "DataAccessLayer",
    "DataAccessRejection",
    "DatabaseAccessDecision",
    "NormalizedColumn",
    "NormalizedResult",
    "NormalizedType",
    "OperationKind",
    "OperationNode",
    "OrderTerm",
    "ParameterSpec",
    "QueryPlan",
    "QueryPlanReference",
    "QueryPlanRegistry",
    "RawColumn",
    "RawRelationalResult",
    "ReadRelationalAdapter",
    "VendorNeutralContract",
    "classify_operation",
]
