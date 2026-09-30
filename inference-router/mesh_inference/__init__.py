"""Inference router: model catalog, providers, failover and per-call cost."""

from mesh_inference.breaker import CircuitBreaker
from mesh_inference.catalog import ModelCatalog, ModelInfo, default_catalog
from mesh_inference.router import InferenceRouter
from mesh_inference.types import (
    Attempt,
    Message,
    ModelRequest,
    ModelResponse,
    SystemSegment,
    ToolDef,
)

__all__ = [
    "Attempt",
    "CircuitBreaker",
    "InferenceRouter",
    "Message",
    "ModelCatalog",
    "ModelInfo",
    "ModelRequest",
    "ModelResponse",
    "SystemSegment",
    "ToolDef",
    "default_catalog",
]
