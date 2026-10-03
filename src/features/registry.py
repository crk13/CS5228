"""Factories give every fold a fresh feature block."""

from .base import FeatureBlock
from .basic import BasicFeatures
from .block import BlockFeatures

FEATURE_REGISTRY = {"basic": BasicFeatures, "block": BlockFeatures}
RESERVED_FEATURES = ("distance", "target_enc", "macro", "spatial")


def register_feature(name: str, factory=None):
    """Register a class or zero-argument factory; also usable as a decorator."""
    def register(value):
        if name in FEATURE_REGISTRY:
            raise ValueError(f"Feature {name!r} is already registered")
        FEATURE_REGISTRY[name] = value
        return value
    return register if factory is None else register(factory)


def make_feature(name: str) -> FeatureBlock:
    if name not in FEATURE_REGISTRY:
        if name in RESERVED_FEATURES:
            raise NotImplementedError(f"Feature {name!r} is reserved for a teammate to implement")
        raise ValueError(f"Unknown feature {name!r}; available: {list(FEATURE_REGISTRY)}")
    block = FEATURE_REGISTRY[name]()
    if not isinstance(block, FeatureBlock):
        raise TypeError(f"Factory for {name!r} must return a FeatureBlock")
    return block
