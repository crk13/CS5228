"""Public feature API. Reserved blocks fail explicitly until implemented."""

from .base import FeatureBlock
from .basic import BasicFeatures, BASIC_COLUMNS
from .block import BlockFeatures, BLOCK_COLUMNS
from .pipeline import FeaturePipeline, BASE_COLUMNS
from .registry import register_feature, make_feature, FEATURE_REGISTRY, RESERVED_FEATURES
