"""Estimator package exports.

Avoid importing every estimator eagerly because some research modules are
optional, moved, or have heavyweight third-party dependencies.
"""

from src.estimators.base import Estimator

__all__ = ["Estimator"]

