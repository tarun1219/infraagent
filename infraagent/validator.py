"""
Compatibility shim: ``infraagent.validator`` → ``infraagent.validators``

agent.py imports from ``infraagent.validator`` (singular); the implementation
lives in ``infraagent.validators`` (plural). This shim re-exports everything
so both import paths work.
"""
from infraagent.validators import (  # noqa: F401
    MultiLayerValidator,
    ValidationReport,
    ValidationError,
    ValidationLayer,
    Severity,
)
