"""Production replay uses only the original native SONIC execution path.

The previous synchronous loop is archived as non-executable audit text in diagnostics/.
No automatic fallback is permitted.
"""
from .native_replay import replay
