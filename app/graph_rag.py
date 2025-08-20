"""
Backward-compatibility shim.
The GraphRAG implementation was moved to app/graph/graph_rag.py.
Importing GraphRAG from this module will re-export the new implementation.
"""

from .graph import GraphRAG  # noqa: F401
