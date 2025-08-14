import os
import sys
import networkx as nx
from collections import namedtuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.graph_rag import GraphRAG

Document = namedtuple('Document', ['page_content', 'metadata'])


class DummyVectorStore:
    def similarity_search(self, query, k=5):
        return [
            Document(
                page_content="{'foo': 'bar'}",
                metadata={'db': 'db', 'table': 'table1', 'columns': ['col1']}
            )
        ]


def test_get_context_includes_related_tables():
    rag = GraphRAG()
    rag.vector_store = DummyVectorStore()

    rag.graph = nx.Graph()
    rag.graph.add_node('db.table1', columns=['col1'], frequent_values="{'foo': 'bar'}")
    rag.graph.add_node('db.table2', columns=['col2'], frequent_values="{'baz': 'qux'}")
    rag.graph.add_edge('db.table1', 'db.table2', weight=0.5)

    context = rag.get_context('query')

    assert 'Table: db.table1' in context
    assert 'Table: db.table2' in context
