from __future__ import annotations
from typing import List, Optional
import pandas as pd
from ..base import BaseVectorStore


class ChromaVectorStore(BaseVectorStore):
    def _set_index_host(self, host: str) -> None:
        self.config["host"] = host

    def _setup_index(self) -> None:
        # TODO: initialize ChromaDB collection(s)
        pass

    def _get_indexes(self) -> list:
        return []

    def _check_if_embedding_exists(self, id: str, namespace: str) -> bool:
        return False

    def add_ddl(self, ddl: str, **kwargs) -> str:
        return "ok"

    def add_documentation(self, doc: str, **kwargs) -> str:
        return "ok"

    def add_question_sql(self, question: str, sql: str, **kwargs) -> str:
        return "ok"

    def get_related_ddl(self, question: str, **kwargs) -> list:
        return []

    def get_related_documentation(self, question: str, **kwargs) -> list:
        return []

    def get_similar_question_sql(self, question: str, **kwargs) -> list:
        return []

    def get_training_data(self, **kwargs) -> pd.DataFrame:
        return pd.DataFrame()

    def remove_training_data(self, id: str, **kwargs) -> bool:
        return True

    def generate_embedding(self, data: str, **kwargs) -> List[float]:
        # Placeholder embedding
        return []
