from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, List, Optional
import pandas as pd


class BaseVectorStore(ABC):
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}

    # internal helpers
    def _set_index_host(self, host: str) -> None:
        raise NotImplementedError

    def _setup_index(self) -> None:
        raise NotImplementedError

    def _get_indexes(self) -> list:
        raise NotImplementedError

    def _check_if_embedding_exists(self, id: str, namespace: str) -> bool:
        raise NotImplementedError

    # public API as requested
    def add_ddl(self, ddl: str, **kwargs) -> str:
        raise NotImplementedError

    def add_documentation(self, doc: str, **kwargs) -> str:
        raise NotImplementedError

    def add_question_sql(self, question: str, sql: str, **kwargs) -> str:
        raise NotImplementedError

    def get_related_ddl(self, question: str, **kwargs) -> list:
        raise NotImplementedError

    def get_related_documentation(self, question: str, **kwargs) -> list:
        raise NotImplementedError

    def get_similar_question_sql(self, question: str, **kwargs) -> list:
        raise NotImplementedError

    def get_training_data(self, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    def remove_training_data(self, id: str, **kwargs) -> bool:
        raise NotImplementedError

    def generate_embedding(self, data: str, **kwargs) -> List[float]:
        raise NotImplementedError
