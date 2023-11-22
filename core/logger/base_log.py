import os
from abc import ABC, abstractmethod
from typing import Any

__all__ = [
    "BaseLog",
]


class BaseLog(ABC):
    ABS_PATH = os.path.abspath(f"logs/")

    APP_SUB_DIR = "ApplicationLogs"
    TRANSACTION_SUB_DIR = "TransactionLogs"

    IS_INIT = True

    @abstractmethod
    def filter(self, record: dict[str, Any]):
        pass

    @abstractmethod
    def config(self, *args, **kwargs):
        pass
