import sys
from typing import Any

from loguru import logger

from .base_log import BaseLog


class StdOutLog(BaseLog):
    def filter(self, record: dict[str, Any]):
        return True

    def config(self, *args, **kwargs):
        return {
            "sink": sys.stdout,
            "format": "<green>[{time:YYYY-MM-DD at HH:mm:ss}]</green> "
            "<level>{level:10}</level> "
            "<cyan>{line:4}: {module}:{function}</cyan> "
            "<magenta>process:{process}\tthread:{thread}</magenta> <yellow>|</yellow> "
            "<level>{message}</level>",
            "filter": self.filter,
            "enqueue": True,
        }


class SuccessLog(BaseLog):
    def filter(self, record: dict[str, Any]):
        return record["level"].name == "SUCCESS"

    def config(self, *args, **kwargs):
        return {
            "sink": f"{self.ABS_PATH}/"
            + "Session_{time:YYYY_MM_DD}/"
            + f"/{self.APP_SUB_DIR}/{self.__class__.__name__}.log",
            "format": "<green>[{time:YYYY-MM-DD at HH:mm:ss}]</green> "
            "<cyan>{line:4}: {module}:{function}</cyan> "
            "<yellow>|</yellow> "
            "<level>{message}</level>",
            "level": "SUCCESS",
            "filter": self.filter,
            "enqueue": True,
            "rotation": "00:00",
        }


class InfoLog(BaseLog):
    def filter(self, record: dict[str, Any]):
        return record["level"].name == "INFO"

    def config(self, *args, **kwargs):
        return {
            "sink": f"{self.ABS_PATH}/"
            + "Session_{time:YYYY_MM_DD}/"
            + f"/{self.APP_SUB_DIR}/{self.__class__.__name__}.log",
            "format": "<green>[{time:YYYY-MM-DD at HH:mm:ss}]</green> "
            "<cyan>{line:4}: {module}:{function}</cyan> "
            "<yellow>|</yellow> "
            "<level>{message}</level>",
            "level": "INFO",
            "filter": self.filter,
            "enqueue": True,
            "rotation": "00:00",
        }


class WarningLog(BaseLog):
    def filter(self, record: dict[str, Any]):
        return record["level"].name == "WARNING"

    def config(self, *args, **kwargs):
        return {
            "sink": f"{self.ABS_PATH}/"
            + "Session_{time:YYYY_MM_DD}/"
            + f"/{self.APP_SUB_DIR}/{self.__class__.__name__}.log",
            "format": "<green>[{time:YYYY-MM-DD at HH:mm:ss}]</green> "
            "<cyan>{line:4}: {module}:{function}</cyan> "
            "<yellow>|</yellow> "
            "<level>{message}</level>",
            "level": "WARNING",
            "filter": self.filter,
            "enqueue": True,
            "rotation": "00:00",
        }


class ErrorLog(BaseLog):
    def filter(self, record: dict[str, Any]):
        return record["level"].name == "ERROR"

    def config(self, *args, **kwargs):
        return {
            "sink": f"{self.ABS_PATH}/"
            + "Session_{time:YYYY_MM_DD}/"
            + f"/{self.APP_SUB_DIR}/{self.__class__.__name__}.log",
            "format": "<green>[{time:YYYY-MM-DD at HH:mm:ss}]</green> "
            "<level>{level:10}</level> "
            "<cyan>{line:4}: {module}:{function}</cyan> "
            "<magenta>process:{process}\tthread:{thread}</magenta> <yellow>|</yellow> "
            "<level>{message}</level>",
            "backtrace": True,
            "level": "ERROR",
            "filter": self.filter,
            "enqueue": True,
            "rotation": "00:00",
        }


class CriticalLog(BaseLog):
    def filter(self, record: dict[str, Any]):
        return record["level"].name == "CRITICAL"

    def config(self, *args, **kwargs):
        return {
            "sink": f"{self.ABS_PATH}/"
            + "Session_{time:YYYY_MM_DD}/"
            + f"/{self.APP_SUB_DIR}/{self.__class__.__name__}.log",
            "format": "<green>[{time:YYYY-MM-DD at HH:mm:ss}]</green> "
            "<level>{level:10}</level> "
            "<cyan>{line:4}: {module}:{function}</cyan> "
            "<magenta>process:{process} thread:{thread}</magenta> <yellow>|</yellow> "
            "<level>{message}</level>",
            "backtrace": True,
            "level": "CRITICAL",
            "filter": self.filter,
            "enqueue": True,
            "rotation": "00:00",
        }


loggers = [log() for log in BaseLog.__subclasses__() if log.IS_INIT]
logger.configure(
    handlers=[log.config() for log in loggers],
)
