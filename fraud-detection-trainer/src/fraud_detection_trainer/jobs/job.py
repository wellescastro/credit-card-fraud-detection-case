"""Base for high-level project jobs."""

import abc
import types as TS
import typing as T
from dataclasses import dataclass, field

from fraud_detection_trainer.infra import services

Locals = T.Dict[str, T.Any]


@dataclass
class Job(abc.ABC):
    """Base class for a job.

    use a job to execute runs in  context.
    e.g., to define common services like logger

    Parameters:
        logger_service (services.LoggerService): manage the logger system.
        mlflow_service (services.MlflowService): manage the mlflow system.
    """

    mlflow_service: services.MlflowService
    logger_service: services.LoggerService = field(default_factory=services.LoggerService)

    def __enter__(self) -> T.Self:
        """Enter the job context.

        Returns:
            T.Self: return the current object.
        """
        self.logger_service.start()
        logger = self.logger_service.logger()
        logger.debug("[START] Logger service: {}", self.logger_service)
        logger.debug("[START] Mlflow service: {}", self.mlflow_service)
        self.mlflow_service.start()
        return self

    def __exit__(
        self,
        exc_type: T.Type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: TS.TracebackType | None,
    ) -> T.Literal[False]:
        """Exit the job context.

        Args:
            exc_type (T.Type[BaseException] | None): ignored.
            exc_value (BaseException | None): ignored.
            exc_traceback (TS.TracebackType | None): ignored.

        Returns:
            T.Literal[False]: always propagate exceptions.
        """
        logger = self.logger_service.logger()
        logger.debug("[STOP] Mlflow service: {}", self.mlflow_service)
        self.mlflow_service.stop()
        logger.debug("[STOP] Logger service: {}", self.logger_service)
        self.logger_service.stop()
        return False  # re-raise exceptions

    @abc.abstractmethod
    def run(self) -> Locals:
        """Run the job in context.

        Returns:
            Locals: local job variables.
        """
