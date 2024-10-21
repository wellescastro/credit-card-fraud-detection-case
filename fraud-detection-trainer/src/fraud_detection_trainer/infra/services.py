from __future__ import annotations

import abc
import contextlib as ctx
import sys
import typing as T
from dataclasses import asdict, dataclass

import loguru
import mlflow
import mlflow.tracking as mt

from fraud_detection_trainer.infra.config import EnvironmentSettings


class Service(abc.ABC):
    """Base class for a global service."""

    @abc.abstractmethod
    def start(self) -> None:
        """Start the service."""

    def stop(self) -> None:
        """Stop the service."""
        # does nothing by default


@dataclass
class LoggerService(Service):
    """Service for logging messages.

    https://loguru.readthedocs.io/en/stable/api/logger.html

    Parameters:
        sink (str): logging output.
        level (str): logging level.
        format (str): logging format.
        colorize (bool): colorize output.
        serialize (bool): convert to JSON.
        backtrace (bool): enable exception trace.
        diagnose (bool): enable variable display.
        catch (bool): catch errors during log handling.
    """

    sink: str = "stderr"
    level: str = "DEBUG"
    format: str = (
        "<green>[{time:YYYY-MM-DD HH:mm:ss.SSS}]</green>"
        "<level>[{level}]</level>"
        "<cyan>[{name}:{function}:{line}]</cyan>"
        " <level>{message}</level>"
    )
    colorize: bool = True
    serialize: bool = False
    backtrace: bool = True
    diagnose: bool = False
    catch: bool = True

    @T.override
    def start(self) -> None:
        loguru.logger.remove()
        config = asdict(self)
        sinks = {"stderr": sys.stderr, "stdout": sys.stdout}
        config["sink"] = sinks.get(config["sink"], config["sink"])
        loguru.logger.add(**config)

    def logger(self) -> loguru.Logger:
        """Return the main logger.

        Returns:
            loguru.Logger: the main logger.
        """
        return loguru.logger


@dataclass
class MlflowService(Service):
    """Service for Mlflow tracking and registry.

    Parameters:
        environment_settings (EnvironmentSettings): environment settings.add()
        experiment_name (str): the name of tracking experiment.
        registry_name (str): the name of model registry.
        autolog_disable (bool): disable autologging.
        autolog_disable_for_unsupported_versions (bool): disable autologging for unsupported versions.
        autolog_exclusive (bool): If True, enables exclusive autologging.
        autolog_log_input_examples (bool): If True, logs input examples during autologging.
        autolog_log_model_signatures (bool): If True, logs model signatures during autologging.
        autolog_log_models (bool): If True, enables logging of models during autologging.
        autolog_log_datasets (bool): If True, logs datasets used during autologging.
        autolog_silent (bool): If True, suppresses all Mlflow warnings during autologging.
    """

    @dataclass
    class RunConfig:
        """Run configuration for Mlflow tracking.

        Parameters:
            name (str): name of the run.
            description (str | None): description of the run.
            tags (dict[str, T.Any] | None): tags for the run.
            log_system_metrics (bool | None): enable system metrics logging.
        """

        name: str
        description: str | None = None
        tags: dict[str, T.Any] | None = None
        log_system_metrics: bool | None = True

    # environment settings
    environment_settings: EnvironmentSettings
    # experiment
    experiment_name: str = "fraud_detection"
    # registry
    registry_name: str = "fraud_detection"
    # autolog
    autolog_disable: bool = False
    autolog_disable_for_unsupported_versions: bool = False
    autolog_exclusive: bool = False
    autolog_log_input_examples: bool = True
    autolog_log_model_signatures: bool = True
    autolog_log_models: bool = False
    autolog_log_datasets: bool = False
    autolog_silent: bool = False

    @T.override
    def start(self) -> None:
        # server uri
        mlflow.set_tracking_uri(uri=self.environment_settings.mlflow_tracking_uri)
        mlflow.set_registry_uri(uri=self.environment_settings.mlflow_registry_uri)
        # experiment
        mlflow.set_experiment(experiment_name=self.experiment_name)
        # autolog
        mlflow.autolog(
            disable=self.autolog_disable,
            disable_for_unsupported_versions=self.autolog_disable_for_unsupported_versions,
            exclusive=self.autolog_exclusive,
            log_input_examples=self.autolog_log_input_examples,
            log_model_signatures=self.autolog_log_model_signatures,
            log_datasets=self.autolog_log_datasets,
            silent=self.autolog_silent,
        )

    @ctx.contextmanager
    def run_context(
        self, run_config: RunConfig
    ) -> T.Generator[mlflow.ActiveRun, None, None]:
        """Yield an active Mlflow run and exit it afterwards.

        Args:
            run (str): run parameters.

        Yields:
            T.Generator[mlflow.ActiveRun, None, None]: active run context. Will be closed as the end of context.
        """
        with mlflow.start_run(
            run_name=run_config.name,
            tags=run_config.tags,
            description=run_config.description,
            log_system_metrics=run_config.log_system_metrics,
        ) as run:
            yield run

    def client(self) -> mt.MlflowClient:
        """Return a new Mlflow client.

        Returns:
            MlflowClient: the mlflow client.
        """
        return mt.MlflowClient(
            tracking_uri=self.environment_settings.mlflow_tracking_uri,
            registry_uri=self.environment_settings.mlflow_registry_uri,
        )
