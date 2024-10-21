from collections import defaultdict

import click
import yaml

from fraud_detection_trainer.infra.config import EnvironmentSettings, JobConfiguration
from fraud_detection_trainer.jobs.training_job import TrainingJob


@click.command()
@click.option(
    "--config_path", type=str, required=True, help="Path to the training config"
)
def main(config_path: str):
    with open(config_path) as stream:
        config = yaml.safe_load(stream)

        config = defaultdict(lambda: defaultdict(dict), config)

        environment_settings = EnvironmentSettings()

        if config["job"]["kind"] == "train":
            job_config = JobConfiguration(**config["job"]["configuration"])
            training_job = TrainingJob(
                job_config=job_config, environment_settings=environment_settings
            )
            training_job.run()
        else:
            raise ValueError(f"Invalid job kind: {config["job"]["kind"]}")


if __name__ == "__main__":
    main()
