import logging
from collections.abc import Mapping, Sequence

from clearml import Logger, Task, TaskTypes
from clearml.backend_api.session.defs import MissingConfigError

from ..distributed import rank_zero

logger = logging.getLogger(__name__)


def start_clearml_logger(
    project_name: str | None,
    task_name: str | None,
    task_type: str | TaskTypes = TaskTypes.training,
    tags: Sequence[str] | None = None,
    reuse_last_task_id: bool | str = True,
    continue_last_task: bool | str | int = False,
    output_uri: str | bool | None = None,
    auto_connect_arg_parser: bool | Mapping[str, bool] = True,
    auto_connect_frameworks: bool | Mapping[str, bool] = True,
    auto_resource_monitoring: bool = True,
    auto_connect_streams: bool | Mapping[str, bool] = True,
    deferred_init: bool = False,
) -> None | Logger:
    """
    Just a fuse function to quickly start the clearml logger. It sets up patches to pytorch lightning logging hooks so it doesn't need to be passed to any lightning logger.
    This function also checks if the NODE_RANK and LOCAL_RANK env variables have been set. In which case clearml will only be initialized on global rank 0.
    For information on all the arguments please see: https://clear.ml/docs/latest/docs/references/sdk/task/ or https://github.com/allegroai/clearml/blob/master/clearml/task.py.

    General Clearml instructions:
    Unless using offline mode, to use clearml, you must first make an account on their website https://app.clear.ml/login?redirect=%2Fsettings%2Fworkspace-configuration.
    Then, you must create a ~/clearml.conf file and specify server address as shown here https://clear.ml/docs/latest/docs/configs/clearml_conf/.
    Otherwise, offline mode instructions can be found here: https://clear.ml/docs/latest/docs/guides/set_offline/

    """
    if rank_zero():
        try:
            task = Task.init(
                project_name=project_name,
                task_name=task_name,
                task_type=task_type,
                tags=tags,
                reuse_last_task_id=reuse_last_task_id,
                continue_last_task=continue_last_task,
                output_uri=output_uri,
                auto_connect_arg_parser=auto_connect_arg_parser,
                auto_connect_frameworks=auto_connect_frameworks,
                auto_resource_monitoring=auto_resource_monitoring,
                auto_connect_streams=auto_connect_streams,
                deferred_init=deferred_init,
            )
        except MissingConfigError:
            logger.warning("No clearml config found. Continuing without clearml.")
            return None

        return task.get_logger()
