from typing import Tuple

import torch

from .metric_modules import MultiClassificationTaskMetric


def get_multi_event_metric_module(
    logit_dim_per_event=[],
    metric_types: Tuple[str] = ("AUC",),
    comm_pg: torch.distributed.ProcessGroup = None,
):
    assert len(metric_types) == 1, "Only one ranking metric type is supported"
    eval_metrics_modules = MultiClassificationTaskMetric(
        logit_dim_per_event=logit_dim_per_event,
        number_of_tasks=len(logit_dim_per_event),
        metric_type=metric_types[0],
        process_group=comm_pg,
    ).cuda()
    return eval_metrics_modules
