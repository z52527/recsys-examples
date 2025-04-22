from typing import Optional

from configs import HSTUConfig, RankingConfig, RetrievalConfig
from megatron.core.distributed import DistributedDataParallelConfig
from model.ranking_gr import RankingGR
from model.retrieval_gr import RetrievalGR

from . import ranking_gr, retrieval_gr

__all__ = ["ranking_gr", "retrieval_gr"]


def get_ranking_model(
    hstu_config: HSTUConfig,
    task_config: RankingConfig,
    ddp_config: Optional[DistributedDataParallelConfig] = None,
) -> RankingGR:
    """
    Get a ranking model.

    Args:
        hstu_config (HSTUConfig): The HSTU configuration.
        task_config (RankingConfig): The ranking task configuration.
        ddp_config (DistributedDataParallelConfig): The distributed data parallel configuration.

    Returns:
        RankingGR: The ranking model.
    """
    assert isinstance(task_config, RankingConfig), "please provide a ranking config"
    return RankingGR(
        hstu_config=hstu_config, task_config=task_config, ddp_config=ddp_config
    )


def get_retrieval_model(
    hstu_config: HSTUConfig,
    task_config: RetrievalConfig,
    ddp_config: Optional[DistributedDataParallelConfig] = None,
) -> RetrievalGR:
    """
    Get a retrieval model.

    Args:
        hstu_config (HSTUConfig): The HSTU configuration.
        task_config (RetrievalConfig): The retrieval task configuration.
        ddp_config (DistributedDataParallelConfig): The distributed data parallel configuration.

    Returns:
        RetrievalGR: The retrieval model.
    """
    assert isinstance(task_config, RetrievalConfig), "please provide a retrieval config"
    return RetrievalGR(
        hstu_config=hstu_config, task_config=task_config, ddp_config=ddp_config
    )
