from dataclasses import dataclass

from .common import ModelId, NodeId, RunnerId


@dataclass
class ModelMetadata:
    model_id: ModelId
    pretty_name: str
    storage_size_kilobytes: int
    n_layers: int


@dataclass
class CreateInstanceCommand:
    model_meta: ModelMetadata


@dataclass
class BaseShardMetadata:
    """
    Defines a specific shard of the model that is ready to be run on a device.
    Replaces previous `Shard` object.
    """
    model_meta: ModelMetadata
    device_rank: int
    world_size: int


@dataclass
class PipelineShardMetadata(BaseShardMetadata):
    """
    Pipeline parallelism shard meta.

    Layers are represented as a half-open interval [start_layer, end_layer),
    where start_layer is inclusive and end_layer is exclusive.
    """
    start_layer: int
    end_layer: int
    n_layers: int

    @property
    def is_first_layer(self) -> bool:
        return self.start_layer == 0

    @property
    def is_last_layer(self) -> bool:
        return self.end_layer == self.n_layers

ShardMetadata = PipelineShardMetadata # Future union here

@dataclass
class ShardAssignments:
    model_id: ModelId
    runner_to_shard: dict[RunnerId, ShardMetadata]
    node_to_runner: dict[NodeId, RunnerId]
