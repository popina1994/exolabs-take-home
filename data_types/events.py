from enum import Enum
from dataclasses import dataclass

from .profiling import NodePerformanceProfile
from .common import NodeId, InstanceId, EventId
from .shards import ShardAssignments
from .topology import Multiaddr


@dataclass
class Host:
    multiaddr: Multiaddr


@dataclass
class Instance:
    instance_id: InstanceId
    instance_active: bool
    shard_assignments: ShardAssignments
    hosts: list[Host]


class EventType(str, Enum):
    """
    Here are all the unique kinds of events that can be sent over the network.
    """
    # Instance Events
    InstanceCreated = "InstanceCreated"
    InstanceDeleted = "InstanceDeleted"
    InstanceActivated = "InstanceActivated"
    InstanceDeactivated = "InstanceDeactivated"
    InstanceReplacedAtomically = "InstanceReplacedAtomically"

    # Node Performance Events
    WorkerStatusUpdated = "WorkerStatusUpdated"
    NodePerformanceMeasured = "NodePerformanceMeasured"


@dataclass
class BaseEvent:
    event_id: EventId


@dataclass
class InstanceCreated(BaseEvent):
    instance: Instance


@dataclass
class InstanceActivated(BaseEvent):
    instance_id: InstanceId


@dataclass
class InstanceDeactivated(BaseEvent):
    instance_id: InstanceId


@dataclass
class InstanceDeleted(BaseEvent):
    instance_id: InstanceId


@dataclass
class NodePerformanceMeasured(BaseEvent):
    node_id: NodeId
    node_profile: NodePerformanceProfile


@dataclass
class WorkerStatusUpdated(BaseEvent):
    node_id: NodeId
    worker_running: bool

Event = (
    InstanceCreated
    | InstanceActivated
    | InstanceDeactivated
    | InstanceDeleted
    | NodePerformanceMeasured
    | WorkerStatusUpdated
)
