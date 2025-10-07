from .data_types.topology import Topology
from .data_types.events import Event, Instance, Host, InstanceCreated, InstanceDeleted
from .data_types.shards import CreateInstanceCommand
from .data_types.common import InstanceId, EventId

from copy import deepcopy

from .placement_utils import (
    filter_cycles_by_memory,
    get_hosts_from_subgraph,
    get_shard_assignments,
    get_smallest_cycles,
)

def get_instance_placements(
    command: CreateInstanceCommand,
    topology: Topology,
    current_instances: dict[InstanceId, Instance],
    instance_id: InstanceId | None = None,
) -> dict[InstanceId, Instance]:
    if instance_id is None:
        instance_id = InstanceId()
    available_models = [current_instances[instance].shard_assignments.model_id for instance in current_instances]
    if command.model_meta.model_id in available_models:
        raise ValueError(f"Instance for {command.model_meta.model_id} already exists")
    
    all_nodes = topology.list_nodes()
    cycles = topology.get_cycles()
    # we can also always just have a node on its own
    singleton_cycles = [[node] for node in all_nodes]
    candidate_cycles = cycles + singleton_cycles

    cycles_with_sufficient_memory = filter_cycles_by_memory(candidate_cycles, command.model_meta.storage_size_kilobytes * 1024)
    if not cycles_with_sufficient_memory:
        raise ValueError("No cycles found with sufficient memory")

    smallest_cycles = get_smallest_cycles(cycles_with_sufficient_memory)
    selected_cycle = max(smallest_cycles, key=lambda cycle: sum(node.node_profile.memory.ram_available for node in cycle))
    
    shard_assignments = get_shard_assignments(command.model_meta, selected_cycle)
    
    cycle_digraph: Topology = topology.get_subgraph_from_nodes(selected_cycle)
    hosts: list[Host] = get_hosts_from_subgraph(cycle_digraph)
    
    target_instances = deepcopy(current_instances)
    target_instances[instance_id] = Instance(
        instance_id=instance_id,
        instance_active=True,
        shard_assignments=shard_assignments,
        hosts=hosts
    )
    return target_instances


def get_transition_events(
    current_instances: dict[InstanceId, Instance],
    target_instances: dict[InstanceId, Instance],
) -> list[Event]:
    events: list[Event] = []

    # find instances to create
    for instance_id, instance in target_instances.items():
        if instance_id not in current_instances:
            events.append(
                InstanceCreated(
                    event_id = EventId(),
                    instance=instance,
                )
            )

    # find instances to delete
    for instance_id in current_instances:
        if instance_id not in target_instances:
            events.append(
                InstanceDeleted(
                    event_id = EventId(),
                    instance_id=instance_id,
                )
            )
    
    return events

