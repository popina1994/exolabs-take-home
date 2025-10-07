from .data_types.topology import Topology, TopologyNode
from .data_types.common import NodeId, RunnerId
from .data_types.events import Host
from .data_types.shards import ModelMetadata
from .data_types.shards import ShardAssignments
from .data_types.shards import PipelineShardMetadata

def filter_cycles_by_memory(cycles: list[list[TopologyNode]], required_memory: int) -> list[list[TopologyNode]]:
    filtered_cycles: list[list[TopologyNode]] = []
    for cycle in cycles:
        total_mem = sum(node.node_profile.memory.ram_available for node in cycle)
        if total_mem >= required_memory:
            filtered_cycles.append(cycle)
    return filtered_cycles


def get_smallest_cycles(cycles: list[list[TopologyNode]]) -> list[list[TopologyNode]]:
    min_nodes = min(len(cycle) for cycle in cycles)
    return [cycle for cycle in cycles if len(cycle) == min_nodes]

def get_shard_assignments(
    model_meta: ModelMetadata,
    selected_cycle: list[TopologyNode],
) -> ShardAssignments:
    cycle_memory = sum(node.node_profile.memory.ram_available for node in selected_cycle)
    total_layers = model_meta.n_layers
    runner_to_shard: dict[RunnerId, PipelineShardMetadata] = {}
    node_to_runner: dict[NodeId, RunnerId] = {}

    layers_assigned = 0
    for i, node in enumerate(selected_cycle):
        if i == len(selected_cycle) - 1:
            node_layers = total_layers - layers_assigned
        else:
            node_layers = round(total_layers * (node.node_profile.memory.ram_available / cycle_memory))
            node_layers = max(1, node_layers)

        runner_id = RunnerId()
        shard = PipelineShardMetadata(
            model_meta=model_meta,
            device_rank=i,
            world_size=len(selected_cycle),
            start_layer=layers_assigned,
            end_layer=layers_assigned + node_layers,
            n_layers=total_layers
        )

        runner_to_shard[runner_id] = shard
        node_to_runner[node.node_id] = runner_id
        layers_assigned += node_layers

    shard_assignments = ShardAssignments(
        model_id=model_meta.model_id,
        runner_to_shard=runner_to_shard,
        node_to_runner=node_to_runner
    )

    return shard_assignments


def get_hosts_from_subgraph(cycle_digraph: Topology) -> list[Host]:
    cycles = cycle_digraph.get_cycles()
    if not cycles:
        return []
    
    cycle = cycles[0]
    hosts: list[Host] = []
    for i in range(len(cycle)):
        current_node = cycle[i]
        next_node = cycle[(i + 1) % len(cycle)]
        
        for connection in cycle_digraph.list_connections():
            if (connection.local_node.node_id == current_node.node_id and 
                connection.send_back_node.node_id == next_node.node_id):
                host = Host(
                    multiaddr=connection.send_back_multiaddr
                )
                hosts.append(host)
                break
    
    return hosts
