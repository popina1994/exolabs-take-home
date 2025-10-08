from .data_types.topology import Topology, TopologyNode, Connection, NodeId
from .data_types.topology_snapshot import TopologySnapshot
from .data_types.events import Event, Instance, Host, InstanceCreated, InstanceDeleted, InstanceActivated, InstanceDeactivated, InstanceReplacedAtomically
from .data_types.shards import CreateInstanceCommand
from .data_types.common import InstanceId, EventId

from copy import deepcopy
from itertools import permutations
from enum import Enum
import logging
import sys

from .placement_utils import (
    filter_cycles_by_memory,
    get_hosts_from_subgraph,
    get_shard_assignments,
    get_smallest_cycles,
)

class PlacementAlgorithm(Enum):
    Cycle = 1
    MinimalLatency = 2

"""
Find the cycle with the smallest amount of memory that can store the whole command. model.
Then, for each of the nodes belonging to cycle proportionally assign number of layers in the model as
much as this node contributes to the total amount of memory.
These layers combined on node node represent shard.
Each shard is run using one runner.
In other words, for each model and shard, there is one runner on a node.

If there are multiple cycles to choose, first one with the smallest number of nodes,
than we choose the one with the largest amount of
available memory.
Runner generally is a thread that runs a task?
Instances are just a wrapper of shard asignements (mapping between shards and nodes)
 and hosts
"""
def get_instance_placements_cycle(
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
    # Why do we need to have cycles? In other words why do we need to have all connections?
    singleton_cycles = [[node] for node in all_nodes]
    candidate_cycles = cycles + singleton_cycles

    cycles_with_sufficient_memory = filter_cycles_by_memory(candidate_cycles, command.model_meta.storage_size_kilobytes * 1024)
    if not cycles_with_sufficient_memory:
        raise ValueError("No cycles found with sufficient memory")

    smallest_cycles = get_smallest_cycles(cycles_with_sufficient_memory)
    selected_cycle = max(smallest_cycles, key=lambda cycle: sum(node.node_profile.memory.ram_available for node in cycle))

    # We need to determine the sharding of layers to nodes using proportional ratio?
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



def get_latency_on_path(prev_node: TopologyNode, node: TopologyNode,
                        shortest_paths: dict[NodeId, dict[NodeId, list[TopologyNode]]],
                        topology: Topology, data_to_move_size: int) -> int | None:
    prev_node_path: TopologyNode | None = None
    path_latency: int = 0

    for node_path in shortest_paths[prev_node.node_id][node.node_id]:
        if prev_node_path is None:
            prev_node_path = node_path
            continue
        # extract connection
        connection: Connection | None = topology.get_connection(
            prev_node_path.node_id, node_path.node_id)
        if connection is None:
            print("CONNECTION", prev_node_path.node_id, node_path.node_id)
            return None
        # bandwidth
        path_latency += connection.connection_profile.latency
        # We assume the data is moved from a node to a node as a whole,
        #  and is not forwarded further until the whole data is moved.
        # If the data can be moved from a node to a node as a stream, then
        # the total_latency is increased only once
        path_latency += data_to_move_size / connection.connection_profile.throughput
        prev_node_path = node_path

    return path_latency

"""
GOAL: minimize the total latency from the start to the finish of the neural network inference over
multiple nodes N_1, N_2, ..., N_n each with the memory M_1, M_2, ..., M_n, that are connected with network links of bandwidth B_{i, j} each with the latency L_{i, j}.

Formal definition of the problem:
We define a traversal of a graph as a list of paths where one node can occur
at most once at the start of the path and at most once at the end of the path.
A path is an alternative sequence of nodes and edges starting with a node and ending with a node,
where each edge is adjacent in the graph to both nodes in the list.
A multi-resource scheduling of sequence of tasks T_1, T_2, ..., T_k, is a traversal of the graph where each i-th N_j node in the sequence of paths must satisfy M_{j} >= T_i.
In other words, there is sufficient memory in the N_j node to evaluate task T_i.


We want find to find -resource scheduling that for any of the following sequences:
(N_{1,1}, E_{1,1}, N_{1, 2}, ..., E_{1, l_1}, N_{1,l_1+1}),
(N_{1,l_1+1}, E_{2, 1}, N_{2, 2}, ..., E_{1, l_2},  N_{2,l_2 + 1}),
...
(N_{l-1,l_{l-1} + 1}, E_{l, 1}, N_{l, 2}, ..., E_{l, l_l},  N_{l,l_l + 1}), ...
find the following value minimized
sum_i ^l ({L(E_{i, 1}) + B(C_{i, 1}) * d + L(E_{i, 2}) + B(C_{i, 2}) * d + ...
+ E_{i, L_1}} + B(C_{i, L_1}) * d + C(N_{i,l_{i} + 1}))
+ C(N_{l, l_l +1})

where L returns the latency of the edge E_{i, 1}, C returns the time to compute all the assigned operations on the input node, B returns the bandwidth of the input communication link,
d returns the amount of data that needs to be transferred between two nodes.
d is always the same for one model.

Optimal solution (exponential)
We need to try each of N! traversals and in each case we compute the function and see which function
is minimal.

Simple heuristic can be to pick the node with the highest number of FLOPs and then pick the next
node with the highest number of FLOPs etc and select for this to be the optimal choice.

"""
def get_instance_placements_snapshot(
    command: CreateInstanceCommand,
    topology_snapshot: TopologySnapshot,
    current_instances: dict[InstanceId, Instance],
    instance_id: InstanceId | None = None,
) -> dict[InstanceId, Instance]:
    if instance_id is None:
        instance_id = InstanceId()
    available_models = [current_instances[instance].shard_assignments.model_id for instance in current_instances]
    if command.model_meta.model_id in available_models:
        raise ValueError(f"Instance for {command.model_meta.model_id} already exists")

    all_nodes = topology_snapshot.topology.list_nodes()
    shortest_paths = topology_snapshot.topology.get_shortest_paths()
    best_perm: list[TopologyNode] = []
    best_latency: int = sys.maxsize

    all_perms = [list(perm) for perm in permutations(all_nodes)]

    logging.basicConfig(level=logging.DEBUG)
    logging.debug(f"SHORTEST PATHS: {shortest_paths}")
    for all_nodes_perm in all_perms:
        prev_node = None
        total_latency = 0
        model_size_to_store: int = command.model_meta.storage_size_kilobytes * 1024
        # TODO: ceil division?
        data_to_move_size = model_size_to_store / command.model_meta.n_layers
        node_available_memory: dict[NodeId, int] = {node.node_id: node.node_profile.memory.ram_available for node in all_nodes_perm}
        there_is_path = True
        cur_perm = all_nodes_perm

        for idx, node in enumerate(all_nodes_perm):
            print("IDX", idx, node.node_id, command.model_meta.storage_size_kilobytes )
            if model_size_to_store == 0:
                cur_perm = all_nodes_perm[:idx]
                break

            memory_used = min(node_available_memory[node.node_id], model_size_to_store)
            logging.debug(f"MEMORY USED{memory_used} f{node_available_memory[node.node_id]} {model_size_to_store}")
            node_available_memory[node.node_id] -= memory_used
            model_size_to_store -= memory_used
            #TODO: Model different layers of computation
            # TODO: add latency when the bandwidth is occupied?
            #TODO: Add different compute models costs?
            # Here we assume models is memory bound in other words
            # compute is faster than memory reads, thus we are bound on comp_latency.
            # Otherwise, we would compute the latency as the maximum of these two ratios.
            comp_latency = int(memory_used / node.node_profile.system.mem_bandwidth_kbps * 1024)
            total_latency += comp_latency
            if prev_node is not None:
                path_latency = get_latency_on_path(prev_node=prev_node, node=node, shortest_paths=shortest_paths, topology=topology_snapshot.topology,
                                                   data_to_move_size=data_to_move_size)
                if path_latency is None:
                    there_is_path = False
                    break
                else:
                    total_latency += path_latency
            prev_node = node

        if model_size_to_store > 0:
            raise ValueError("No cycles found with sufficient memory")

        if there_is_path and (total_latency <= best_latency):
            best_perm = cur_perm
            best_latency = total_latency
    print(best_perm)
    shard_assignments = get_shard_assignments(command.model_meta, best_perm, True)

    cycle_digraph: Topology = topology_snapshot.topology.get_subgraph_from_nodes(best_perm)
    hosts: list[Host] = get_hosts_from_subgraph(cycle_digraph)

    target_instances = deepcopy(current_instances)
    target_instances[instance_id] = Instance(
        instance_id=instance_id,
        instance_active=True,
        shard_assignments=shard_assignments,
        hosts=hosts
    )
    return target_instances



"""
Find the cycle with the smallest amount of memory that can store the whole command. model.
Then, for each of the nodes belonging to cycle proportionally assign number of layers in the model as
much as this node contributes to the total amount of memory.
These layers combined on node node represent shard.
Each shard is run using one runner.
In other words, for each model and shard, there is one runner on a node.

If there are multiple cycles to choose, first one with the smallest number of nodes,
than we choose the one with the largest amount of
available memory.
Runner generally is a thread that runs a task?
Instances are just a wrapper of shard asignements (mapping between shards and nodes)
 and hosts
"""
def get_instance_placements(
    command: CreateInstanceCommand,
    topology_snapshot: TopologySnapshot,
    current_instances: dict[InstanceId, Instance],
    instance_id: InstanceId | None = None,
    placement_algorithm: PlacementAlgorithm = PlacementAlgorithm.Cycle
) -> dict[InstanceId, Instance]:
    if placement_algorithm == PlacementAlgorithm.Cycle:
        return get_instance_placements_cycle(command=command, topology=topology_snapshot.topology, current_instances=current_instances, instance_id=instance_id)
    else:
        return get_instance_placements_snapshot(command=command, topology_snapshot=topology_snapshot, current_instances=current_instances, instance_id=instance_id)

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

    # Here we check if the instance is activated/deactivated events.
    # We cannot generate NodePerformanceMeasured and WorkerStatusUpdated events,
    # since instances do not contain any performance information or worker status info.
    # So the other WorkerStatusUpdated should be generated by some callback from the worker.
    # NodePerformanceMeasured should be generated by some worker that tracks the state of the system.

    for instance_id in current_instances:
        if instance_id in target_instances:
            if current_instances[instance_id].instance_active and not target_instances[instance_id].instance_active:
                events.append(
                    InstanceDeactivated(
                        event_id=EventId(),
                        instance_id=instance_id,)
                )

            if not current_instances[instance_id].instance_active and target_instances[instance_id].instance_active:
                events.append(
                    InstanceActivated(
                        event_id=EventId(),
                        instance_id=instance_id,)
                )
            if current_instances[instance_id] != target_instances[instance_id]:
                events.append(
                    InstanceReplacedAtomically(
                        event_id=EventId(),
                        instance_id=instance_id,)
                )


    return events

