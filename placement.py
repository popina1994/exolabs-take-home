from .data_types.topology import Topology
from .data_types.topology_snapshot import TopologySnapshot
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


We want find to find -resource scheduling that has minimal the following sequence:
find l paths
(N_{1,1}, E_{1,1}, N_{1, 2}, ..., E_{1, l_1}, N_{1,l_1+1}),
(N_{1,l_1+1}, E_{2, 1}, N_{2, 2}, ..., E_{1, l_2},  N_{2,l_2 + 1}),
...
(N_{l-1,l_{l-1} + 1}, E_{l, 1}, N_{l, 2}, ..., E_{l, l_l},  N_{l,l_l + 1}), ...
where
sum_i ^l {L(E_{i, 1}) + B(C_{i, 1}) * d + L(E_{i, 2}) + B(C_{i, 2}) * d + ...
+ E_{i, L_1}} + B(C_{i, L_1}) * d + C(N_{1,i}) + C(N_{1, l+1}) is minimized,
where L returns the latency of the edge E_{i, 1}, C returns the time to compute all the assigned operations on the input node, B returns the bandwidth of the input link,
d returns the amount of data that needs to be transferred between two nodes.

Optimal solution (exponential)
We need to try each of N! traversals and in each case we compute the function and see which function
is minimal.

Simple heuristic can be to pick the node with the highest number of FLOPs and then pick the next
node with the highest number of FLOPs etc and select for this to be the optimal choice.

"""
def get_instance_placements_snapshot(
    command: CreateInstanceCommand,
    topology: TopologySnapshot,
    current_instances: dict[InstanceId, Instance],
    instance_id: InstanceId | None = None,
) -> dict[InstanceId, Instance]:
    return dict()
    # if instance_id is None:
    #     instance_id = InstanceId()
    # available_models = [current_instances[instance].shard_assignments.model_id for instance in current_instances]
    # if command.model_meta.model_id in available_models:
    #     raise ValueError(f"Instance for {command.model_meta.model_id} already exists")

    # all_nodes = topology.list_nodes()
    # cycles = topology.get_cycles()
    # # we can also always just have a node on its own
    # # Why do we need to have cycles? In other words why do we need to have all connections?
    # singleton_cycles = [[node] for node in all_nodes]
    # candidate_cycles = cycles + singleton_cycles

    # cycles_with_sufficient_memory = filter_cycles_by_memory(candidate_cycles, command.model_meta.storage_size_kilobytes * 1024)
    # if not cycles_with_sufficient_memory:
    #     raise ValueError("No cycles found with sufficient memory")

    # smallest_cycles = get_smallest_cycles(cycles_with_sufficient_memory)
    # selected_cycle = max(smallest_cycles, key=lambda cycle: sum(node.node_profile.memory.ram_available for node in cycle))

    # # We need to determine the sharding of layers to nodes using proportional ratio?
    # shard_assignments = get_shard_assignments(command.model_meta, selected_cycle)

    # cycle_digraph: Topology = topology.get_subgraph_from_nodes(selected_cycle)
    # hosts: list[Host] = get_hosts_from_subgraph(cycle_digraph)

    # target_instances = deepcopy(current_instances)
    # target_instances[instance_id] = Instance(
    #     instance_id=instance_id,
    #     instance_active=True,
    #     shard_assignments=shard_assignments,
    #     hosts=hosts
    # )
    # return target_instances




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

