from typing import Callable

import pytest

from ..placement import get_instance_placements, get_transition_events, PlacementAlgorithm
from ..data_types.topology import Topology
from ..data_types.topology_snapshot import TopologySnapshot
from ..data_types.common import NodeId, ModelId
from ..data_types.events import (
    InstanceCreated,
    InstanceDeleted,
    InstanceActivated,
    InstanceDeactivated,
    InstanceReplacedAtomically
)
from ..data_types.shards import CreateInstanceCommand
from ..data_types.shards import ModelMetadata, ShardMetadata
from ..data_types.topology import Connection, TopologyNode
from ..data_types.common import InstanceId
from ..data_types.events import Instance
from ..data_types.shards import ShardAssignments
import logging
import itertools


@pytest.fixture
def topology() -> Topology:
    print("Topology")
    return Topology()


@pytest.fixture
def instance() -> Instance:
    return Instance(
        instance_id=InstanceId(),
        instance_active=True,
        shard_assignments=ShardAssignments(
            model_id=ModelId("test-model"), runner_to_shard={}, node_to_runner={}
        ),
        hosts=[],
    )

@pytest.fixture
def create_instances():
    def create_instances_in(n: int):
        return [Instance(
            instance_id=InstanceId(),
            instance_active=True,
            shard_assignments=ShardAssignments(
                model_id=ModelId("test-model"), runner_to_shard={}, node_to_runner={}
            ),
            hosts=[],
        ) for _ in range(n)]
    return create_instances_in


@pytest.fixture
def model_meta() -> ModelMetadata:
    print("Model meta")
    return ModelMetadata(
        model_id=ModelId("test-model"),
        storage_size_kilobytes=1000,
        pretty_name="Test Model",
        n_layers=10,
    )


def create_instance_command(model_meta: ModelMetadata) -> CreateInstanceCommand:
    return CreateInstanceCommand(
        model_meta=model_meta,
    )


# NOTE: The current placer is intentionally simple for the take home - update the tests as you update placement!
@pytest.mark.parametrize(
    "placement_algorithm,mem_bandwidth_kbps,total_mem_kb,models",
    [
        (
            PlacementAlgorithm.Cycle,
            (900_000_000, 800_000_000, 100_000_000, 800_000_000),
            (512_000_000, 256_000_000, 256_000_000, 126_000_000),
            [
                (150_000_000, 150, [150, 0, 0, 0]),
                (150_000_000, 150, [150, 0, 0, 0]),
                (150_000_000, 150, [0, 150, 0, 0]),
            ],
        ),
        (
            PlacementAlgorithm.MinimalLatency,
            (100_000_000, 800_000_000, 700_000_000, 900_000_000),
            (512_000_000, 256_000_000, 256_000_000, 126_000_000),
            [
                (150_000_000, 150, [0, 150, 0, 0]),
                (150_000_000, 150, [0, 0, 150, 0]),
                (150_000_000, 150, [150, 0, 0, 0]),
            ],
        ),
    ],
)
def test_get_instance_placements_with_multiple_models(
    placement_algorithm: PlacementAlgorithm,
    mem_bandwidth_kbps: list[int],
    total_mem_kb: list[int],
    models: list[tuple[int, int, list[int]]],
    topology: Topology,
    create_nodes_comprehensive: Callable[
        [list[NodeId], list[int], list[int], list[int]], list[TopologyNode]
    ],
    create_connection: Callable[[TopologyNode, TopologyNode], Connection],
):
    # arrange

    node_ids = [NodeId(str(i + 1)) for i in range(len(total_mem_kb))]
    logging.info(node_ids)
    nodes = create_nodes_comprehensive(
        node_ids, total_mem_kb, total_mem_kb, mem_bandwidth_kbps
    )
    print(nodes)
    node_dict = {node.node_id: node for node in nodes}

    for node in nodes:
        topology.add_node(node)

    for node_1 in nodes:
        for node_2 in nodes:
            if node_1 != node_2:
                topology.add_connection(create_connection(node_1, node_2))

    instances = {}
    for i, (model_mem_kb, n_layers, expected_layers) in enumerate(models):
        model_meta = ModelMetadata(
            model_id=ModelId(f"test-model-{i}"),
            storage_size_kilobytes=model_mem_kb,
            pretty_name=f"Test Model {i}",
            n_layers=n_layers,
        )

        create_instance_command = CreateInstanceCommand(
            model_meta=model_meta,
        )

        # act
        topology_snapshot = TopologySnapshot(topology)
        instance_id = InstanceId()
        instances = get_instance_placements(
            create_instance_command, topology_snapshot, instances, instance_id, placement_algorithm
        )

        instance = instances[instance_id]
        shards: list[ShardMetadata | None] = []
        for node_id in node_ids:
            if node_id in instance.shard_assignments.node_to_runner:
                shard = instance.shard_assignments.runner_to_shard[
                    instance.shard_assignments.node_to_runner[node_id]
                ]
                shard_mem_kb = round(
                    (shard.end_layer - shard.start_layer) / n_layers * model_mem_kb
                )
                print("NODE", node_id, node_dict[node_id].node_profile.memory.ram_available)
                node_dict[node_id].node_profile.memory.ram_available -= (
                    shard_mem_kb * 1024
                )
                print("NODE", node_id, node_dict[node_id].node_profile.memory.ram_available)
                shards.append(shard)
            else:
                shards.append(None)

        layer_split = [
            0 if shard == None else shard.end_layer - shard.start_layer
            for shard in shards
        ]

        # assert
        assert len(instances) == i + 1
        assert instance.shard_assignments.model_id == model_meta.model_id

        assert layer_split == expected_layers

        shards_sorted = sorted(
            [shard for shard in shards if shard], key=lambda s: s.start_layer
        )
        assert shards_sorted[0].start_layer == 0
        assert shards_sorted[-1].end_layer == n_layers


# NOTE: The current placer is intentionally simple for the take home - update the tests as you update placement!
@pytest.mark.parametrize(
    "placement_algorithm, mem_bandwidth_kbps,total_mem_kb,available_mem_kb,model_mem_kb,total_layers,expected_layers",
    [
        (algo, *case)
        for algo, case in itertools.product(
            [PlacementAlgorithm.Cycle, PlacementAlgorithm.MinimalLatency],
            [
                (
                    (900_000_000, 800_000_000, 800_000_000, 800_000_000),
                    (512_000_000, 256_000_000, 256_000_000, 126_000_000),
                    (50_000_000, 100_000_000, 256_000_000, 126_000_000),
                    150_000_000,
                    150,
                    [0, 0, 150, 0],
                ),
                (
                    (900_000_000, 800_000_000, 1_000, 800_000_000),
                    (512_000_000, 256_000_000, 256_000_000, 126_000_000),
                    (50_000_000, 100_000_000, 256_000_000, 126_000_000),
                    150_000_000,
                    150,
                    [0, 0, 150, 0],
                ),
                (
                    (900_000_000, 800_000_000, 1_000, 800_000_000),
                    (512_000_000, 256_000_000, 256_000_000, 126_000_000),
                    (50_000_000, 206_000_000, 256_000_000, 126_000_000),
                    150_000_000,
                    150,
                    [0, 0, 150, 0],
                ),
            ]
        )
    ],
)
def test_get_instance_placements_create_instance_comprehensive(
    placement_algorithm: PlacementAlgorithm,
    mem_bandwidth_kbps: list[int],
    total_mem_kb: list[int],
    available_mem_kb: list[int],
    model_mem_kb: int,
    total_layers: int,
    expected_layers: list[int],
    topology: Topology,
    model_meta: ModelMetadata,
    create_nodes_comprehensive: Callable[
        [list[NodeId], list[int], list[int], list[int]], list[TopologyNode]
    ],
    create_connection: Callable[[TopologyNode, TopologyNode], Connection],
):
    # arrange
    model_meta.n_layers = total_layers
    model_meta.storage_size_kilobytes = model_mem_kb

    create_instance_command = CreateInstanceCommand(
        model_meta=model_meta,
    )

    node_ids = [NodeId(str(i + 1)) for i in range(len(expected_layers))]
    nodes = create_nodes_comprehensive(
        node_ids, total_mem_kb, available_mem_kb, mem_bandwidth_kbps
    )

    for node in nodes:
        topology.add_node(node)

    for node_1 in nodes:
        for node_2 in nodes:
            if node_1 != node_2:
                topology.add_connection(create_connection(node_1, node_2))

    # act
    topology_snapshot = TopologySnapshot(topology)
    placements = get_instance_placements(create_instance_command, topology_snapshot, {}, None, placement_algorithm)

    # assert
    assert len(placements) == 1
    instance_id = list(placements.keys())[0]
    instance = placements[instance_id]
    assert instance.shard_assignments.model_id == model_meta.model_id

    shards: list[ShardMetadata | None] = []
    for node_id in node_ids:
        if node_id in instance.shard_assignments.node_to_runner:
            shards.append(
                instance.shard_assignments.runner_to_shard[
                    instance.shard_assignments.node_to_runner[node_id]
                ]
            )
        else:
            shards.append(None)

    layer_split = [
        0 if shard == None else shard.end_layer - shard.start_layer for shard in shards
    ]

    assert layer_split == expected_layers

    shards_sorted = sorted(
        [shard for shard in shards if shard], key=lambda s: s.start_layer
    )
    assert shards_sorted[0].start_layer == 0
    assert shards_sorted[-1].end_layer == total_layers


@pytest.mark.parametrize(
    "placement_algorithm, available_memory,total_layers,expected_layers",
    [
        (alg, mem, total, expected)
        for alg, (mem, total, expected) in itertools.product(
            [PlacementAlgorithm.MinimalLatency],
            [
                ((500, 500, 1000), 12, (3, 3, 6)),
                ((500, 500, 500), 12, (4, 4, 4)),
                ((312, 518, 1024), 12, (2, 3, 7)),
            ],
        )
    ],
)
def test_get_instance_placements_create_instance1(
    placement_algorithm: PlacementAlgorithm,
    available_memory: tuple[int, int, int],
    total_layers: int,
    expected_layers: tuple[int, int, int],
    topology: Topology,
    model_meta: ModelMetadata,
    create_node: Callable[[int, NodeId | None], TopologyNode],
    create_connection: Callable[[TopologyNode, TopologyNode], Connection],
):
    # arrange
    model_meta.n_layers = total_layers
    model_meta.storage_size_kilobytes = sum(
        available_memory
    )  # make it exactly fit across all nodes

    create_instance_command = CreateInstanceCommand(
        model_meta=model_meta,
    )
    node_id_a = NodeId()
    node_id_b = NodeId()
    node_id_c = NodeId()
    node_a = create_node(available_memory[0] * 1024, node_id_a)
    node_b = create_node(available_memory[1] * 1024, node_id_b)
    node_c = create_node(available_memory[2] * 1024, node_id_c)
    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_node(node_c)
    topology.add_connection(create_connection(node_a, node_b))
    topology.add_connection(create_connection(node_b, node_c))
    topology.add_connection(create_connection(node_c, node_a))

    # act
    topology_snapshot = TopologySnapshot(topology=topology)
    placements = get_instance_placements(create_instance_command, topology_snapshot, {}, None, placement_algorithm=placement_algorithm)

    # assert
    assert len(placements) == 1
    instance_id = list(placements.keys())[0]
    instance = placements[instance_id]
    assert instance.shard_assignments.model_id == model_meta.model_id

    runner_id_a = instance.shard_assignments.node_to_runner[node_id_a]
    runner_id_b = instance.shard_assignments.node_to_runner[node_id_b]
    runner_id_c = instance.shard_assignments.node_to_runner[node_id_c]

    shard_a = instance.shard_assignments.runner_to_shard[runner_id_a]
    shard_b = instance.shard_assignments.runner_to_shard[runner_id_b]
    shard_c = instance.shard_assignments.runner_to_shard[runner_id_c]

    assert shard_a.end_layer - shard_a.start_layer == expected_layers[0]
    assert shard_b.end_layer - shard_b.start_layer == expected_layers[1]
    assert shard_c.end_layer - shard_c.start_layer == expected_layers[2]

    shards = [shard_a, shard_b, shard_c]
    shards_sorted = sorted(shards, key=lambda s: s.start_layer)
    assert shards_sorted[0].start_layer == 0
    assert shards_sorted[-1].end_layer == total_layers


@pytest.mark.parametrize(
    "placement_algorithm",
    [
        (
            PlacementAlgorithm.Cycle
        ),
        (
            PlacementAlgorithm.MinimalLatency
        ),
    ],
)
def test_get_instance_placements_snapshot_one_node_exact_fit(
    placement_algorithm: PlacementAlgorithm,
    create_node: Callable[[int, NodeId | None], TopologyNode],
) -> None:
    topology = Topology()
    node_id = NodeId()
    topology.add_node(create_node(1000 * 1024, node_id))
    create_instance_command = CreateInstanceCommand(
        model_meta=ModelMetadata(
            model_id=ModelId("test-model"),
            storage_size_kilobytes=1000,
            pretty_name="Test Model",
            n_layers=10,
        ),
    )
    topology_snapshot = TopologySnapshot(topology=topology)
    placements = get_instance_placements(create_instance_command, topology_snapshot, {}, None, placement_algorithm)

    assert len(placements) == 1
    instance_id = list(placements.keys())[0]
    instance = placements[instance_id]
    assert instance.shard_assignments.model_id == "test-model"
    assert len(instance.shard_assignments.node_to_runner) == 1
    assert len(instance.shard_assignments.runner_to_shard) == 1
    assert len(instance.shard_assignments.runner_to_shard) == 1


@pytest.mark.parametrize(
    "placement_algorithm",
    [
        (
            PlacementAlgorithm.Cycle
        ),
        (
            PlacementAlgorithm.MinimalLatency
        ),
    ],
)
def test_get_instance_placements_snapshot_one_node_fits_with_extra_memory(
    placement_algorithm: PlacementAlgorithm,
    create_node: Callable[[int, NodeId | None], TopologyNode],
) -> None:
    topology = Topology()
    node_id = NodeId()
    topology.add_node(create_node(1001 * 1024, node_id))
    create_instance_command = CreateInstanceCommand(
        model_meta=ModelMetadata(
            model_id=ModelId("test-model"),
            storage_size_kilobytes=1000,
            pretty_name="Test Model",
            n_layers=10,
        ),
    )
    topology_snapshot = TopologySnapshot(topology=topology)
    placements = get_instance_placements(create_instance_command, topology_snapshot, {}, None, placement_algorithm)

    assert len(placements) == 1
    instance_id = list(placements.keys())[0]
    instance = placements[instance_id]
    assert instance.shard_assignments.model_id == "test-model"
    assert len(instance.shard_assignments.node_to_runner) == 1
    assert len(instance.shard_assignments.runner_to_shard) == 1
    assert len(instance.shard_assignments.runner_to_shard) == 1


@pytest.mark.parametrize(
    "placement_algorithm",
    [
        (
            PlacementAlgorithm.Cycle
        ),
        (
            PlacementAlgorithm.MinimalLatency
        ),
    ],
)
def test_get_instance_placements_one_node_not_fit(
    placement_algorithm: PlacementAlgorithm,
    create_node: Callable[[int, NodeId | None], TopologyNode],
) -> None:
    topology = Topology()
    node_id = NodeId()
    topology.add_node(create_node(1000 * 1024, node_id))
    create_instance_command = CreateInstanceCommand(
        model_meta=ModelMetadata(
            model_id=ModelId("test-model"),
            storage_size_kilobytes=1001,
            pretty_name="Test Model",
            n_layers=10,
        ),
    )
    topology_snapshot = TopologySnapshot(topology=topology)
    with pytest.raises(ValueError, match="No cycles found with sufficient memory"):
        _ = get_instance_placements(create_instance_command, topology_snapshot, {}, None, placement_algorithm)

def test_get_transition_events_no_change(instance: Instance):
    # arrange
    instance_id = InstanceId()
    current_instances = {instance_id: instance}
    target_instances = {instance_id: instance}

    # act
    events = get_transition_events(current_instances, target_instances)

    # assert
    assert len(events) == 0


def test_get_transition_events_create_instance(instance: Instance):
    # arrange
    instance_id = InstanceId()
    current_instances: dict[InstanceId, Instance] = {}
    target_instances: dict[InstanceId, Instance] = {instance_id: instance}

    # act
    events = get_transition_events(current_instances, target_instances)

    # assert
    assert len(events) == 1
    assert isinstance(events[0], InstanceCreated)


def test_get_transition_events_delete_instance(instance: Instance):
    # arrange
    instance_id = InstanceId()
    current_instances: dict[InstanceId, Instance] = {instance_id: instance}
    target_instances: dict[InstanceId, Instance] = {}

    # act
    events = get_transition_events(current_instances, target_instances)

    # assert
    assert len(events) == 1
    assert isinstance(events[0], InstanceDeleted)
    assert events[0].instance_id == instance_id


def test_get_transition_events_activate_instance(
        create_instances: Callable[[int], list[Instance]]):
    # arrange
    instance_id = InstanceId()
    instances = create_instances(2)
    instances[0].instance_id = instance_id
    instances[1].instance_id = instance_id
    instances[0].instance_active = False
    instances[1].instance_active = True
    current_instances: dict[InstanceId, Instance] = {instance_id: instances[0]}
    target_instances: dict[InstanceId, Instance] = {instance_id: instances[1]}

    # act
    events = get_transition_events(current_instances, target_instances)

    # assert
    assert len(events) == 2
    assert isinstance(events[0], InstanceActivated)
    assert isinstance(events[1], InstanceReplacedAtomically)
    assert events[0].instance_id == instance_id
    assert events[1].instance_id == instance_id

def test_get_transition_events_deactivate_instance(
   create_instances: Callable[[int], list[Instance]]):
    # arrange
    instance_id = InstanceId()
    instances = create_instances(2)
    instances[0].instance_id = instance_id
    instances[1].instance_id = instance_id
    instances[0].instance_active = True
    instances[1].instance_active = False
    current_instances: dict[InstanceId, Instance] = {instance_id: instances[0]}
    target_instances: dict[InstanceId, Instance] = {instance_id: instances[1]}

    # act
    events = get_transition_events(current_instances, target_instances)

    # assert
    assert len(events) == 2
    assert isinstance(events[0], InstanceDeactivated)
    assert isinstance(events[1], InstanceReplacedAtomically)
    assert events[0].instance_id == instance_id
    assert events[1].instance_id == instance_id
