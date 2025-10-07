import pytest

from ..data_types.common import NodeId
from ..data_types.topology import Multiaddr
from ..data_types.profiling import (
    MemoryPerformanceProfile,
    NodePerformanceProfile,
    SystemPerformanceProfile,
    ConnectionProfile,
)
from ..data_types.topology import Connection, TopologyNode


@pytest.fixture
def create_nodes_comprehensive():
    def _create_nodes_comprehensive(
        node_ids: list[NodeId],
        mem_total_kb: list[int],
        mem_available_kb: list[int],
        mem_bandwidth_kbps: list[int],
    ) -> list[TopologyNode]:
        if (
            len(node_ids) != len(mem_total_kb)
            or len(node_ids) != len(mem_available_kb)
            or len(node_ids) != len(mem_bandwidth_kbps)
        ):
            raise ValueError("Misconfigured test: different length arrays")

        return [
            TopologyNode(
                node_id=node_ids[i],
                node_profile=NodePerformanceProfile(
                    model_id="test",
                    chip_id="test",
                    friendly_name="test",
                    memory=MemoryPerformanceProfile(
                        ram_total=mem_total_kb[i] * 1024,
                        ram_available=mem_available_kb[i] * 1024,
                        swap_total=1000,
                        swap_available=1000,
                    ),
                    network_interfaces=[],
                    system=SystemPerformanceProfile(
                        flops_fp16=1000, mem_bandwidth_kbps=mem_bandwidth_kbps[i]
                    ),
                ),
            )
            for i in range(len(node_ids))
        ]

    return _create_nodes_comprehensive


@pytest.fixture
def create_node():
    def _create_node(memory: int, node_id: NodeId | None = None) -> TopologyNode:
        if node_id is None:
            node_id = NodeId()
        return TopologyNode(
            node_id=node_id,
            node_profile=NodePerformanceProfile(
                model_id="test",
                chip_id="test",
                friendly_name="test",
                memory=MemoryPerformanceProfile(
                    ram_total=1000,
                    ram_available=memory,
                    swap_total=1000,
                    swap_available=1000,
                ),
                network_interfaces=[],
                system=SystemPerformanceProfile(flops_fp16=1000),
            ),
        )

    return _create_node


@pytest.fixture
def create_connection():
    port_counter = 1235

    def _create_connection(
        source_node: TopologyNode, sink_node: TopologyNode, send_back_port: int | None = None
    ) -> Connection:
        nonlocal port_counter
        if send_back_port is None:
            send_back_port = port_counter
            port_counter += 1
        return Connection(
            local_node=source_node,
            send_back_node=sink_node,
            send_back_multiaddr=Multiaddr(
                f"/ip4/127.0.0.1/tcp/{send_back_port}"
            ),
            connection_profile=ConnectionProfile(
                throughput=1000, latency=1000, jitter=1000
            ),
        )

    return _create_connection
