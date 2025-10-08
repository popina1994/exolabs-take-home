from collections.abc import Generator
from dataclasses import dataclass
from typing import override

import rustworkx as rx

from .common import NodeId
from .profiling import ConnectionProfile, NodePerformanceProfile

@dataclass
class TopologyNode:
    node_id: NodeId
    node_profile: NodePerformanceProfile

class Multiaddr(str):
    pass

@dataclass
class Connection:
    local_node: TopologyNode
    send_back_node: TopologyNode
    # The multiaddr of the send_back_node_id
    send_back_multiaddr: Multiaddr
    connection_profile: ConnectionProfile | None = None

    @override
    def __hash__(self) -> int:
        return hash((self.local_node.node_id, self.send_back_node.node_id, self.send_back_multiaddr))


class Topology:
    def __init__(self) -> None:
        self._graph: rx.PyDiGraph[TopologyNode, Connection] = rx.PyDiGraph()
        self._node_id_to_rx_id_map: dict[NodeId, int] = dict()
        self._rx_id_to_node_id_map: dict[int, NodeId] = dict()
        self._edge_id_to_rx_id_map: dict[Connection, int] = dict()
        self._edge_to_edge_id: dict[tuple[NodeId, NodeId], Connection] = dict()


    def add_node(self, node: TopologyNode) -> None:
        if node.node_id in self._node_id_to_rx_id_map:
            return
        rx_id = self._graph.add_node(node)
        self._node_id_to_rx_id_map[node.node_id] = rx_id
        self._rx_id_to_node_id_map[rx_id] = node.node_id

    def contains_node(self, node_id: NodeId) -> bool:
        return node_id in self._node_id_to_rx_id_map

    def contains_connection(self, connection: Connection) -> bool:
        return connection in self._edge_id_to_rx_id_map

    def add_connection(
        self,
        connection: Connection,
    ) -> None:
        if connection.local_node.node_id not in self._node_id_to_rx_id_map:
            self.add_node(connection.local_node)
        if connection.send_back_node.node_id not in self._node_id_to_rx_id_map:
            self.add_node(connection.send_back_node)

        src_id = self._node_id_to_rx_id_map[connection.local_node.node_id]
        sink_id = self._node_id_to_rx_id_map[connection.send_back_node.node_id]

        rx_id = self._graph.add_edge(src_id, sink_id, connection)
        self._edge_id_to_rx_id_map[connection] = rx_id
        self._edge_to_edge_id[(connection.local_node.node_id, connection.send_back_node.node_id)] = connection

    def list_nodes(self) -> Generator[TopologyNode]:
        return (self._graph[i] for i in self._graph.node_indices())

    def list_connections(self) -> Generator[Connection]:
        return (connection for _, _, connection in self._graph.weighted_edge_list())

    def get_node_profile(self, node_id: NodeId) -> NodePerformanceProfile | None:
        try:
            rx_idx = self._node_id_to_rx_id_map[node_id]
            return self._graph.get_node_data(rx_idx).node_profile
        except KeyError:
            return None

    def update_node_profile(
        self, node_id: NodeId, node_profile: NodePerformanceProfile
    ) -> None:
        rx_idx = self._node_id_to_rx_id_map[node_id]
        self._graph[rx_idx].node_profile = node_profile

    def update_connection_profile(self, connection: Connection) -> None:
        rx_idx = self._edge_id_to_rx_id_map[connection]
        self._graph.update_edge_by_index(rx_idx, connection)

    def get_connection(self, nodeId1: NodeId, nodeId2: NodeId):
        if (nodeId1, nodeId2) in self._edge_to_edge_id:
            return self._edge_to_edge_id[(nodeId1, nodeId2)]
        return None

    def get_connection_profile(
        self, connection: Connection
    ) -> ConnectionProfile | None:
        try:
            rx_idx = self._edge_id_to_rx_id_map[connection]
            return self._graph.get_edge_data_by_index(rx_idx).connection_profile
        except KeyError:
            return None

    def remove_node(self, node_id: NodeId) -> None:
        rx_idx = self._node_id_to_rx_id_map[node_id]
        self._graph.remove_node(rx_idx)

        del self._node_id_to_rx_id_map[node_id]
        del self._rx_id_to_node_id_map[rx_idx]

    def remove_connection(self, connection: Connection) -> None:
        rx_idx = self._edge_id_to_rx_id_map[connection]
        self._graph.remove_edge_from_index(rx_idx)
        del self._edge_id_to_rx_id_map[connection]
        if rx_idx in self._rx_id_to_node_id_map:
            del self._rx_id_to_node_id_map[rx_idx]

    def get_cycles(self) -> list[list[TopologyNode]]:
        cycle_idxs = rx.simple_cycles(self._graph)
        cycles: list[list[TopologyNode]] = []
        for cycle_idx in cycle_idxs:
            cycle = [self._graph[idx] for idx in cycle_idx]
            cycles.append(cycle)

        return cycles

    def get_subgraph_from_nodes(self, nodes: list[TopologyNode]) -> "Topology":
        node_idxs = [node.node_id for node in nodes]
        rx_idxs = [self._node_id_to_rx_id_map[idx] for idx in node_idxs]
        topology = Topology()
        for rx_idx in rx_idxs:
            topology.add_node(self._graph[rx_idx])
        for connection in self.list_connections():
            if (
                connection.local_node.node_id in node_idxs
                and connection.send_back_node.node_id in node_idxs
            ):
                topology.add_connection(connection)
        return topology

    def get_shortest_paths(self) -> dict[NodeId, dict[NodeId, list[TopologyNode]]]:
        rx_shortest_paths: rx.AllPairsPathLengthMapping = \
            rx.all_pairs_dijkstra_shortest_paths(self._graph, edge_cost_fn=lambda e: e.connection_profile.latency)

        shortest_paths_dict: dict[NodeId, dict[NodeId, list[TopologyNode]]] = {}
        for rx_source_node, rx_target_node_paths in rx_shortest_paths.items():
            source_node = self._rx_id_to_node_id_map[rx_source_node]
            target_node_paths = {}
            for rx_target_node, rx_path in rx_target_node_paths.items():
                path = [self._graph.get_node_data(rx_node) for rx_node in rx_path]
                target_node_paths[self._rx_id_to_node_id_map[rx_target_node]] = path
            shortest_paths_dict[source_node] = target_node_paths


        return shortest_paths_dict
