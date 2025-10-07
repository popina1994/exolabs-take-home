from .topology import Topology
from dataclasses import dataclass
import copy

@dataclass
class TopologySnapshot:
    topology: Topology

    def __init__(self, topology: Topology):
        self.topology = copy.deepcopy(topology)