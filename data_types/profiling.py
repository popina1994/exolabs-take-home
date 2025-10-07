from dataclasses import dataclass

@dataclass
class MemoryPerformanceProfile:
    ram_total: int
    ram_available: int
    swap_total: int
    swap_available: int


@dataclass
class SystemPerformanceProfile:
    flops_fp16: float

    mem_bandwidth_kbps: int = 0
    gpu_usage: float = 0.0
    temp: float = 0.0
    sys_power: float = 0.0
    pcpu_usage: float = 0.0
    ecpu_usage: float = 0.0
    ane_power: float = 0.0


@dataclass
class NetworkInterfaceInfo:
    name: str
    ip_address: str
    type: str


@dataclass
class NodePerformanceProfile:
    model_id: str
    chip_id: str
    friendly_name: str
    memory: MemoryPerformanceProfile
    network_interfaces: list[NetworkInterfaceInfo]
    system: SystemPerformanceProfile


@dataclass
class ConnectionProfile:
    throughput: float
    latency: float
    jitter: float
