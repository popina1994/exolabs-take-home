from typing import Self
import uuid

class Id(str):
    def __new__(cls, val: str | None = None) -> Self:
        return super().__new__(cls, val if val is not None else str(uuid.uuid4()))

class InstanceId(Id):
    pass

class EventId(Id):
    pass

class NodeId(Id):
    pass

class ModelId(Id):
    pass

class RunnerId(Id):
    pass
