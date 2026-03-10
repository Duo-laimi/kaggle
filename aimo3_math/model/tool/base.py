from abc import ABC, abstractmethod

class ToolBase(ABC):

    name = "default"

    @abstractmethod
    def __call__(self, *args, **kwargs): ...

    def get_tool_schema(self): ...

