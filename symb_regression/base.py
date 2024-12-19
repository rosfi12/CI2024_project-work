from typing import Any, Generic, Optional, Protocol, TypeVar

import numpy as np
import numpy.typing as npt

NodeType = TypeVar("NodeType", bound="INode[Any]")


class INode(Protocol, Generic[NodeType]):
    op: Optional[str]
    value: Optional[float]
    left: Optional[NodeType]
    right: Optional[NodeType]

    def evaluate(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
    def copy(self) -> NodeType: ...
    def to_string(self) -> str: ...
    def validate(self) -> bool: ...
