from typing import List, Optional, Protocol, TypeVar

import numpy as np
import numpy.typing as npt


class INode(Protocol):
    op: Optional[str]
    value: Optional[float]
    left: Optional["INode"]
    right: Optional["INode"]

    def evaluate(
        self, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...
    def copy(self) -> "INode": ...
    def to_string(self) -> str: ...
    def validate(self) -> bool: ...


NodeType = TypeVar("NodeType", bound=INode)


def get_nodes(node: NodeType) -> List[NodeType]:
    nodes = []
    if node:
        nodes.append(node)
        if node.left:
            nodes.extend(get_nodes(node.left))
        if node.right:
            nodes.extend(get_nodes(node.right))
    return nodes
