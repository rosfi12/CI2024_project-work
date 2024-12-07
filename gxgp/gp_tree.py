#   *        Giovanni Squillero's GP Toolbox
#  / \       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2   +      A no-nonsense GP in pure Python
#    / \
#  10   11   Distributed under MIT License

from typing import Collection

import random  # noqa: F401
from .node import Node  # noqa: F401
from .utils import arity  # noqa: F401

__all__ = ["TreeGP"]


class TreeGP:
    def __init__(
        self,
        operators: Collection,
        variables: int | Collection,
        constants: int | Collection,
        *,
        seed=42,
    ):
        raise NotImplementedError
