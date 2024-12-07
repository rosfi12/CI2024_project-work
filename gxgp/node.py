#   *        Giovanni Squillero's GP Toolbox
#  / \       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2   +      A no-nonsense GP in pure Python
#    / \
#  10   11   Distributed under MIT License

import numbers
import warnings
from typing import Callable, Iterable

from .draw import draw
from .utils import arity

__all__: list[str] = ["Node"]


class Node:
    _func: Callable
    _successors: tuple["Node", ...]
    _arity: int
    _str: str

    def __init__(
        self,
        node=None,
        successors: Iterable["Node"] | None = None,
        *,
        name: str | None = None,
    ):
        if callable(node) and successors is not None:

            def _f(*_args, **_kwargs) -> object:
                return node(*_args)

            self._func = _f
            self._successors = tuple(successors)
            self._arity = arity(node)
            assert self._arity is None or len(tuple(successors)) == self._arity, (
                "Panic: Incorrect number of children."
                + f" Expected {len(tuple(successors))} found {arity(node)} with successors {tuple(successors)}"
            )
            self._leaf = False
            assert all(
                isinstance(s, Node) for s in successors
            ), "Panic: Successors must be `Node`"
            self._successors = tuple(successors)
            if name is not None:
                self._str = name
            elif node.__name__ == "<lambda>":
                self._str = "λ"
            else:
                self._str = node.__name__
        elif isinstance(node, numbers.Number):
            self._func = eval(f"lambda **_kw: {node}")
            self._successors = tuple()
            self._arity = 0
            self._str = f"{node:g}"
        elif isinstance(node, str):
            self._func = eval(f"lambda *, {node}, **_kw: {node}")
            self._successors = tuple()
            self._arity = 0
            self._str = str(node)
        else:
            assert False

    def __call__(self, **kwargs):
        return self._func(*[c(**kwargs) for c in self._successors], **kwargs)

    def __str__(self):
        return self.long_name

    def __len__(self):
        return 1 + sum(len(c) for c in self._successors)

    @property
    def value(self) -> "Node":
        return self()

    @property
    def arity(self) -> int:
        return self._arity

    @property
    def successors(self) -> list["Node"]:
        return list(self._successors)

    @successors.setter
    def successors(self, new_successors) -> None:
        assert len(new_successors) == len(self._successors)
        self._successors = tuple(new_successors)

    @property
    def is_leaf(self) -> bool:
        return not self._successors

    @property
    def short_name(self) -> str:
        return self._str

    @property
    def long_name(self) -> str:
        if self.is_leaf:
            return self.short_name
        else:
            return (
                f"{self.short_name}("
                + ", ".join(c.long_name for c in self._successors)
                + ")"
            )

    @property
    def subtree(self) -> set["Node"]:
        result: set["Node"] = set()
        _get_subtree(result, self)
        return result

    def draw(self) -> None:
        try:
            return draw(self)
        except Exception as msg:
            warnings.warn(f"Drawing not available ({msg})", UserWarning, 2)
            return None


def _get_subtree(bunch: set["Node"], node: "Node"):
    bunch.add(node)
    for c in node._successors:
        _get_subtree(bunch, c)
