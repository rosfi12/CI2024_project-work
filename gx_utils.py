# Copyright © 2022 Giovanni Squillero <squillero@polito.it>
# https://github.com/squillero/computational-intelligence
# Free for personal or classroom use; see 'LICENSE.md' for details.

import heapq
from collections import Counter
from typing import Any, Generator


class PriorityQueue:
    """A basic Priority Queue with simple performance optimizations"""

    _data_heap: list[Any]
    _data_set: set[Any]

    def __init__(self) -> None:
        self._data_heap = list()
        self._data_set = set()

    def __bool__(self) -> bool:
        return bool(self._data_set)

    def __contains__(self, item) -> bool:
        return item in self._data_set

    def push(self, item, p: int | None = None) -> None:
        assert item not in self, "Duplicated element"
        if p is None:
            p = len(self._data_set)
        self._data_set.add(item)
        heapq.heappush(self._data_heap, (p, item))

    def pop(self) -> Any:
        p, item = heapq.heappop(self._data_heap)
        self._data_set.remove(item)
        return item


class Multiset:
    """Multiset"""

    def __init__(self, init: Any = None) -> None:
        self._data: Counter[Any] = Counter()
        if init:
            for item in init:
                self.add(item)

    def __contains__(self, item: Any) -> bool:
        return item in self._data and self._data[item] > 0

    def __getitem__(self, item) -> int:
        return self.count(item)

    def __iter__(self) -> Generator[Any, None, None]:
        return (i for i in sorted(self._data.keys()) for _ in range(self._data[i]))

    def __len__(self) -> int:
        return sum(self._data.values())

    def __copy__(self) -> "Multiset":
        t = Multiset()
        t._data = self._data.copy()
        return t

    def __str__(self) -> str:
        return f"M{{{', '.join(repr(i) for i in self)}}}"

    def __repr__(self) -> str:
        return str(self)

    def __or__(self, other: "Multiset") -> "Multiset":
        tmp = Multiset()
        for i in set(self._data.keys()) | set(other._data.keys()):
            tmp.add(i, cnt=max(self[i], other[i]))
        return tmp

    def __and__(self, other: "Multiset") -> "Multiset":
        return self.intersection(other)

    def __add__(self, other: "Multiset") -> "Multiset":
        return self.union(other)

    def __sub__(self, other: "Multiset") -> "Multiset":
        tmp = Multiset(self)
        for i, n in other._data.items():
            tmp.remove(i, cnt=n)
        return tmp

    def __eq__(self, other) -> bool:
        if not isinstance(other, Multiset):
            raise
        return list(self) == list(other)

    def __le__(self, other: "Multiset") -> bool:
        for i, n in self._data.items():
            if other.count(i) < n:
                return False
        return True

    def __lt__(self, other: "Multiset") -> bool:
        return self <= other and not self == other

    def __ge__(self, other: "Multiset") -> bool:
        return other <= self

    def __gt__(self, other: "Multiset") -> bool:
        return other < self

    def add(self, item, *, cnt: int = 1) -> None:
        assert cnt >= 0, "Can't add a negative number of elements"
        if cnt > 0:
            self._data[item] += cnt

    def remove(self, item, *, cnt=1) -> None:
        assert item in self, "Item not in collection"
        self._data[item] -= cnt
        if self._data[item] <= 0:
            del self._data[item]

    def count(self, item):
        return self._data[item] if item in self._data else 0

    def union(self, other: "Multiset") -> "Multiset":
        t = Multiset(self)
        for i in other._data.keys():
            t.add(i, cnt=other[i])
        return t

    def intersection(self, other: "Multiset") -> "Multiset":
        t = Multiset()
        for i in self._data.keys():
            t.add(i, cnt=min(self[i], other[i]))
        return t
