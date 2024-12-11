#   *        Giovanni Squillero's GP Toolbox
#  / \       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2   +      A no-nonsense GP in pure Python
#    / \
#  10   11   Distributed under MIT License

from copy import deepcopy

from .node import Node
from .random import gxgp_random
from icecream import ic


def xover_swap_subtree(tree1: Node, tree2: Node) -> Node:
    if tree1.is_leaf or tree2.is_leaf:
        print("[ERROR]: Cannot swap leaf nodes")
        return deepcopy(tree1)

    offspring: Node = deepcopy(tree1)
    ic(offspring)
    node: Node | None = None
    successors: list[Node] | None = None
    while not successors:
        node = gxgp_random.choice(list(offspring.subtree))
        successors = node.successors
    # assert node is not None, "[ERROR]: tree1 has not subtree! FIXME!"
    i: int = gxgp_random.randrange(len(successors))
    ic(successors, i)
    successors[i] = deepcopy(gxgp_random.choice(list(tree2.subtree)))
    ic(successors, i)
    # node.successors = successors
    return offspring
