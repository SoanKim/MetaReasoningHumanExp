#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim) at 18:57 on 31/1/25
# Title: (Enter feature name here)
# Explanation: (Enter explanation here)

from createGame import *
from collections import defaultdict

"""
The first state is Root and Elements, which is expanded already by default.
The first action to choose is among {same, error, different}.
This is just bandit arms without states. Just nodes.

Calculate UCB1 for each node (same, error, different).
When neither of them was visited, choose a random one.
Check if the node was visited. If not, roll out to the terminal state.
Update Q and N of the child node and the parent node (root).

Go to the initial state and update the UCB1 value of the root, and the child nodes'.
Go to another element node and rollout, and then update the V and Q of that very node.
And also update the V and Q of the root node.

After all nodes are already visited, chose the best child of the highest UCB and expand.
"""


class Node:
    """
    A representation of a single board state.
    This class receives "states," "actions," "answer" and change "states" and "actions" into nodes.
    MCTS constructs a tree of these Nodes.

    everything regarding prbIdx (which is not for the Game function)
    1. find_children
    2. find_random_child
    3. is_terminal
    4. reward
    """
    # the first action of the elements
    nodeID = 0
    # Records the number of times states have been visited
    visits = defaultdict(lambda: 0)

    def __init__(self, element, parent=None, parentAction=None):
        """
        It may be confusing, but the income of the class is parent.
        This is the total env and going to be looping through trials.
        parent is the current state initialized with empty children
        """

        super().__init__()
        self.parent = parent  # None for root
        self.parentAction = parentAction  # None for root
        self.element = element
        self.nodeID = Node.nodeID  # There are 16 node IDs

        # N is both for (state) or (state, action) pairs.
        self.N = 0

        # Q is for (state, action) pairs only.
        self.Q = 0

        # A state node has child nodes (state and action pairs))
        self.children = {}

        Node.nodeID += 1

    @property
    def isFullyExpanded(self):
        """
        Check if this is a leaf node.
        If not a leaf node,
        choose the best child maximizing UCB1.
        If never been sampled, roll out.
        Else, add the new state and select the random child
        """
        if self.element == 0:  # depth 0 == root state.
            return np.sum(self.navi[:, self.element]) == 3
        else:
            return np.sum(self.navi[self.element, :]) == 5

    def addChild(self):
        print("self.element", self.element)
        self.children[self.nodeID] = self.legalMove(self.element)
        return self.children[self.nodeID]