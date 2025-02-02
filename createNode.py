#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim) at 18:57 on 31/1/25
# Title: (Enter feature name here)
# Explanation: (Enter explanation here)

from createGame import *
from collections import defaultdict

"""
The first state is Root and Elements, which is by default.

The first action to choose is among {same, error, different}.
This is just arms without state. Just nodes.

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
    This class receives game trials and change each state and action into nodes.
    MCTS constructs a tree of these Nodes.

    everything regarding prbIdx (which is not for the Game function)
    1. find_children
    2. find_random_child
    3. is_terminal
    4. reward
    """

    def __init__(self, current=None, state=None, parentAction=None, parent=None):
        """
        This is the total env and going to be looping through trials.
        """
        self.prbInit = Game.initPrb()
        self.state = state
        self.parentAction = parentAction
        self.parent = parent

        # current = row: element, column: dim
        self.current = None

        # N is both for (state) or (state, action) pairs.
        self.N = defaultdict(int)

        # Q is for (state, action) pairs only.
        self.Q = defaultdict(int)

        # This is rewards
        self.result = defaultdict(int)

        # A state node has child nodes ((state, action) pairs)
        self.children = dict()

    def addChild(self, children: dict) -> None:
        for child in children:
            self.children[child.action] = child

    def isExpanded(self):
        """
        Check if this is a leaf node.
        If not a leaf node,
        choose the best child maximizing UCB1.
        If never been sampled, roll out.
        Else, add the new state and select the random child
        """
        return len(self.children) > 0

    def isTerminal(self):
        """
        Loop forever:
        If never been sampled, roll out with state and random action.
        Else, add the new state and select the random child
        """
        return np.sum(len(self.actionAvail)) == 0 and len(self.children) > 0

    def getReward(self, terminalAction):
        if terminalAction == self.answer:
            rwd = 1
        else:
            rwd = 0
        reward = self.leafVal[terminalAction] + rwd
        return reward
