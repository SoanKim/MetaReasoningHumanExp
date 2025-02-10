#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim) at 18:57 on 31/1/25
# Title: (Enter feature name here)
# Explanation: (Enter explanation here)

from createGame import *

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
from collections import defaultdict

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

    def __init__(self, parent=None, parentAction=None):
        """
        It may be confusing, but the income of the class is parent.
        This is the total env and going to be looping through trials.
        parent is the current state initialized with empty children
        """
        self.parent = parent  # None for root
        self.parentAction = parentAction  # None for root
        # matrix coordinates before depth 2, from 2 onwards, cardCandidates

        # state is numAvailActions
        #self.state = np.zeros((1, 3)) if state is None else state

        # N is both for (state) or (state, action) pairs.
        self.N = 0

        # Q is for (state, action) pairs only.
        self.Q = -np.inf

        # a state node has child nodes (state and action pairs))
        self.children = defaultdict(int)

        # update depth.
        self.depth = 0

        """
        |---|---|---|---|---|
        | 0 | 0 | 1 | 2 | 3 |
        | 1 | 0 | 1 | 2 | 3 |
        | 2 | 0 | 1 | 2 | 3 |
        |---|---|---|---|---|
        sum(1st col) < 3: depth 0
        sum(1st col) == 3: depth 1
        sum(2nd-4th cols) < 12: depth 1
        sum(2nd-4th cols) == 12: depth 2
        
        before selecting a leaf val: depth 2
        after selecting a leaf val: depth 3
        after receiving reward = terminal
        
        """
    def addChild(self, action, depth, ucb):
        self.children['action'].append(action)
        self.children['depth'].append(depth)
        self.children['ucb'].append(ucb)

    def trackDepth(self, navi):
        """
        Check if this is a leaf node.
        If not a leaf node,
        choose the best child maximizing UCB1.
        If never been sampled, roll out.
        Else, add the new state and select the random child
        """
        if np.sum(navi[:, 0]) < 3:
            self.depth = 0
        elif np.sum(navi[:, 0]) == 3 and np.sum(navi[:, 1:]) < 12:
            self.depth = 1
        elif np.sum(navi[:, 0]) == 3 and np.sum(navi[:, 1:]) == 12:
            self.depth = 2
        return self.depth

    def isFullyExpanded(self, navi):
        if self.depth == 0:
            return np.sum(navi[:, 0]) == 3
        elif self.depth == 1:
            return np.sum(navi[:, 1:]) == 12
        elif self.depth == 2:
            return True


# if __name__ == '__main__':
#
#     game = Game()
#     contextM, cardAvail, answer, navi, leafState = game.prbInit()
#     root = Node()
#     print(root.parent)