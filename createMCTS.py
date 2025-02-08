#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim) at 13:57 on 1/2/25
# Title: (Enter feature name here)
# Explanation: (Enter explanation here)

from createGame import *
from createNode import Node
from copy import deepcopy
import numpy as np
import random
import math


class MCTS(Node):
    """
    The tree builds up the information.
    Explore more promising nodes.
    This requires node and generates a tree.
    Select --> Expand --> Simulate --> Backup
    s_{t+1} = (s_{t}, a_{t+1})
    intRwd(intermediate reward) is different from external reward

    **to do**: to make function for calculating element according to the prob.
    """

    def __init__(self, prbIdx):  # game = an indexed problem
        super().__init__(prbIdx)
        self.explorConstant = 2
        self.game = Game(prbIdx)
        # initialize game
        self.contextM, self.cardAvail, self.answer, self.navi = self.game.prbInit()
        self.root = Node(self.navi)

    def UCB(self, child):
        """
        MAB algorithm
        Tree policy: Calculate UCB of each child
        """
        if child.N > 0:
            child.Q += child.V / child.N
            ucb = child.Q + self.explorConstant * math.sqrt(math.log(self.parent.N) / child.N)
        else:
            ucb = 0
        return ucb

    def select(self):  # Should select from the navi position
        """
        Select the node that you want to explore (expand),
        which is not fully expanded (all the actions available were explored).

        LATER, update it based on probabilities!
        """
        # if not self.isFullyExpanded:
        #     return self
        # else:
        children = list(self.children[Node.timeStep].values())  # element as an argument?

        bestChildren = []  # It can be plural
        bestUCB = -np.inf

        for child in children:
            ucb = self.UCB(child)
            if ucb > bestUCB:
                bestChildren.append(child)
                bestUCB += ucb

        # In case bestChild is plural
        bestChild = np.argmax(bestChildren)
        return bestChild

    def expand(self, depth, element):
        """
        Choosing an action available and append it to the tree.

        LATER, update it based on probabilities!
        """
        if not np.any(np.sum(self.navi[element], axis=0)) == 5:
            actions = self.legalMove(depth, element)
            action = random.choice(actions)

            self.children[action] = []
            return self.navi[element, action]
        return self

    def simulate(self, element):
        """
        MC simulation to the terminal state.
        Can be either heuristic or random.
        Receives a reward.
        Average out the reward.
        Backprop the value up the node and up the tree.
        """
        cumRwd = 0
        state = self.legalMove(self.depth, element)  # doesn't change the original.

        # while not self.isTerminal:
        action = random.choice(np.where(state[element] == 0))
        imRwd = self.contextM[self.depth, element, action]
        cumRwd += imRwd

        return cumRwd

    def backprop(self, reward):
        """ Updates reward and visit counts, propagating up the tree. """
        # Increment the number of visits
        self.visits += 1

        # Calculate the new average reward
        self.Q += (reward - self.Q) / self.N

        # Propagate reward to the parent
        if self.parent:
            self.parent.backprop(reward)


    # Tim Miller
        # selectedNode = self.select()  # row: element and col: action
        # selectedNodeAction = selectedNode.action
        # interRwd = self.simulate()
        # selectedNode.N[selectedNode] = selectedNode.N[selectedNode] + 1
        # selectedNode.Q[selectedNode, selectedNodeAction] = selectedNode.Q[selectedNode, selectedNodeAction] + interRwd

        # if self.elemState == 0:
        #     q_value = self.qTable[selectedNodeAction, self.elemState]
        #     delta = (1 / (self.N[selectedNode, selectedNodeAction])) * (interRwd - self.getValue())
        # self.qfunction.update(self.state, selectedNodeAction, delta)
        #
        # if self.parent != None:
        #     self.parent.back_propagate(self.reward + interRwd, self)

    #""" Simulate the outcome of an action, and return the child node """

    # MODEL-BASED
    # def get_outcome_child(self, action):
    #     # Choose one outcome based on transition probabilities
    #     (next_state, reward, done) = self.mdp.execute(self.state, action)
    #
    #     # Find the corresponding state and return if this already exists
    #     for (child, _) in self.children[action]:
    #         if next_state == child.state:
    #             return child
    #
    #     # This outcome has not occured from this state-action pair previously
    #     new_child = SingleAgentNode(
    #         self.mdp, self, next_state, self.qfunction, self.bandit, reward, action
    #     )
    #
    #     # Find the probability of this outcome (only possible for model-based) for visualising tree
    #     probability = 0.0
    #     for (outcome, probability) in self.mdp.get_transitions(self.state, action):
    #         if outcome == next_state:
    #             self.children[action] += [(new_child, probability)]
    #             return new_child
