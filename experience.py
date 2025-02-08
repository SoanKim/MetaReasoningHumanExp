#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim) at 15:37 on 5/2/25
# Title: (Enter feature name here)
# Explanation: (Enter explanation here)

from collections import deque, namedtuple


class replayBuffer:
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, action_size=15, stateIdx=None, buffer_size=90):
        self.action_size = action_size  # It can vary depending on the depth of the tree.
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple(
            "Experience", field_names=["state", "context", "action", "reward", "nextState", "done"])
        self.stateIdx = 0 if None else stateIdx

    def add(self, state, context, action, reward, nextState, done):
        """Add a new experience to memory."""
        e = self.experience(state, context, action, reward, nextState, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = self.memory[self.stateIdx]

        states = [e.t for e in experiences if e is not None]
        contexts = [e.context for e in experiences if e is not None]
        actions = [e.action for e in experiences if e is not None]
        rewards = [e.reward for e in experiences if e is not None]
        nextStates = [e.nextState for e in experiences if e is not None]
        dones = [e.done for e in experiences if e is not None]
        return states, contexts, actions, rewards, nextStates, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


# if __name__ == "__main__":
#     replay = replayBuffer(3, 0).add()