- Example
<img src="https://github.com/SoanKim/MetaReasoningHumanExp/blob/0116c2a7d2007356a6385c6cd3568c7b725b8f56/mcts.gif"/>

# Metareasoning on MCTS
## There are 7 files:
### 1. createGame.py: 
- Initialize a problem (import a problem and answer from human data per trial) </br>
- Map the problem structure into a grid.
- Next state function
- Terminal function
- Reward function
```
inputs: problem index, action chosen by MCTS
outputs: state matrix (self.context), available cards on the leaf node(self.cardAvail list), answer (scalar)

self.actions (3*5)
| R | C | F | S | B |
|---|---|---|---|---|
| 0 | 0 | 1 | 2 | 3 |
| 1 | 0 | 1 | 2 | 3 |
| 2 | 0 | 1 | 2 | 3 |

np.argwhere (self.navi)
| Root|Color| Fill|Shape| Back| 
|-----|-----|-----|-----|-----|
| 0 0 | 0 1 | 0 2 | 0 3 | 0 4 |
| 1 0 | 1 1 | 1 2 | 1 3 | 1 4 |
| 2 0 | 2 1 | 2 2 | 2 3 | 2 4 |

self.contexts (one example): mapping the structure of a problem into a table
| R | C | F | S | B |
|---|---|---|---|---|
| 3 | 1 | 1 | 0 | 1 |
| 30| 9 | 9 | 6 | 6 |
| 7 | 0 | 0 | 4 | 3 |

inputs: subject index (first subject) and problem index
outputs: problem (self.prb) -> array, shape: (5, 4)
         answer(self.prbAnswer), -> scalar
         card candidates per leaf(self.cardAvail) -> list, len(3), with 4 dimensions by 3 embedded.
         and the leaf value penalizeing based on the num of candidates (self.leafVal) -> array(3, 4)
```
### 2. createNode.py:
- Turn each (state) or (state, action) into nodes </br>
- Nodes are memory-less.
- The node does not store contexts: only place-holder-like states, the chosen actions (S-A-S'-A'), and UCB1.
- Instead, the context (i.e., cards) of the nodes go into replayBuffer of the agent to learn between-trial knowledge.
- Properties: number of visits (self.N), values (self.Q), parent action (self.parentAction), and children. 
- Function: addChild (next state according to current action), isExpanded, 
```
inputs: self.prb (array), self.prbAnswer (scalar), cardAvail (list), leafVal (array(3, 4)))
outputs: self.Q, self.N
```
### 3. createMCTS.py:
- functions: Selects, expands, simulate, update the nodes.
- select 
```
inputs: children
output: bestChild (action) using UCB1.
```
- expand function
```
inputs: bestChild
output: next state of the bestChild
```
- rollout function
```
inputs: current child
output: next state, random action, Q (value and reward)
```
- backprop function
```
inputs: Q
output: update the Q and N
```
### 4. agent.py:
- uses replay buffer to store state, action, Q, N, reward of the total problem

### 5. main.py:
- iterate through the 90 problems
- agent solves each problem
- keep track of the scores to compare it to a human's

### 6. results:
- Learning curve (MCTS)
<img src="https://github.com/SoanKim/MetaReasoningHumanExp/blob/919bc25b5d7b96d8cd50a16f85a656e74146eb0d/MCTSresult.png" width="50%" height="50%"/>
- Learning curves (Human)
<img src="https://github.com/SoanKim/MetaReasoningHumanExp/blob/42355ce9bf83bcd26c786855e9a0a8f85ae3fe55/%5BExp1.%5D%20human_accuracy.png" width="50%" height="50%"/>
- Q table
 <img src="https://github.com/SoanKim/MetaReasoningHumanExp/blob/5ffbdb30a9b71e3999547896feee5726c0cce633/Q.gif" width="50%" height="50%"/>
- UCB scores
 <img src="https://github.com/SoanKim/MetaReasoningHumanExp/blob/5ffbdb30a9b71e3999547896feee5726c0cce633/ucb.gif" width="50%" height="50%"/>
- Number of visits
 <img src="https://github.com/SoanKim/MetaReasoningHumanExp/blob/5ffbdb30a9b71e3999547896feee5726c0cce633/visits.gif" width="50%" height="50%"/>