import math
import time
import random
from collections import defaultdict


class Node:

    # Record a unique node id to distinguish duplicated states
    next_node_id = 0

    # Records the number of times states have been visited
    visits = defaultdict(lambda: 0)

    def __init__(self, mdp, parent, state, qfunction, bandit, reward=0.0, action=None):
        self.mdp = mdp
        self.parent = parent
        self.state = state
        self.id = Node.next_node_id
        Node.next_node_id += 1

        # The Q function used to store state-action values
        self.qfunction = qfunction

        # A multi-armed bandit for this node
        self.bandit = bandit

        # The immediate reward received for reaching this state, used for backpropagation
        self.reward = reward

        # The action that generated this node
        self.action = action

    """ Select a node that is not fully expanded """

    def select(self): abstract


    """ Expand a node if it is not a terminal node """

    def expand(self): abstract


    """ Backpropogate the reward back to the parent node """

    def back_propagate(self, reward, child): abstract


    """ Return the value of this node """

    def get_value(self):
        (_, max_q_value) = self.qfunction.get_max_q(
            self.state, self.mdp.get_actions(self.state)
        )
        return max_q_value

    """ Get the number of visits to this state """

    def get_visits(self):
        return Node.visits[self.state]


class MCTS:
    def __init__(self, mdp, qfunction, bandit, terminating_heuristic_q_function=None):
        self.mdp = mdp
        self.qfunction = qfunction
        self.bandit = bandit
        self.terminating_heuristic_q_function = terminating_heuristic_q_function

    """
    Execute the MCTS algorithm from the initial state given, with timeout in seconds
    """

    def mcts(self, timeout=100, root_node=None, seed=None):
        if root_node is None:
            root_node = self.create_root_node()
            #print(root_node.state)

        start_time = time.time()
        current_time = time.time()
        simulation_rollout_count = 0
        do_nothing_count = 0
        initial_actions = None
        print("time out for mcts given as: ", timeout)
        while current_time < start_time + timeout:
            
            # Find a state node to expand
            selected_node = root_node.select()
            print(f"selected node depth: {selected_node.state.depth}")
            
            # if we can expand some more
            if not self.mdp.is_terminal(selected_node.state):
                child, action_in_expansion = selected_node.expand()
                if initial_actions is None:
                    self.initial_actions = action_in_expansion
                    initial_actions = 1
                reward = self.simulate(child, seed=seed)
                print("cumulative reward after expansion and simulation: ", reward)
                selected_node.back_propagate(reward, child)
            else:
                do_nothing_count+=1
                print("fully expanded tree. using simple back propagation: ", do_nothing_count)
                selected_node.back_propagate_simple(0.0)
            
            simulation_rollout_count +=1
            print("time taken for one iteration of mcts: ", time.time() - current_time)
            current_time = time.time()
        print("number of rollouts achieved: ", simulation_rollout_count)
            

        return root_node

    """ Create a root node representing an initial state """

    def create_root_node(self): abstract


    """ Choose a random action. Heustics can be used here to improve simulations. """

    def choose(self, state):
        return random.choice(self.mdp.get_actions_in_simulation(state))

    """ Simulate until a terminal state (TODO: DIFFERENT FROM GET OUTCOME, because this is pure random with no fixed action)""" 

    def simulate(self, node, seed=None):
        state = node.state
        cumulative_reward = 0.0
        depth = 0
        while not self.mdp.is_terminal(state): # termination here is governed by max depth given in mdp
            # Choose an action to execute
            action = self.choose(state)
            # Execute the action
            (next_state, reward) = self.mdp.execute_in_simulation(state, action, seed=seed)
            print("one step reward in simulation: ", reward)

            # Discount the reward
            cumulative_reward += pow(self.mdp.get_discount_factor(), depth) * reward
            depth += 1

            state = next_state
            print("simulating... state depth: ", state.depth)
        
        # in addition, apply a heuristic to the terminating state given by q-function.
        if self.terminating_heuristic_q_function is not None:
            print("getting terminal action actions and rewards")
            # get possible actions
            # actions = self.mdp.get_actions_in_simulation(state)
            # # get the value of these actions and return the average
            # action_rewards = [self.mdp.get_reward(state.conversation, action, None) for action in actions]
            # cumulative_reward += pow(self.mdp.get_discount_factor(), depth) * float(sum(action_rewards)/len(action_rewards))
            cumulative_reward += 0.0 # don't use heuristic
        return cumulative_reward
