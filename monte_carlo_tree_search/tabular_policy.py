import random
from collections import defaultdict
from monte_carlo_tree_search.policy import DeterministicPolicy


class TabularPolicy(DeterministicPolicy):
    def __init__(self, default_action=None):
        self.policy_table = defaultdict(lambda: default_action)

    def select_action(self, state):
        return self.policy_table[state]

    def update(self, state, action):
        self.policy_table[state] = action
