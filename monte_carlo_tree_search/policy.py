class Policy:
    def select_action(self, state):
        abstract


class DeterministicPolicy(Policy):
    def update(self, state, action):
        abstract


class StochasticPolicy(Policy):
    def update(self, states, actions, rewards):
        abstract

    def get_probability(self, state, action):
        abstract
