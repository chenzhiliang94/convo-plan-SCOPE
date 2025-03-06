import random

class MDP:
    """Return all states of this MDP"""

    def get_states(self):
        abstract

    """ Return all actions with non-zero probability from this state """

    def get_actions(self, state):
        abstract

    """ Return all non-zero probability transitions for this action
        from this state, as a list of (state, probability) pairs
    """

    def get_transitions(self, state, action):
        abstract

    """ Return the reward for transitioning from state to
        nextState via action
    """

    def get_reward(self, state, action, next_state):
        abstract

    """ Return true if and only if state is a terminal state of this MDP """

    def is_terminal(self, state):
        abstract

    """ Return the discount factor for this MDP """

    def get_discount_factor(self):
        abstract

    """ Return the initial state of this MDP """

    def get_initial_state(self):
        abstract

    """ Return all goal states of this MDP """

    def get_goal_states(self):
        abstract

    """ Return a new state and a reward for executing action in state,
    based on the underlying probability. This can be used for
    model-free learning methods, but requires a model to operate.
    Override for simulation-based learning
    """

    def execute(self, state, action):
        rand = random.random()
        cumulative_probability = 0.0
        for (new_state, probability) in self.get_transitions(state, action):
            if cumulative_probability <= rand <= probability + cumulative_probability:
                return (new_state, self.get_reward(state, action, new_state))
            cumulative_probability += probability
            if cumulative_probability >= 1.0:
                raise (
                    "Cumulative probability >= 1.0 for action "
                    + str(action)
                    + " from "
                    + str(state)
                )

        raise BaseException(
            "No outcome state in simulation for action"
            + str(action)
            + " from "
            + str(state)
        )

    """ 
    Execute a policy on this mdp for a number of episodes.
    Return the cumulative reward of each episode as a list.
    When True, random_on_duplicate detects when a state has been visited before, and selects a random action to avoid infinitely looping policies.
    """

    def execute_policy(self, policy, episodes=100, random_on_duplicate=False):
        cumulative_rewards = []
        states = set()
        for _ in range(episodes):
            cumulative_reward = 0.0
            state = self.get_initial_state()
            step = 0
            while not self.is_terminal(state):
                if state in states and random_on_duplicate:
                    action = random.choice(self.get_actions(state))
                else:
                    action = policy.select_action(state)
                    if random_on_duplicate: states.add(state)

                (next_state, reward) = self.execute(state, action)
                cumulative_reward += reward * (self.discount_factor ** step)
                state = next_state
                step += 1
            cumulative_rewards += [cumulative_reward]
        return cumulative_rewards
