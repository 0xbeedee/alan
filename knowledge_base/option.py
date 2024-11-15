from .utils import is_similar_state


class Option:
    """A class that executes options (https://www.sciencedirect.com/science/article/pii/S0004370299000521), as extracted from the knowledge base."""

    def __init__(self, initiation_set, policy, termination_condition):
        self.initiation_set = initiation_set
        self.policy = policy
        self.termination_condition = termination_condition

    def can_initiate(self, state):
        return is_similar_state(state, self.initiation_set)

    def execute(self, state):
        # TODO follow the option's policy until termination
        actions = []
        current_state = state
        for step in self.policy:
            action = step.action
            actions.append(action)
            current_state = environment_step(current_state, action)
            if self.termination_condition(current_state):
                break
        return actions


def environment_step(s, a):
    # TODO
    pass


# # modify agent's action selection
# def select_action(state):
#     for option in options_list:
#         if option.can_initiate(state):
#             return option.execute(state)
#     return agent_policy(state)
