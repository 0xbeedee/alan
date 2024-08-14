class Alan:
    """The generic agent from which to inherit in order to concretely construct agents for environments."""

    def __init__(self, env, num_steps_before_sleep):
        self.env = env
        self.num_steps_before_sleep = num_steps_before_sleep
        self.num_steps_awake = 0

    # TODO
    def act(self, *args, **kwargs):
        pass

    # TODO
    def update_models(self, *args, **kwargs):
        pass
