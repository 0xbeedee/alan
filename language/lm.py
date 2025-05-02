class LanguageModel:
    # TODO should provide an alternative way of generating rewards by using language models (replacing the old intrinsic system), and perhaps some reasoning capabilities/system 2-type capabilities => REWARDS ARE UNIVERSAL!
    def __init__(self) -> None:
        # TODO this should load the (pre-trained) model and set it up
        pass

    def prepare_model(self, context):
        # TODO prepares the model by feeding it the context (e.g., in the case of FrozenLake, having it read the FrozenLake entry in the gymnasium docs)
        pass

    def rewrite_rewards_(self, knowledge_base):
        # TODO rewrites all the rewards of the various transitions within the trajectories of the KB (we then train our policy on these trajectories)
        pass
