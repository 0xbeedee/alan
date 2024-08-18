from intrinsic.icm import ICM


class SelfModel:
    def __init__(self, obs_net, action_space) -> None:
        self.intrinsic_module = ICM(obs_net, action_space.n)

    def __call__(self, obs_batch, act_batch, obs_prime_batch, sleep=False):
        i_reward, _ = self.intrinsic_module.forward(
            obs_batch, act_batch, obs_prime_batch
        )

        if sleep:
            self._sleep()

        # the policy will take this reward an extrinsic one as input and decide how to act
        return i_reward

    # def _sleep(self):
    #     aggregate_experience()
    #     build_hierarchies()
    #     add_new_knowledge_to_KB()
