# TODO better name
class AcclimatisationModule:
    # TODO that's a whole lot of hyperparameters to tune...
    def __init__(
        self, damping_factor=0.1, delta_scale=0.1, extrinsic_scale=1.0, smoothing=0.95
    ):
        self.damping_factor = damping_factor
        self.delta_scale = delta_scale
        self.extrinsic_scale = extrinsic_scale
        self.smoothing = smoothing
        self.previous_intrinsic_reward = 0
        self.ema_intrinsic = 0  # Exponential Moving Average for intrinsic reward

    def compute_reward(self, intrinsic_reward, extrinsic_reward):
        # Dampen the intrinsic reward
        damped_intrinsic = self.damping_factor * intrinsic_reward

        # Update Exponential Moving Average
        self.ema_intrinsic = (
            self.smoothing * self.ema_intrinsic
            + (1 - self.smoothing) * damped_intrinsic
        )

        # Compute delta based on EMA
        delta = damped_intrinsic - self.ema_intrinsic

        # Scale delta and combine with extrinsic reward
        total_reward = (self.delta_scale * delta) + (
            self.extrinsic_scale * extrinsic_reward
        )

        self.previous_intrinsic_reward = damped_intrinsic

        return total_reward, delta, extrinsic_reward

    def reset(self):
        self.previous_intrinsic_reward = 0
        self.ema_intrinsic = 0
