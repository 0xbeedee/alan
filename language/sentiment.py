import torch
from transformers import pipeline

import numpy as np


class SentimentAnalyser:
    """A module that assigns rewards to transitions within completed trajectories based on the sentiment of the messages provided by the environment.

    N.B.: The class assumes that the messages are provided by the `obs` field of the batch.
    """

    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
            # XXX: for some reason, device=cpu breaks the pipeline
            # device=device,
        )

        self.device = device

    def get_reward(self, obs_array: np.ndarray) -> np.ndarray:
        # TODO: this could probably be parallelised
        messages = [obs["message"] for obs in obs_array]
        rews = np.zeros(len(messages), dtype=np.float32)
        for i, message in enumerate(messages):
            if message:
                sentiment = self.sentiment_pipeline(message)
                factor = 1.0 if sentiment[0]["label"] == "POSITIVE" else -1.0
                # sentiment rews are in the [-1, 1] range (score is in [0, 1])
                rews[i] = factor * sentiment[0]["score"]
            else:
                rews[i] = 0.0
        return rews
