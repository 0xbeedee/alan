import time
import os
import datetime
import struct

import gymnasium as gym
from nle import nethack


class RecordTTY(gym.Wrapper):
    """Records the TTY "frames" seen as the agent steps in the NetHack environment and stores them for later replay."""

    def __init__(
        self,
        env: gym.Env,
        output_path: str,
    ):
        super().__init__(env)
        self.output_path = output_path
        self.recording_file = None
        self.start_time = None
        # we're not allowed to access the env's _observation_keys directly
        self._observation_keys = list(
            (
                "glyphs",
                "chars",
                "colors",
                "specials",
                "blstats",
                "message",
                "inv_glyphs",
                "inv_strs",
                "inv_letters",
                "inv_oclasses",
                "screen_descriptions",
                "tty_chars",
                "tty_colors",
                "tty_cursor",
            ),
        )

    def reset(self):
        if self.recording_file is None:
            self.start_recording()
        self.start_time = time.time()
        obs, info = self.env.reset()
        self.record_frame()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.record_frame()
        return obs, reward, terminated, truncated, info

    def render(self):
        # NOTE: this override makes render() non-standard: do not use it together with Tianshou's Collector!
        obs = self.env.unwrapped.last_observation
        tty_chars = obs[self._observation_keys.index("tty_chars")]
        tty_colors = obs[self._observation_keys.index("tty_colors")]
        tty_cursor = obs[self._observation_keys.index("tty_cursor")]
        return nethack.tty_render(tty_chars, tty_colors, tty_cursor)

    def close(self):
        if self.recording_file is not None:
            self.recording_file.close()
            self.recording_file = None
        return self.env.close()

    def start_recording(self):
        self.recording_file = open(self.output_path, "wb")

    def record_frame(self):
        if self.recording_file is None:
            return

        frame = self.render()
        timestamp = time.time() - self.start_time

        # write frame to file in ttyrec format
        self.recording_file.write(
            struct.pack(
                "<iiiB", int(timestamp), int((timestamp % 1) * 1e6), len(frame), 0
            )
        )
        self.recording_file.write(frame.encode("utf-8"))
