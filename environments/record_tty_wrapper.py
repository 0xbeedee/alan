import time
import struct
from typing import Any, Tuple, Dict, SupportsFloat

import gymnasium as gym
from nle import nethack


class RecordTTY(gym.Wrapper):
    """Records the TTY "frames" seen as the agent steps in the NetHack environment and stores them for later replay in ttyrec format."""

    def __init__(
        self,
        env: gym.Env,  # should be NLE environment
        output_path: str,
    ):
        super().__init__(env)
        self.output_path = output_path
        self.recording_file = None
        self.start_time: float | None = None

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
        # store indices for faster access and clarity
        try:
            self._tty_chars_idx = self._observation_keys.index("tty_chars")
            self._tty_colors_idx = self._observation_keys.index("tty_colors")
            self._tty_cursor_idx = self._observation_keys.index("tty_cursor")
        except ValueError as e:
            raise ValueError(
                "Could not find expected NLE TTY keys in observation structure."
            ) from e

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> Tuple[Any, Dict[str, Any]]:
        if self.recording_file is None:
            self.start_recording()
        self.start_time = time.time()
        obs, info = self.env.reset(seed=seed, options=options)
        self.record_frame()
        return obs, info

    def step(
        self, action: Any
    ) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.record_frame()
        return obs, reward, terminated, truncated, info

    def render(self) -> str:
        """Non-standard render method for NLE TTY output.

        WARNING: This method accesses `env.unwrapped.last_observation` directly
        assuming a specific tuple structure based on NLE internals, and calls
        `nle.nethack.tty_render`. It does NOT call `self.env.render()`. This will likely break compatibility with standard tools (e.g., Tianshou Collectors) that expect `render()` to return RGB arrays or follow standard modes.
        It is intended specifically for use by this TTY recorder.
        """
        try:
            obs = self.env.unwrapped.last_observation
            tty_chars = obs[self._tty_chars_idx]
            tty_colors = obs[self._tty_colors_idx]
            tty_cursor = obs[self._tty_cursor_idx]
            return nethack.tty_render(tty_chars, tty_colors, tty_cursor)
        except (AttributeError, IndexError, TypeError) as e:
            print(f"Error accessing NLE observation for TTY rendering: {e}")
            # returning empty avoids crashing recording.
            return ""

    def close(self) -> None:
        if self.recording_file is not None:
            try:
                self.recording_file.close()
            except Exception as e:
                print(f"Error closing ttyrec file {self.output_path}: {e}")
            finally:
                self.recording_file = None
        self.env.close()

    def start_recording(self) -> None:
        try:
            self.recording_file = open(self.output_path, "wb")
        except IOError as e:
            print(f"Error opening ttyrec file {self.output_path}: {e}")
            self.recording_file = None  # None if open failed

    def record_frame(self) -> None:
        if self.recording_file is None or self.start_time is None:
            return

        frame_str = self.render()
        # if render failed, frame_str might be empty, skip writing
        if not frame_str:
            return

        timestamp = time.time() - self.start_time
        frame_bytes = frame_str.encode("utf-8")

        try:
            header = struct.pack(
                "<iiIB", int(timestamp), int((timestamp % 1) * 1e6), len(frame_bytes), 0
            )
            self.recording_file.write(header)
            self.recording_file.write(frame_bytes)
        except (IOError, struct.error) as e:
            print(f"Error writing frame to ttyrec file {self.output_path}: {e}")
