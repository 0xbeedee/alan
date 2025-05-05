import gymnasium as gym
import numpy as np
import cv2
from typing import Any, List, Tuple, Dict, SupportsFloat


class RecordRGB(gym.Wrapper):
    """Records the RGB arrays seen as the agent steps in the environment and stores the video for later replay."""

    def __init__(self, env: gym.Env, output_path: str, fps: int = 30):
        super().__init__(env)
        self.output_path = output_path
        self.fps = fps
        self.frames: List[np.ndarray] = []

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> Tuple[Any, Dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self.start_recording()
        self.record_frame()
        return obs, info

    def step(
        self, action: Any
    ) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.record_frame()

        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | List[np.ndarray] | None:
        # render mode might return list or None, handle flexibly
        return self.env.render()

    def close(self) -> None:
        self.stop_recording()
        self.env.close()

    def start_recording(self) -> None:
        self.frames = []

    def stop_recording(self) -> None:
        if self.frames:
            self.save_video()

    def record_frame(self) -> None:
        frame = self.render()
        if isinstance(frame, np.ndarray):
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                self.frames.append(frame)

    def save_video(self) -> None:
        if not self.frames:
            print("No frames to save.")
            return

        try:
            height, width, _ = self.frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (width, height))

            for frame in self.frames:
                # frame must be uint8
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            out.release()
        except Exception as e:
            print(f"Error saving video: {e}")
        finally:
            # clear frames even if saving failed
            self.frames = []
