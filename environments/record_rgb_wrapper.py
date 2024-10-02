import gymnasium as gym
import numpy as np
import cv2


class RecordRGB(gym.Wrapper):
    """Records the RGB arrays seen as the agent steps in the environment and stores the video for later replay."""

    def __init__(self, env: gym.Env, output_path: str, fps: int = 30):
        super().__init__(env)
        self.output_path = output_path
        self.fps = fps
        self.frames = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.start_recording()
        self.record_frame()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.record_frame()
        return obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.stop_recording()
        return self.env.close()

    def start_recording(self):
        self.frames = []

    def stop_recording(self):
        if self.frames:
            self.save_video()

    def record_frame(self):
        frame = self.render()
        self.frames.append(frame)

    def save_video(self):
        if not self.frames:
            print("No frames to save.")
            return

        height, width, _ = self.frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (width, height))

        for frame in self.frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        out.release()
        self.frames = []
