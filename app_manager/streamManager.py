import gradio as gr
import numpy as np


class GradioStreamManager:
    def __init__(self):
        self.frame = np.ones((720, 1280, 3), dtype=np.uint8) * 100 

    def update(self, new_frame):
        self.frame = new_frame.copy()

    def get(self):
        return self.frame.copy()
