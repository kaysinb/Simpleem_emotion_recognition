import cv2
import numpy as np


class DisplayText:

    def __init__(self, initial_frame_shape):
        self.shape = (initial_frame_shape/np.array([7, 1, 1])).astype(int)
        self.full_frame_shape = initial_frame_shape + np.array([self.shape[0], 0, 0])

    def show_text(self, frame, text, font_scale=2, color=(255, 0, 0)):
        response_frame = np.zeros(self.shape, dtype="uint8")
        one_symbol_width = int(37 / 2 * font_scale)
        x = self.shape[1] // 2 - len(text) // 2 * one_symbol_width + 15
        y = int(self.shape[0] / 1.5)
        cv2.putText(response_frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, 2, cv2.LINE_AA)

        full_frame = np.concatenate((response_frame, frame))
        return full_frame
