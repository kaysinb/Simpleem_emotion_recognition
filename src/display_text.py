import cv2
import numpy as np


class DisplayText:

    def __init__(self, initial_frame_shape):
        self.shape = (initial_frame_shape/np.array([5.5, 1, 1])).astype(int)
        self.full_frame_shape = initial_frame_shape + np.array([self.shape[0], 0, 0])

    def show_text(self, frame, recommendation, attendance):
        response_frame = np.zeros(self.shape, dtype="uint8")
        fons_scale_1 = 2
        thickness_1 = 2
        text_size_1 = cv2.getTextSize(recommendation[0], cv2.FONT_HERSHEY_SIMPLEX, fons_scale_1, thickness_1)[0]
        x_1 = int(response_frame.shape[1]/2 - text_size_1[0]/2)
        y_1 = int(text_size_1[1]*1.5)
        cv2.putText(response_frame, recommendation[0], (x_1, y_1), cv2.FONT_HERSHEY_SIMPLEX,
                    fons_scale_1, recommendation[1], thickness_1, cv2.LINE_AA)

        fons_scale_2 = 1
        thickness_2 = 1
        text_2 = attendance[0]
        text_size_2 = cv2.getTextSize(text_2, cv2.FONT_HERSHEY_SIMPLEX, fons_scale_2, thickness_2)[0]
        full_text = True

        while text_size_2[0] > response_frame.shape[1]:
            text_2 = text_2[:text_2.rfind(',')]
            text_size_2 = cv2.getTextSize(text_2 + ', et al.', cv2.FONT_HERSHEY_SIMPLEX, fons_scale_2, thickness_2)[0]
            full_text = False
        if not full_text:
            text_2 += ', et al.'


        x_2 = int(response_frame.shape[1]*0.01)
        y_2 = int(y_1 + text_size_2[1] * 2.0)
        cv2.putText(response_frame, text_2, (x_2, y_2), cv2.FONT_HERSHEY_SIMPLEX,
                    fons_scale_2, attendance[1], thickness_2, cv2.LINE_AA)

        full_frame = np.concatenate((response_frame, frame))
        return full_frame


def write_signature(frame, student):
    for student_name in student.group:
        student_face = student.group[student_name].face_coordinates
        if student_face is not None:
            emotion = student.group[student_name].current_emotion
            text = student_name + ' is ' + emotion
            text_scale = (student_face[2] - student_face[0]) / 125
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, 1)[0][0]
            x_text = int(((student_face[0] + student_face[2]) / 2) - text_size / 2)
            y_text = int(student_face[1])
            cv2.putText(frame, text, (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, text_scale,
                        student.group[student_name].student_mark, 1, 255, )

    return frame
