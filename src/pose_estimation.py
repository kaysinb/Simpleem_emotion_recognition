import yaml
import os
import numpy as np
from collections import deque
from pose_estimation_utils.TDDFA_ONNX import TDDFA_ONNX
from pose_estimation_utils.utils.pose import viz_pose, plot_pose_box


class PoseEstimation:
    """Estimate head pose according to the facial landmarks"""

    def __init__(self):
        cfg = yaml.load(open('./pose_estimation_utils/configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '4'
        self.tddfa = TDDFA_ONNX(**cfg)
        self.n_pre = 2  # average smoothing by looking on n_pre poses

    def __call__(self, frame, student, class_name):
        self.pose_estimation(frame, student, class_name)
        self.calculate_angles(frame, student, class_name)


    def pose_estimation(self, frame, student, class_name):
        frame_bgr = frame[..., ::-1]
        for name in student.group[class_name]:
            face = student.group[class_name][name].face_coordinates
            if face is not None:
                if student.group[class_name][name].landmarks is None:

                    param_lst, roi_box_lst = self.tddfa(frame_bgr, [face])
                    landmarks = self.tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)[0]
                    student.group[class_name][name].landmarks = deque([landmarks]*self.n_pre)
                    student.group[class_name][name].param_lst = deque([param_lst[0]] * self.n_pre)

                param_lst, roi_box_lst = self.tddfa(frame_bgr, [student.group[class_name][name].landmarks[-1]],
                                                    crop_policy='landmark')
                landmarks = self.tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)[0]

                lm_mean_x = np.mean(landmarks[0])
                lm_mean_y = np.mean(landmarks[1])
                left = face[0]
                right = face[2]
                top = face[1]
                bottom = face[3]

                # if face is lost
                if lm_mean_x < left or lm_mean_x > right or lm_mean_y < top or lm_mean_y > bottom:
                    param_lst, roi_box_lst = self.tddfa(frame_bgr, [face])
                    landmarks = self.tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)[0]
                    param_lst, roi_box_lst = self.tddfa(frame_bgr, [landmarks],
                                                        crop_policy='landmark')
                    landmarks = self.tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)[0]

                    student.group[class_name][name].landmarks = deque([landmarks] * self.n_pre)
                    student.group[class_name][name].param_lst = deque([param_lst[0]] * self.n_pre)

                student.group[class_name][name].landmarks.popleft()
                student.group[class_name][name].param_lst.popleft()

                student.group[class_name][name].landmarks.append(landmarks.copy())
                student.group[class_name][name].param_lst.append(param_lst[0].copy())

            else:
                student.group[class_name][name].landmarks = None
                student.group[class_name][name].param_lst = None

    @ staticmethod
    def calculate_angles(frame, student, class_name):
        frame_with_pose = frame.copy()
        for name in student.group[class_name]:
            person = student.group[class_name][name]
            if person.param_lst is not None:
                ver_ave = np.mean(person.landmarks, axis=0)
                param_ave = np.mean(person.param_lst, axis=0)
                P, angles = viz_pose(param_ave)
                person.angles = angles
                person.P = P
                person.ver = ver_ave
            else:
                person.angles = None
                person.P = None
                person.angles = {'yaw': None, 'pitch': None, 'roll': None}

    @staticmethod
    def plot_pose(frame, student, class_name):
        frame_with_pose = frame.copy()
        for name in student.group[class_name]:
            person = student.group[class_name][name]
            if person.P is not None and person.ver is not None:
                frame_with_pose = plot_pose_box(frame_with_pose, person.P, person.ver, person.student_mark)
        return frame_with_pose


