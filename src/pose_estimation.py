import yaml
import os
import numpy as np
from collections import deque
from pose_estimation_utils.TDDFA_ONNX import TDDFA_ONNX
from pose_estimation_utils.utils.pose import viz_pose


class PoseEstimation:
    """Estimate head pose according to the facial landmarks"""

    def __init__(self):
        cfg = yaml.load(open('./pose_estimation_utils/configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '4'
        self.tddfa = TDDFA_ONNX(**cfg)
        self.n_pre = 2  # average smoothing by looking on n_pre poses

    def __call__(self, frame, student):
        self.pose_estimation(frame, student)
        frame_with_pose = self.calculate_angles(frame, student)
        return frame_with_pose

    def pose_estimation(self, frame, student):
        frame_bgr = frame[..., ::-1]
        for name in student.group:
            face = student.group[name].face_coordinates
            if face is not None:
                if student.group[name].landmarks is None:

                    param_lst, roi_box_lst = self.tddfa(frame_bgr, [face])
                    landmarks = self.tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)[0]
                    student.group[name].landmarks = deque([landmarks]*self.n_pre)
                    student.group[name].param_lst = deque([param_lst[0]] * self.n_pre)

                param_lst, roi_box_lst = self.tddfa(frame_bgr, [student.group[name].landmarks[-1]], crop_policy='landmark')
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

                    student.group[name].landmarks = deque([landmarks] * self.n_pre)
                    student.group[name].param_lst = deque([param_lst[0]] * self.n_pre)

                student.group[name].landmarks.popleft()
                student.group[name].param_lst.popleft()

                student.group[name].landmarks.append(landmarks.copy())
                student.group[name].param_lst.append(param_lst[0].copy())

            else:
                student.group[name].landmarks = None
                student.group[name].param_lst = None

    @ staticmethod
    def calculate_angles(frame, student):
        frame_with_pose = frame.copy()
        for name in student.group:
            person = student.group[name]
            if person.param_lst is not None:
                ver_ave = np.mean(person.landmarks, axis=0)
                param_ave = np.mean(person.param_lst, axis=0)
                frame_with_pose, angles = viz_pose(frame_with_pose, param_ave, ver_ave, person.student_mark)
                person.angles = angles

        return frame_with_pose


