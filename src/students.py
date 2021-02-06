import cv2
import os
import numpy as np
import time
import pandas as pd
from collections import deque
from src.pose_estimation import Stabilizer
from src.pose_estimation import PoseEstimation


class Student:
    """ """
    group = {}  # Dict of all students. Key - name. Value - object.
    names = []  # Names of students. The same as embeddings
    embeddings = None  # Embeddings matrix of all students. Order the same as students/

    def __init__(self, path, name, detector, list_of_emotions):

        # Unique properties of the student
        self.name = name
        self.path = path
        self.face_image = None
        self._number_of_embeddings = 0  # Number of embeddings used for student. Is used to drop Inf values during unpacking
        student_embeddings = None  # All embeddings of student in np.array
        self._stable_pose = None

        # Instant properties
        self.face_coordinates = None
        self.photos = []
        self._pose = None
        self.landmarks = None  # Landmarks
        self.emotions = None

        # Properties which is used for logging
        self._emotion_logg = deque()
        self._angle_logg = deque()
        self._logging_time = deque()
        self._stud_is_on_frame = deque()

        # Service properties
        self.list_of_emotions = list_of_emotions

        self.pose_stabilizers = [Stabilizer(
            state_num=2,
            measure_num=1,
            cov_process=1,
            cov_measure=0.0001) for _ in range(6)]

        # Generate embeddings for each student

        for file_name in os.listdir(self.path):
            print('Getting photo .... : ' + path + '/' + file_name)
            image = cv2.imread(path + '/' + file_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            self.photos.append(image)

            new_embedding = detector.get_initial_embedding(image)

            if new_embedding is not None:
                self._number_of_embeddings += 1
                if student_embeddings is not None:
                    student_embeddings = np.dstack((student_embeddings, new_embedding))
                else:
                    student_embeddings = new_embedding
            else:
                print('Bad photo for initialization {}!!'.format(path + '/' + file_name))

        Student.group[self.name] = self
        Student.names.append(self.name)

        # Adding embedding matrix of student to common matrix

        infinite_value = np.inf

        if Student.embeddings is not None:
            stu_emb_depth = Student.embeddings.shape[2]  # Depth of all students embedding
            slf_emb_depth = student_embeddings.shape[2]  # Depth of student embedding

            # Concatenation of arrays with notequal shape
            if stu_emb_depth == slf_emb_depth:
                Student.embeddings = np.vstack((Student.embeddings, student_embeddings))
            elif stu_emb_depth > slf_emb_depth:
                block_to_add_shape = (1, Student.embeddings.shape[1], stu_emb_depth - slf_emb_depth)
                block_to_add = np.full(block_to_add_shape, infinite_value)
                Student.embeddings = np.vstack((Student.embeddings, np.dstack((student_embeddings, block_to_add))))

            elif stu_emb_depth < slf_emb_depth:
                block_to_add_shape = (
                    Student.embeddings.shape[0], Student.embeddings.shape[1], slf_emb_depth - stu_emb_depth)
                block_to_add = np.full(block_to_add_shape, infinite_value)
                Student.embeddings = np.vstack((np.dstack((Student.embeddings, block_to_add)), student_embeddings))

        else:
            Student.embeddings = student_embeddings

    # Methods for pose estimation
    @property
    def pose(self):
        """ Getter of the stable pose """
        return self._stable_pose

    @pose.setter
    def pose(self, pose):
        """ Setter of the pose with filter smoothing"""
        steady_pose = []
        pose_np = np.array(pose).flatten()
        for value, ps_stb in zip(pose_np, self.pose_stabilizers):
            ps_stb.update([value])
            steady_pose.append(ps_stb.state[0])
        steady_pose = np.reshape(steady_pose, (-1, 3))

        self._pose = pose
        self._stable_pose = steady_pose

    # Methods for logging
    def logging(self):
        """ Emotions and pose logging for frame """

        if self.face_coordinates is None:
            self._stud_is_on_frame.appendleft(False)
            self._angle_logg.appendleft([None] * 3)
        else:
            self._stud_is_on_frame.appendleft(True)
            self._angle_logg.appendleft(PoseEstimation.rot_params_rv(self._pose[0]))

        self._logging_time.appendleft(time.time())
        self._emotion_logg.appendleft(self.emotions)

    def get_student_logs(self):
        """ Getting total logs for one of the student """

        time_df = pd.DataFrame(list(self._logging_time), columns=['time'])
        emotion_df = pd.DataFrame(list(self._emotion_logg), columns=self.list_of_emotions)
        angle_df = pd.DataFrame(list(self._angle_logg), columns=['roll', 'pitch', 'yaw'])
        onframe_df = pd.DataFrame(list(self._stud_is_on_frame), columns=['is_on_frame'])

        output_df = pd.concat([time_df, emotion_df, angle_df, onframe_df], axis=1)

        return output_df
