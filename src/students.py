import cv2
import os
import numpy as np
import time
import pandas as pd
from collections import deque


class Student:
    """ """
    group = {}  # Dict of dicts of all students.  Key_0 - class_name, Key_1 - name, Value - object.
    names = {}  # Names - dict of students. The same as embeddings
    embeddings = {}  # Embeddings - dict of matrix of all students. Order the same as students/
    _logging_time = {}  # Dict of arrays with logging time
    start_lesson = {}  # Dict of flags
    time_length_const = {}  # Dict of time length consts

    def __init__(self, photos, name, class_name, detector, list_of_emotions):

        # Unique properties of the student
        self.name = name
        self.class_name = class_name
        self.face_image = None
        self._number_of_embeddings = 0  # Number of embeddings used for student. Is used to drop Inf values during unpacking
        student_embeddings = None  # All embeddings of student in np.array
        self.student_mark = (0, 255, 0)

        # Instant properties
        self.face_coordinates = None
        self.param_lst = None
        self.landmarks = None  # Landmarks
        self.emotions = None
        self.current_emotion = None
        self.angles = {'yaw': None, 'pitch': None, 'roll': None}

        # Properties which is used for logging
        self._emotion_logg = []
        self._angle_logg = []
        self._stud_is_on_frame = []

        # Service properties
        self.list_of_emotions = list_of_emotions
        self.list_of_pos_emotions = ['neutral', 'happy']
        self.indexes_of_pos_emotions = []

        # Init of indexes of positive and negative emotions
        for item in self.list_of_pos_emotions:
            try:
                self.indexes_of_pos_emotions.append(self.list_of_emotions.index(item))
            except ValueError:
                print('Positive emotion {} is not in list of emotions'.format(item))

        self.indexes_of_neg_emotions = [index for index in range(len(self.list_of_emotions)) if
                                        index not in self.indexes_of_pos_emotions]
        self.frame_color = (0, 0, 255)

        # Generate embeddings for each student

        for image in photos:
            new_embedding = detector.get_initial_embedding(image)

            if new_embedding is not None:
                self._number_of_embeddings += 1
                if student_embeddings is not None:
                    student_embeddings = np.dstack((student_embeddings, new_embedding))
                else:
                    student_embeddings = new_embedding
                    student_embeddings = np.expand_dims(student_embeddings, axis=0)
                    student_embeddings = np.expand_dims(student_embeddings, axis=2)
            else:
                print('Bad photo for initialization!')



        if self.class_name in Student.names:
            Student.names[self.class_name].append(self.name)
            Student.group[self.class_name][self.name] = self
        else:
            Student.names[self.class_name] = [self.name]
            Student.start_lesson[self.class_name] = True
            Student.group[self.class_name] = {}
            Student.group[self.class_name][self.name] = self
            Student._logging_time[self.class_name] = []


        # Adding embedding matrix of student to common matrix

        infinite_value = np.inf

        if self.class_name in Student.embeddings:
            stu_emb_depth = Student.embeddings[self.class_name].shape[2]  # Depth of all students embedding
            slf_emb_depth = student_embeddings.shape[2]  # Depth of student embedding

            # Concatenation of arrays with notequal shape
            if stu_emb_depth == slf_emb_depth:
                Student.embeddings[self.class_name] = np.vstack((Student.embeddings[self.class_name], student_embeddings))
            elif stu_emb_depth > slf_emb_depth:
                block_to_add_shape = (1, Student.embeddings[self.class_name].shape[1], stu_emb_depth - slf_emb_depth)
                block_to_add = np.full(block_to_add_shape, infinite_value)
                Student.embeddings[self.class_name] = np.vstack((Student.embeddings[self.class_name],
                                                                 np.dstack((student_embeddings, block_to_add))))

            elif stu_emb_depth < slf_emb_depth:
                block_to_add_shape = (
                    Student.embeddings[self.class_name].shape[0], Student.embeddings[self.class_name].shape[1],
                    slf_emb_depth - stu_emb_depth)
                block_to_add = np.full(block_to_add_shape, infinite_value)
                Student.embeddings[self.class_name] = np.vstack((np.dstack((Student.embeddings[self.class_name],
                                                                            block_to_add)), student_embeddings))

        else:
            Student.embeddings[self.class_name] = student_embeddings

    # Methods for logging
    def logging(self):
        """ Emotions and pose logging for frame for one student"""

        if self.face_coordinates is None:
            self._stud_is_on_frame.append(False)
            self._angle_logg.append([None] * 3)
        else:
            self._stud_is_on_frame.append(True)
            angles = []
            for key in self.angles:
                angles.append(self.angles[key])
            self._angle_logg.append(angles)

        self._emotion_logg.append(self.emotions)

    @classmethod
    def logging_of_group(cls, class_name):
        """ Emotions and pose logging for frame for whole students group. """
        cls._logging_time[class_name].append(time.time())

        for student_name in cls.group[class_name]:
            cls.group[class_name][student_name].logging()

    def get_student_logs(self):
        """ Getting total logs for one of the student """

        time_df = pd.DataFrame(list(Student._logging_time[self.class_name]), columns=['time'])
        emotion_df = pd.DataFrame(list(self._emotion_logg), columns=self.list_of_emotions)
        angle_df = pd.DataFrame(list(self._angle_logg), columns=['roll', 'pitch', 'yaw'])
        onframe_df = pd.DataFrame(list(self._stud_is_on_frame), columns=['is_on_frame'])

        pose_df = pd.concat([time_df, angle_df, onframe_df], axis=1)
        emotion_df = pd.concat([time_df, emotion_df], axis=1)

        return pose_df, emotion_df

    @classmethod
    def get_group_log(cls, class_name):
        """ Getting total log for the whole group in dataframe form """
        total_log = {}
        for student_name in cls.group[class_name]:
            total_log[student_name] = cls.group[class_name][student_name].get_student_logs()

        return total_log

    @staticmethod
    def comma_sep_str(list_of_str):
        """ Convert list of strings to one string separated by comma.
            It is used for recommendation strings forming.
        """
        output_str = ''
        for num_index, one_str in enumerate(list_of_str):
            if num_index != 0:
                output_str += ', '
            output_str += one_str
            if num_index == len(list_of_str) - 1:
                output_str += '.'
        return output_str

    @classmethod
    def get_recommendation(cls, class_name):
        """         Providing recomendations about lesson.
                    pace and emotional conditions of students.
                    At the output we have two lines with:
                         - recommendations
                         - list of absent students
                    Also color of this lines is noticed.  
        """

        time_constant = 30  # Negative emotions duration before recommendation is made
        crit_angle = 20  # Critical angle of head for student.
        lesson_start_time = 5
        negative_students = {}  # Dict of students with negative emotions
        distracted_students = {}  # Dict of distracted students (high head angle)
        absent_students = []  # List of absent students

        if cls.start_lesson[class_name]:
            lesson_time = cls._logging_time[class_name][-1] - cls._logging_time[class_name][0]
            if lesson_time > lesson_start_time:
                cls.start_lesson[class_name] = False
                cls.time_length_const[class_name] = len(cls._logging_time[class_name])
            attendance = (' ', (0, 255, 0))
            recommendation = ('Welcome to the lesson!', (0, 255, 0))
            return recommendation, attendance

        tail_length = int(cls.time_length_const[class_name] * time_constant /
                          (cls._logging_time[class_name][-1] -
                           cls._logging_time[class_name][-cls.time_length_const[class_name]]))

        for student_name in cls.group[class_name]:
            if student_name == 'lecturer':
                continue
            # Getting number of log lines for time_constant duration 
            person = Student.group[class_name][student_name]
            person.student_mark = (0, 255, 0)

            # Here we need to check that we have enough information
            frames_num_with_student = sum(np.array(person._stud_is_on_frame[-tail_length:]))
            if frames_num_with_student > 0.8 * tail_length:

                # Get names of students with negative emotions
                emotions_tail = np.array(person._emotion_logg[-tail_length:])[person._stud_is_on_frame[-tail_length:]]
                tmp = np.zeros_like(emotions_tail)
                tmp[np.arange(len(emotions_tail)), emotions_tail.argmax(1)] = 1

                column_sum = tmp.sum(axis=0)

                if 3.1*sum(np.take(column_sum, person.indexes_of_pos_emotions)) < sum(
                        np.take(column_sum, person.indexes_of_neg_emotions)):
                    negative_students[student_name] = sum(np.take(column_sum, person.indexes_of_neg_emotions))
                    person.student_mark = (255, 0, 0)

                # Get list of students with high head pose angle
                angle_tail = np.array(person._angle_logg[-tail_length:])
                angle_tail[angle_tail == None] = 0
                angle_tail = np.abs(angle_tail)

                if sum(angle_tail[:, 0] > crit_angle) > 0.5 * tail_length:
                    distracted_students[student_name] = sum(angle_tail[:, 0] > crit_angle) / tail_length
                    person.student_mark = (255, 0, 0)

            elif frames_num_with_student < 0.05 * tail_length:
                absent_students.append(student_name)


        # Creating lines to return
        absent_st_number = len(absent_students)
        if absent_st_number == 0:
            attendance = ('No one is absent in the class', (0, 255, 0))
        else:
            absent_st_str = cls.comma_sep_str(absent_students)
            attendance = ('Don\'t see {} student(s): {}'.format(absent_st_number, absent_st_str), (0, 255, 255))

        if len(negative_students) + len(distracted_students) == 0:
            recommendation = ('Lesson is going well', (0, 255, 0))
            return recommendation, attendance
        else:
            problem_students = len(set(list(negative_students.keys()) + list(distracted_students.keys())))
            total_number_of_students = len(cls.names[class_name])
            if problem_students >= 0.5 * total_number_of_students and total_number_of_students > 1:
                recommendation = ('Class needs to have a break', (0, 0, 255))
                return recommendation, attendance
            if len(list(distracted_students.keys())) > 0:
                max_distracted_student = list(distracted_students.keys())[0]
                recommendation = (' Pay attention to {} (distracted)'.format(max_distracted_student), (0, 255, 255))
                return recommendation, attendance
            if len(list(negative_students.keys())) > 0:
                max_negative_student = list(negative_students.keys())[0]
                recommendation = (' Pay attention to {} (bad mood)'.format(max_negative_student), (0, 255, 255))
                return recommendation, attendance
