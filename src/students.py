import cv2
import os
import numpy as np
import time
import pandas as pd
from collections import deque


class Student:
    """ """
    group = {}  # Dict of all students. Key - name. Value - object.
    names = []  # Names of students. The same as embeddings
    embeddings = None  # Embeddings matrix of all students. Order the same as students/
    _logging_time = []
    start_lesson = True
    time_length_const = None

    def __init__(self, photos, name, detector, list_of_emotions):

        # Unique properties of the student
        self.name = name
        self.face_image = None
        self._number_of_embeddings = 0  # Number of embeddings used for student. Is used to drop Inf values during unpacking
        student_embeddings = None  # All embeddings of student in np.array
        self.student_mark = (0, 255, 0)

        # Instant properties
        self.face_coordinates = None
        self.photos = photos
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

        for image in self.photos:
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
    def logging_of_group(cls):
        """ Emotions and pose logging for frame for whole students group. """
        cls._logging_time.append(time.time())

        for student_name in cls.group:
            cls.group[student_name].logging()

    def get_student_logs(self):
        """ Getting total logs for one of the student """

        time_df = pd.DataFrame(list(Student._logging_time), columns=['time'])
        emotion_df = pd.DataFrame(list(self._emotion_logg), columns=self.list_of_emotions)
        angle_df = pd.DataFrame(list(self._angle_logg), columns=['roll', 'pitch', 'yaw'])
        onframe_df = pd.DataFrame(list(self._stud_is_on_frame), columns=['is_on_frame'])

        pose_df = pd.concat([time_df, angle_df, onframe_df], axis=1)
        emotion_df = pd.concat([time_df, emotion_df], axis=1)

        return pose_df, emotion_df

    @classmethod
    def get_group_log(cls):
        """ Getting total log for the whole group in dataframe form """
        total_log = {}
        for student_name in cls.group:
            total_log[student_name] = cls.group[student_name].get_student_logs()

        return total_log

    @classmethod
    def comma_sep_str(cls, list_of_str):
        """ Convert list of strings to one string separated by comma.
            It is used for recomendation strings forming.
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
    def get_recommendation(cls):
        """         Providing recomendations about lesson.
                    pace and emotional conditions of students.
                    At the output we have two lines with:
                         - recommendations
                         - list of absent students
                    Also color of this lines is noticed.  
        """

        time_constant = 20  # Negative emotions duration before recommendation is made
        crit_angle = 20  # Critical angle of head for student.
        lesson_start_time = 5
        negative_students = {}  # Dict of students with negative emotions
        distracted_students = {}  # Dict of distracted students (high head angle)
        absent_students = []  # List of absent students

        if cls.start_lesson:
            lesson_time = cls._logging_time[-1] - cls._logging_time[0]
            if lesson_time > lesson_start_time:
                cls.start_lesson = False
                cls.time_length_const = len(cls._logging_time)
            attendance = (' ', (0, 255, 0))
            recommendation = ('Welcome to the lesson!', (0, 255, 0))
            return recommendation, attendance

        tail_length = int(cls.time_length_const * time_constant / (cls._logging_time[-1] -
                                                                   cls._logging_time[-cls.time_length_const]))

        for student_name in cls.group:
            if student_name == 'lecturer':
                continue
            # Getting number of log lines for time_constant duration 
            person = Student.group[student_name]
            person.student_mark = (0, 255, 0)

            # Here we need to check that we have enough information
            frames_num_with_student = sum(np.array(person._stud_is_on_frame[-tail_length:]))
            if frames_num_with_student > 0.8 * tail_length:

                # Get names of students with negative emotions
                emotions_tail = np.array(person._emotion_logg[-tail_length:])[person._stud_is_on_frame[-tail_length:]]
                tmp = np.zeros_like(emotions_tail)
                tmp[np.arange(len(emotions_tail)), emotions_tail.argmax(1)] = 1

                column_sum = tmp.sum(axis=0)

                if sum(np.take(column_sum, person.indexes_of_pos_emotions)) < sum(
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

        # # This print is used to 
        # print(distracted_students)
        # print(negative_students)
        # print(absent_students)

        # Creating lines to return
        absent_st_number = len(absent_students)
        if absent_st_number == 0:
            attendance = ('All students in the classroom', (0, 255, 0))
        else:
            absent_st_str = cls.comma_sep_str(absent_students)
            attendance = ('Don\'t see {} student(s): {}'.format(absent_st_number, absent_st_str), (0, 255, 255))

        if len(negative_students) + len(distracted_students) == 0:
            recommendation = ('Lesson is going well', (0, 255, 0))
            return recommendation, attendance
        else:
            problem_students = len(set(list(negative_students.keys()) + list(distracted_students.keys())))
            total_number_of_students = len(list(cls.group.keys()))
            if problem_students > 0.5 * total_number_of_students and total_number_of_students > 1:
                recommendation = ('Class need to have a break', (0, 0, 255))
                return recommendation, attendance
            if len(list(distracted_students.keys())) > 0:
                max_distracted_student = list(distracted_students.keys())[0]
                recommendation = (' Pay attention to {} (distracted)'.format(max_distracted_student), (0, 255, 255))
                return recommendation, attendance
            if len(list(negative_students.keys())) > 0:
                max_negative_student = list(negative_students.keys())[0]
                recommendation = (' Pay attention to {} (bad mood)'.format(max_negative_student), (0, 255, 255))
                return recommendation, attendance
