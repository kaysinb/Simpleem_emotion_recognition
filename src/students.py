import cv2
import os
import numpy as np
import time
import pandas as pd
from collections import deque
from src.pose_estimation import Stabilizer
from src.pose_estimation import PoseEstimation
import itertools


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
        self._emotion_logg = []
        self._angle_logg = []
        self._logging_time = []
        self._stud_is_on_frame = []

        # Service properties
        self.list_of_emotions = list_of_emotions
        self.list_of_pos_emotions = ['neutral','happy']
        self.indexes_of_pos_emotions = []
        
        # Init of indexes of positive and negative emotions
        for item in self.list_of_pos_emotions:
            try:
                self.indexes_of_pos_emotions.append(self.list_of_emotions.index(item))
            except ValueError:
                print('Positive emotion {} is not in list of emotions'.format(item))
        #all_idexes = list(range(len(self.list_of_emotions)))
        self.indexes_of_neg_emotions = [index for index in range(len(self.list_of_emotions)) if index not in self.indexes_of_pos_emotions]
        self.frame_color = (0,0,255)

        self.pose_stabilizers = [Stabilizer(
            state_num=2,
            measure_num=1,
            cov_process=0.01,
            cov_measure=1) for _ in range(6)]

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
    def logging(self, time_of_log = None):
        """ Emotions and pose logging for frame for one student"""

        if self.face_coordinates is None:
            self._stud_is_on_frame.append(False)
            self._angle_logg.append([None] * 3)
        else:
            self._stud_is_on_frame.append(True)
            self._angle_logg.append(PoseEstimation.rot_params_rv(self._pose[0]))
        
        if time_of_log is None:
            self._logging_time.append(time.time())
        else:
            self._logging_time.append(time_of_log)
        
        self._emotion_logg.append(self.emotions)
    
    @classmethod
    def logging_of_group(cls):
        """ Emotions and pose logging for frame for whole students group. """
        mutual_log_time = time.time()

        for student_name in cls.group:
            cls.group[student_name].logging(time_of_log = mutual_log_time)

    def get_student_logs(self):
        """ Getting total logs for one of the student """

        time_df = pd.DataFrame(list(self._logging_time), columns=['time'])
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
                output_str +=' ,'
            output_str += one_str 
            if num_index == len(list_of_str)-1:
                output_str +='.'
        return output_str
    
    
    @classmethod
    def get_recomendation(cls):
        """         Providing recomendations about lesson.
                    pace and emotional conditions of students.
                    At the output we have two lines with:
                         - recomendations
                         - list of absent students
                    Also color of this lines is noticed.  
        """
        
        time_constant = 10 # Negative emotions duration before recomendation is made
        crit_angle = 20 # Critical angle of head for student.
        negative_students = {} # List of students with negative emotions
        distracted_students = {} # List of distracted students (high head angle)
        absent_students = [] # List of absent students

        for student_name in cls.group:
            # Getting number of log lines for time_constant duration 
            person = Student.group[student_name]
            time_tail_200 = np.array(person._logging_time[-200:])
            time_tail = time_tail_200 -  time_tail_200[-1]
            tail_length = np.sum((time_tail + time_constant)>0)
            
            # Here we need to check that we have enough information
            frames_num_with_student = sum(np.array(person._stud_is_on_frame[-tail_length:]))
            if frames_num_with_student > 0.8*tail_length :
                
                # Get names of students with negative emotions
                emotions_tail = np.array(person._emotion_logg[-tail_length:])[person._stud_is_on_frame[-tail_length:]]
                tmp = np.zeros_like(emotions_tail)
                tmp[np.arange(len(emotions_tail)), emotions_tail.argmax(1)] = 1
                
                colomn_sum = tmp.sum(axis = 0)
                
                if sum(np.take(colomn_sum, person.indexes_of_pos_emotions)) < sum(np.take(colomn_sum, person.indexes_of_neg_emotions)):
                    negative_students[student_name] = sum(np.take(colomn_sum, person.indexes_of_neg_emotions))
                
                # Get list of students with high head pose angle
                angle_tail = np.array(person._angle_logg[-tail_length:])
                angle_tail[angle_tail == None] = 0
                angle_tail = np.abs(angle_tail)
                
                if sum(angle_tail[:,1] > crit_angle)> 0.9*tail_length:
                    distracted_students[student_name] = sum(angle_tail[:,1] > crit_angle)/tail_length
            elif frames_num_with_student < 0.05*tail_length:
                    absent_students.append(student_name)
        
        # # This print is used to 
        # print(distracted_students)
        # print(negative_students)
        # print(absent_students)
        
        # Creating lines to return
        absent_st_number = len(absent_students)
        if absent_st_number == 0:
            attendance = ('All students in the classroom',(0,255,0))
        else:
            absent_st_str = cls.comma_sep_str(absent_students)
            attendance = ('{} students absent: {}'.format(absent_st_number,absent_st_str), (255,255,0))
        
        if len(negative_students)+len(distracted_students) == 0:
            recomendation = ('Lesson is going well', (0,255,0))
            return recomendation, attendance
        else:
            problem_students = len(set(list(negative_students.keys())+list(distracted_students.keys())))
            total_number_of_students = len(list(cls.group.keys()))
            if problem_students > 0.5 * total_number_of_students and total_number_of_students>1:
                recomendation = ('Class need to have a break',(0,255,0))
                return recomendation, attendance
            if len(list(negative_students.keys()))>0:
                max_negative_student = list(negative_students.keys())[0]
                recomendation = (' Pay attention to {} (bad mood)'.format(max_negative_student),(0,255,0))
                return recomendation, attendance
            if len(list(distracted_students.keys()))>0:
                max_distracted_student = list(distracted_students.keys())[0]
                recomendation = (' Pay attention to {} (distracted)'.format(max_distracted_student),(0,255,0))
                return recomendation, attendance
            
        

        
