import numpy as np
import time
import pandas as pd
import copy


def generate_new_name(name):
    sub_name = name.split('_')
    new_name = sub_name[0] + '_' + str(int(sub_name[1]) + 1)
    return new_name


class Student:
    """ """
    group = {}  # Dict of dicts of all students.  Key_0 - class_name, Key_1 - name, Value - object.
    names = {}  # Names - dict of students. The same as embeddings
    embeddings = {}  # Embeddings - dict of matrix of all students. Order the same as students/
    detector = None
    max_num_new_embd = 10
    _logging_time = {}  # Dict of arrays with logging time
    start_lesson = {}  # Dict of flags
    time_length_const = {}  # Dict of time length consts
    list_of_emotions = None
    recognize_all_students = {}
    free_name = {}

    def __init__(self, photos, class_name, name=None):

        # Unique properties of the student
        if name is None:
            while Student.free_name[class_name] in Student.names[class_name]:
                Student.free_name[class_name] = generate_new_name(Student.free_name[class_name])
            self.name = Student.free_name[class_name]
        else:
            self.name = name
        self.class_name = class_name
        self.face_image = None
        self.number_of_embeddings = 0  # Number of embeddings used for student. Is used to drop Inf values during unpacking
        self.student_embeddings = None  # All embeddings of student in np.array
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

        # this parameters we use for drawing pose boxes
        self.P = None
        self.ver = None

        # Service properties
        self.list_of_pos_emotions = ['neutral', 'happy']
        self.indexes_of_pos_emotions = []

        # Init of indexes of positive and negative emotions
        for item in self.list_of_pos_emotions:
            try:
                self.indexes_of_pos_emotions.append(Student.list_of_emotions.index(item))
            except ValueError:
                print('Positive emotion {} is not in list of emotions'.format(item))

        self.indexes_of_neg_emotions = [index for index in range(len(Student.list_of_emotions)) if
                                        index not in self.indexes_of_pos_emotions]
        self.frame_color = (0, 0, 255)

        self.photos = photos
        embeddings = []
        for image in self.photos:
            new_embedding = Student.detector.get_initial_embedding(image)
            if new_embedding is not None:
                embeddings.append(new_embedding)
            else:
                print('Bad photo for initialization!')
        self.add_new_embeddings(embeddings)

    @classmethod
    def group_initialization(cls, class_name):
        cls.names[class_name] = []
        cls.start_lesson[class_name] = True
        cls.group[class_name] = {}
        cls._logging_time[class_name] = []
        cls.free_name[class_name] = 'Person_1'

    def add_new_embeddings(self, embeddings, add_to_existing=False):
        # add embeddings for the student
        for new_embedding in embeddings:
            self.number_of_embeddings += 1
            if self.student_embeddings is not None:
                self.student_embeddings = np.dstack((self.student_embeddings, new_embedding))
            else:
                self.student_embeddings = new_embedding
                self.student_embeddings = np.expand_dims(self.student_embeddings, axis=0)
                self.student_embeddings = np.expand_dims(self.student_embeddings, axis=2)

        if self.student_embeddings is not None:
            if self.name not in Student.names[self.class_name]:
                Student.names[self.class_name].append(self.name)
                Student.group[self.class_name][self.name] = self

            # Adding embedding matrix of student to common matrix

            infinite_value = np.inf

            if self.class_name in Student.embeddings:

                if add_to_existing:
                    index = np.where(np.array(Student.names[self.class_name]) == self.name)[0][0]
                    Student.embeddings[self.class_name] = np.delete(Student.embeddings[self.class_name], index, axis=0)
                    Student.names[self.class_name] = np.delete(Student.names[self.class_name], index, axis=0).tolist()
                    Student.names[self.class_name].append(self.name)

                stu_emb_depth = Student.embeddings[self.class_name].shape[2]  # Depth of all students embedding
                slf_emb_depth = self.student_embeddings.shape[2]  # Depth of student embedding

                # Concatenation of arrays with notequal shape
                if stu_emb_depth == slf_emb_depth:
                    Student.embeddings[self.class_name] = np.vstack(
                        (Student.embeddings[self.class_name], self.student_embeddings))
                elif stu_emb_depth > slf_emb_depth:
                    block_to_add_shape = (
                        1, Student.embeddings[self.class_name].shape[1], stu_emb_depth - slf_emb_depth)
                    block_to_add = np.full(block_to_add_shape, infinite_value)
                    Student.embeddings[self.class_name] = np.vstack((Student.embeddings[self.class_name],
                                                                     np.dstack(
                                                                         (self.student_embeddings, block_to_add))))

                elif stu_emb_depth < slf_emb_depth:
                    block_to_add_shape = (
                        Student.embeddings[self.class_name].shape[0], Student.embeddings[self.class_name].shape[1],
                        slf_emb_depth - stu_emb_depth)
                    block_to_add = np.full(block_to_add_shape, infinite_value)
                    Student.embeddings[self.class_name] = np.vstack((np.dstack((Student.embeddings[self.class_name],
                                                                                block_to_add)),
                                                                     self.student_embeddings))

            else:
                Student.embeddings[self.class_name] = self.student_embeddings

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

    def get_student_logs(self, frame_number='all'):
        """
            Getting total logs for one of the student for the whole lesson or
            only last frame.
        """
        if frame_number == 'all':
            log_slice = slice(0, None)
        elif frame_number == 'last':
            log_slice = slice(-1, None)
        else:
            print('Invalid frame_number function argument.')
            raise ValueError

        time_df = pd.DataFrame(list(Student._logging_time[self.class_name][log_slice]), columns=['time'])
        emotion_df = pd.DataFrame(list(self._emotion_logg[log_slice]), columns=self.list_of_emotions)
        angle_df = pd.DataFrame(list(self._angle_logg[log_slice]), columns=['roll', 'pitch', 'yaw'])
        onframe_df = pd.DataFrame(list(self._stud_is_on_frame[log_slice]), columns=['is_on_frame'])

        pose_df = pd.concat([time_df, angle_df, onframe_df], axis=1)
        emotion_df = pd.concat([time_df, emotion_df], axis=1)

        return pose_df, emotion_df

    @classmethod
    def get_group_log(cls, class_name):
        """ Getting total log for the whole group in dataframe form """
        total_log = {}
        for student_name in cls.group[class_name]:
            total_log[student_name] = cls.group[class_name][student_name].get_student_logs(frame_number='all')

        return total_log

    @classmethod
    def get_frame_log(cls, class_name):
        """ Getting last frame log for the whole group in dataframe form """
        total_log = {}
        for student_name in cls.group[class_name]:
            total_log[student_name] = cls.group[class_name][student_name].get_student_logs(frame_number='last')

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

        time_constant = 20  # Negative emotions duration before recommendation is made
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

                if 3.1 * sum(np.take(column_sum, person.indexes_of_pos_emotions)) < sum(
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


class PotentialStudent:
    queue_of_potential_students = {}
    potential_students_names = {}
    confidence_number = 5  # if number of embeddings > confidence_number we add the PotentialStudent to Students

    def __init__(self, photo, embedding, class_name):
        self.embeddings = [embedding]
        self.photos = [photo]
        self.name = PotentialStudent.get_name(class_name)
        self.class_name = class_name
        PotentialStudent.queue_of_potential_students[self.class_name][self.name] = self
        PotentialStudent.potential_students_names[class_name].append(self.name)

    @classmethod
    def get_name(cls, class_name):
        name = 'potential_1'
        while name in cls.potential_students_names[class_name]:
            name = generate_new_name(name)
        return name

    @classmethod
    def potential_group_initialization(cls, class_name):
        cls.potential_students_names[class_name] = []
        cls.queue_of_potential_students[class_name] = {}

    @classmethod
    def search_among_unknown(cls, photos, new_embeddings, class_name):
        found_persons = []
        found_potential_student = [False] * len(photos)
        for i, embedding in enumerate(new_embeddings):
            for name in cls.potential_students_names[class_name]:
                potential_student = cls.queue_of_potential_students[class_name][name]
                if Student.detector.compare_embeddings(potential_student.embeddings, embedding):
                    found_potential_student[i] = True
                    found_persons.append(name)
                    if len(potential_student.embeddings) < cls.confidence_number:
                        potential_student.embeddings.append(embedding)
                        potential_student.photos.append(photos[i])
                    else:
                        Student(potential_student.photos, class_name)
                        cls.remove_from_queue(class_name, name)

        names = copy.copy(cls.potential_students_names[class_name])
        for name in names:
            if name not in found_persons:
                cls.remove_from_queue(class_name, name)

        for i, embedding in enumerate(new_embeddings):
            if not found_potential_student[i]:
                PotentialStudent(photos[i], embedding, class_name)

    @classmethod
    def remove_from_queue(cls, class_name, name):
        cls.potential_students_names[class_name].remove(name)
        cls.queue_of_potential_students[class_name].pop(name, None)