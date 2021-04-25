import torch
import numpy as np
import cv2
from models.mtcnn import MTCNN
from models.inception_resnet_v1 import InceptionResnetV1
from PIL import Image


class FacesRecognition:
    """
    Detection and recognition faces in the image.
    Keyword arguments:`
        resize {float}: Fractional frame scaling. [default: {1}]"""

    def __init__(self, resize=1, max_face_tilt=None):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Running on device: {}'.format(self.device))
        self.resize = resize
        self.mtcnn = MTCNN(image_size=160, keep_all=True, device=self.device, post_process=False)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.max_face_tilt = max_face_tilt
        self.to_m1p1 = lambda x: (x - 127.5) / 128
        self.from_m1p1 = lambda x: x * 128 + 127.5
        self.name_for_new_person = 'Person_1'

    def __call__(self, frame, students, class_name):
        self.faces_recognition(frame, students, class_name)

    def get_initial_embedding(self, photo):
        face_coordinates, landmarks = self.faces_detection(photo)
        try:
            croped_face = self.get_batch_cropped_faces(photo, face_coordinates, landmarks)
        except TypeError:
            print('Did not find a face')
            return None

        embedding = self.get_embeddings(croped_face)
        if len(embedding) > 1:
            print('Found many faces')

        return embedding[0]

    @staticmethod
    def boundary_conditions(x_l, x_r, y_t, y_b, frame_shape):
        y_max = frame_shape[0] - 1
        x_max = frame_shape[1] - 1

        x_l = x_l if x_l > 0 else 0
        x_l = x_l if x_l < x_max else x_max

        x_r = x_r if x_r > 0 else 0
        x_r = x_r if x_r < x_max else x_max

        y_t = y_t if y_t > 0 else 0
        y_t = y_t if y_t < y_max else y_max

        y_b = y_b if y_b > 0 else 0
        y_b = y_b if y_b < y_max else y_max

        return x_l, x_r, y_t, y_b

    def get_batch_cropped_faces(self, frame, faces_coordinates, landmarks):
        """crop to size 160X160 and put it in one batch """
        batch_cropped_faces = []
        for face_coordinates, points in zip(faces_coordinates, landmarks):
            face = None
            if self.max_face_tilt is not None:
                points = points.astype(int)
                x1, y1 = points[0][0], points[0][1]
                x2, y2 = points[1][0], points[1][1]
                x3, y3 = points[3][0], points[3][1]
                tilt = int(np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi)
                if np.abs(tilt) > self.max_face_tilt:
                    width = int(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
                    height = int(np.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2))

                    x_l = x1 - width - height
                    x_r = x2 + width + height
                    y_t = y1 - width - height
                    y_b = y3 + width + height

                    x_l, x_r, y_t, y_b = FacesRecognition.boundary_conditions(x_l, x_r, y_t, y_b, frame.shape)

                    center_x = width + height
                    center_y = width + height

                    image = Image.fromarray(frame[y_t:y_b, x_l:x_r, :])
                    im_rot = image.rotate(tilt, center=(center_x, center_y), expand=True).resize((256, 256))
                    face = self.mtcnn(np.array(im_rot))
                    if face is not None:
                        face = np.array(face[0], dtype='uint8').transpose((1, 2, 0))

            if face is None:
                y_t = face_coordinates[1]
                y_b = face_coordinates[3]
                x_l = face_coordinates[0]
                x_r = face_coordinates[2]

                x_l, x_r, y_t, y_b = FacesRecognition.boundary_conditions(x_l, x_r, y_t, y_b, frame.shape)

                face = frame[y_t:y_b, x_l:x_r, :]
                face = cv2.resize(face, (160, 160))

            batch_cropped_faces.append(torch.FloatTensor(self.to_m1p1(face.transpose((2, 0, 1)))))
        return batch_cropped_faces

    def faces_detection(self, frame):
        processed_frame = frame.copy()
        if self.resize != 1:
            processed_frame = cv2.resize(processed_frame, (int(frame.shape[1] * self.resize),
                                                           int(frame.shape[0] * self.resize)))

        faces_coordinates, probs, landmarks = self.mtcnn.detect(processed_frame, landmarks=True)

        if faces_coordinates is not None:
            faces_coordinates = faces_coordinates[probs > 0.95]
            if self.resize != 1:
                faces_coordinates /= self.resize
                landmarks /= self.resize

            faces_coordinates = faces_coordinates.astype(int)

            if len(faces_coordinates) == 0:
                faces_coordinates = None
        return faces_coordinates, landmarks

    def get_embeddings(self, batch_cropped_faces):
        batch = torch.stack(batch_cropped_faces).to(self.device)
        embeddings = self.resnet(batch).detach().cpu()
        return embeddings.numpy()

    def compare_faces(self, face_embeddings_to_check, students_embeddings, students_names, tolerance=0.9):

        recognized_embeddings = ['unknown'] * len(face_embeddings_to_check)
        norm_matrix = np.zeros((len(face_embeddings_to_check), students_embeddings.shape[0],
                                students_embeddings.shape[2])) + tolerance * 2

        for i, embedding in enumerate(face_embeddings_to_check):
            embedding = np.expand_dims(embedding, axis=1)
            norm_matrix[i] = np.linalg.norm(students_embeddings - embedding, axis=1)

        for i in range(min(norm_matrix.shape[0], norm_matrix.shape[1])):
            min_dist = np.min(norm_matrix)

            if min_dist < tolerance:
                student_index = np.argwhere(norm_matrix == min_dist)[0]
                recognized_embeddings[student_index[0]] = students_names[student_index[1]]
                norm_matrix[student_index[0]] = tolerance * 2
                norm_matrix[:, student_index[1]] = tolerance * 2

        return recognized_embeddings

    @staticmethod
    def compare_coordinates(possible_name, center, student, class_name, frame):
        center_x, center_y = center
        old_fc = student.group[class_name][possible_name].face_coordinates
        left = old_fc[0]
        right = old_fc[2]
        top = old_fc[1]
        bottom = old_fc[3]
        delta_x = int((right - left)*0.5)
        delta_y = int((bottom - top)*0.5)

        if right + delta_x > center_x > left - delta_x and bottom + delta_y > center_y > top - delta_y:
            return True
        else:
            return False

    def faces_recognition(self, frame, student, class_name):
        faces_coordinates, landmarks = self.faces_detection(frame)

        if faces_coordinates is None:
            recognized_students = []
        else:
            batch_cropped_faces = self.get_batch_cropped_faces(frame,
                                                               faces_coordinates, landmarks)
            embeddings = self.get_embeddings(batch_cropped_faces)

            if class_name in student.embeddings:
                recognized_students = self.compare_faces(embeddings,
                                                         student.embeddings[class_name],
                                                         student.names[class_name],
                                                         tolerance=0.99)
            else:
                recognized_students = ['unknown'] * len(embeddings)

        for name in student.group[class_name]:
            if name in recognized_students:
                student_index = recognized_students.index(name)
                student.group[class_name][name].face_coordinates = faces_coordinates[student_index]
                student.group[class_name][name].face_image = np.array(
                    self.from_m1p1(batch_cropped_faces[student_index]),
                    dtype='uint8').transpose((1, 2, 0))

        if student.recognize_all_students[class_name]:
            for i, person in enumerate(recognized_students):
                if person == 'unknown':
                    found_this_person = False
                    new_fc = faces_coordinates[i]
                    center_x = (new_fc[2] + new_fc[0]) / 2
                    center_y = (new_fc[3] + new_fc[1]) / 2

                    delta = int((new_fc[2] - new_fc[0])/4)
                    x_l, y_t, x_r, y_b = new_fc
                    x_l -= delta
                    x_r += delta
                    y_t -= delta
                    y_b += delta
                    x_l, x_r, y_t, y_b = FacesRecognition.boundary_conditions(x_l, x_r, y_t, y_b, frame.shape)
                    photo = [frame[y_t:y_b, x_l:x_r, :]]
                    for name in student.names[class_name]:
                        if student.group[class_name][name].face_coordinates is not None:

                            if FacesRecognition.compare_coordinates(name, (center_x, center_y), student, class_name, frame):
                                if name not in recognized_students:
                                    student.group[class_name][name].face_coordinates = faces_coordinates[i]
                                    student.group[class_name][name].face_image = np.array(
                                        self.from_m1p1(batch_cropped_faces[i]), dtype='uint8').transpose((1, 2, 0))
                                    recognized_students[i] = name

                                    if student.group[class_name][name].number_of_embeddings < 5:
                                        student.group[class_name][name].add_new_embeddings(photo)

                                    found_this_person = True
                                    break

                    if not found_this_person:
                        self.add_new_person(photo, class_name, student, faces_coordinates[i], batch_cropped_faces[i])

        for name in student.group[class_name]:
            if name not in recognized_students:
                student.group[class_name][name].face_coordinates = None
                student.group[class_name][name].face_image = None
                student.group[class_name][name].landmarks = None

    def add_new_person(self, photo, class_name, student, faces_coordinates, cropped_face):
        while self.name_for_new_person in student.names[class_name]:
            sub_name = self.name_for_new_person.split('_')
            self.name_for_new_person = sub_name[0] + '_' + str(int(sub_name[1]) + 1)
        student(photo, self.name_for_new_person, class_name)
        if self.name_for_new_person in student.group[class_name]:
            student.group[class_name][self.name_for_new_person].face_coordinates = faces_coordinates
            student.group[class_name][self.name_for_new_person].face_image = np.array(
                self.from_m1p1(cropped_face), dtype='uint8').transpose((1, 2, 0))
