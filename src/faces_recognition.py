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

    def __call__(self, frame, students):
        self.faces_recognition(frame, students)

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

            batch_cropped_faces.append(torch.FloatTensor(face.transpose((2, 0, 1)) / 255.0))
        return batch_cropped_faces

    def faces_detection(self, frame):
        processed_frame = frame.copy()
        if self.resize != 1:
            processed_frame = cv2.resize(processed_frame, (int(frame.shape[1] * self.resize),
                                                           int(frame.shape[0] * self.resize)))

        faces_coordinates, probs, landmarks = self.mtcnn.detect(processed_frame, landmarks=True)
        if faces_coordinates is not None:
            if self.resize != 1:
                faces_coordinates /= self.resize
                landmarks /= self.resize

            faces_coordinates = faces_coordinates.astype(int)
        return faces_coordinates, landmarks

    def get_embeddings(self, batch_cropped_faces):
        batch = torch.stack(batch_cropped_faces).to(self.device)
        embeddings = self.resnet(batch).detach().cpu()
        return embeddings.numpy()

    @staticmethod
    def compare_faces(face_embeddings_to_check, students_embeddings, students_names, tolerance=0.9):

        recognized_students = []
        for embedding in face_embeddings_to_check:
            embedding = np.expand_dims(embedding, axis=1)
            norm_matrix = np.linalg.norm(students_embeddings - embedding, axis=1)
            min_dist = np.min(norm_matrix)
            if min_dist < tolerance:
                name_index = np.argwhere(norm_matrix == min_dist)[0, 0]
                recognized_students.append(students_names[name_index])
        return recognized_students

    def faces_recognition(self, frame, student):
        faces_coordinates, landmarks = self.faces_detection(frame)

        if faces_coordinates is None:
            recognized_students = []
        else:
            batch_cropped_faces = self.get_batch_cropped_faces(frame,
                                                               faces_coordinates, landmarks)
            embeddings = self.get_embeddings(batch_cropped_faces)
            recognized_students = self.compare_faces(embeddings,
                                                     student.embeddings,
                                                     student.names,
                                                     tolerance=0.8)

        for name in student.group:
            if name in recognized_students:
                student_index = recognized_students.index(name)
                student.group[name].face_coordinates = faces_coordinates[student_index]
                student.group[name].face_image = np.array(batch_cropped_faces[student_index] * 255,
                                                          dtype='uint8').transpose((1, 2, 0))
                student.group[name].landmarks = landmarks[student_index]
            else:
                student.group[name].face_coordinates = None
                student.group[name].face_image = None
                student.group[name].landmarks = None
