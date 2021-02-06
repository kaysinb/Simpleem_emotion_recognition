import cv2
import os
import numpy as np
from src.faces_recognition import FacesRecognition
from src.emotion_recognition import EmotionsRecognition
from src.students import Student
from src.pose_estimation import PoseEstimation


def main(photos_path, video_path=0, show_video=False, save_video=False):
    faces_recognizer = FacesRecognition(resize=0.5, max_face_tilt=10)
    emotions_recognizer = EmotionsRecognition()

    students_dirs = [f for f in os.scandir(photos_path) if f.is_dir()]  # Folders for every student in path
    emotions_list = emotions_recognizer.emotions

    for one_student in students_dirs:
        Student(one_student.path, one_student.name, faces_recognizer, emotions_list)

    # Create a VideoCapture object and read from input file
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_step = fps//25

    if video_path == 0:
        cap.set(3, 1280)
        cap.set(4, 720)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")

    frame = cap.read()[1]
    frame_height, frame_width = frame.shape[:2]
    pose_estimator = PoseEstimation(img_size=frame.shape[0:2])

    if save_video:
        # Create a VideoCapture object and read from input file
        fourcc = cv2.VideoWriter_fourcc('F', 'M', 'P', '4')
        out_video = cv2.VideoWriter('./output_video.mp4', fourcc, fps//frame_step, (frame_width, frame_height))

    firs_frame = 0  # first frame
    last_frame = None

    if last_frame is None:
        if video_path != 0:
            last_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            last_frame = np.inf

    cap.set(1, firs_frame)  # read from firs_frame

    stride = 1  # we will recognize only every 'stride' frame for saving time
    i = 0  # counter

    while cap.isOpened() and firs_frame + i < last_frame:
        flag = False
        if i % stride == 0:
            flag = True

        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if ret and i % frame_step == 0:
            if flag:

                faces_recognizer(frame, Student)
                emotions_recognizer(Student)

                for student_name in Student.group:
                    one_student = Student.group[student_name]

                    if one_student.landmarks is not None:
                        m = one_student.landmarks
                        one_student.pose = pose_estimator.solve_pose(m)

                    one_student.logging()

            for student_name in Student.group:
                if Student.group[student_name].face_coordinates is not None:
                    one_student = Student.group[student_name]

                    pose = one_student.pose

                    pose_estimator.draw_annotation_box(frame, pose[0], pose[1], color=(255, 128, 128))

                    y = int(one_student.landmarks[0][1])
                    x = int(2 * one_student.landmarks[1][0] - one_student.landmarks[0][0])

                    rot_vect = pose_estimator.rot_params_rv(np.atleast_2d(pose[0]).T)

                    cv2.putText(frame, str(rot_vect[0])[:4], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1,
                                255)
                    cv2.putText(frame, str(rot_vect[1])[:4], (x, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0),
                                1,
                                255)
                    cv2.putText(frame, str(rot_vect[2])[:4], (x, y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0),
                                1,
                                255)

                    y_text = int(
                        one_student.landmarks[0][1] - (one_student.landmarks[3][1] - one_student.landmarks[0][1]))
                    x_text = int(
                        one_student.landmarks[0][0] - (one_student.landmarks[1][0] - one_student.landmarks[0][0]))

                    emotion = emotions_list[np.argmax(one_student.emotions)]
                    cv2.putText(frame, student_name + ' is ' + emotion, (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX , 0.8, (0,0,255), 1, 255)

                    for point in one_student.landmarks:
                        cv2.circle(frame, (int(point[0]), int(point[1])), 3, (0, 0, 255), -1)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if save_video:
                out_video.write(frame)

            if show_video:
                cv2.imshow('meeting', frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        i += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(video_path=0, photos_path='../../er_test/photos', show_video=True, save_video=True)

