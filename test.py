import cv2
import os
import numpy as np
from src.faces_recognition import FacesRecognition
from src.emotion_recognition import EmotionsRecognition
from src.students import Student
from src.pose_estimation import PoseEstimation
from src.display_text import DisplayText, write_signature
from src.emotionsStatisticsGUI import show_statistic_window
from tqdm import tqdm


def main(photos_path, video_path=0, show_video=False, save_video=False):
    faces_recognizer = FacesRecognition(resize=0.5, max_face_tilt=10)
    emotions_recognizer = EmotionsRecognition()

    students_dirs = [f for f in os.scandir(photos_path) if f.is_dir()]  # Folders for every student in path
    emotions_list = emotions_recognizer.emotions

    for one_student in students_dirs:
        student_photos = []
        for file_name in os.listdir(one_student.path):
            print('Getting photo .... : ' + one_student.path + '/' + file_name)
            image = cv2.imread(one_student.path + '/' + file_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            student_photos.append(image)
        Student(student_photos, one_student.name, faces_recognizer, emotions_list)

    # Create a VideoCapture object and read from input file
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    out_fps = 5
    frame_step = fps//out_fps

    if video_path == 0:
        cap.set(3, 1280)
        cap.set(4, 720)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")

    frame = cap.read()[1]
    pose_estimator = PoseEstimation()
    text_displayer = DisplayText(frame.shape)

    if save_video:
        # Create a VideoCapture object and read from input file
        fourcc = cv2.VideoWriter_fourcc('F', 'M', 'P', '4')
        out_video = cv2.VideoWriter('./testing/output_video.mp4', fourcc, fps//frame_step, (text_displayer.full_frame_shape[1],
                                                                                    text_displayer.full_frame_shape[0]))

    firs_frame = 5000  # first frame
    last_frame = None

    if last_frame is None:
        if video_path != 0:
            last_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            last_frame = np.inf

    cap.set(1, firs_frame)  # read from firs_frame

    stride = 1  # we will recognize only every 'stride' frame for saving time
    i = 0  # counter
    text = 'hi'

    for _ in tqdm(range(1000)):
    # while cap.isOpened() and firs_frame + i < last_frame:
        flag = False
        if i % stride == 0:
            flag = True

        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret and i % frame_step == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if flag:
                faces_recognizer(frame, Student)
                emotions_recognizer(Student)
                frame = pose_estimator(frame, Student)
                    
                Student.logging_of_group()

            write_signature(frame, Student)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            recommendation, attendance = Student.get_recommendation()
            full_frame = text_displayer.show_text(frame, recommendation, attendance)

            if save_video:
                out_video.write(full_frame)

            if show_video:
                cv2.imshow('meeting', cv2.resize(full_frame, (full_frame.shape[1]//2, full_frame.shape[0]//2)))

            # Press Q on keyboard to  exit
            if cv2.waitKey(2) & 0xFF == ord('q'):
                break
        i += 1

    cap.release()
    cv2.destroyAllWindows()
    
    show_statistic_window(Student.get_group_log())


if __name__ == '__main__':
    main(video_path=0, photos_path='../../er_test/photos', show_video=True, save_video=True)
    # main(video_path='../../er_test/test_video.mp4', photos_path='../../er_test/photos', show_video=True, save_video=True)
    # main(video_path='../../er_test/test_v.mp4', photos_path='../../er_test/photos_new', show_video=True, save_video=True)


