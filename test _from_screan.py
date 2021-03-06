import cv2
import os
import numpy as np
from src.emotion_recognition import EmotionsRecognition
# from testing.new_students import Student
# from testing.new_faces_recognition import FacesRecognition
from src.students import Student
from src.faces_recognition import FacesRecognition
from src.pose_estimation import PoseEstimation
from src.display_text import DisplayText, write_caption
from tqdm import tqdm
from mss import mss

from src.emotionsStatisticsGUI import get_real_time_stat_window


def main(photos_path, show_video=False, save_video=False, real_time_stat=False):
    faces_recognizer = FacesRecognition(resize=0.9, max_face_tilt=10)
    emotions_recognizer = EmotionsRecognition()
    class_name = 'first'
    students_dirs = [f for f in os.scandir(photos_path) if f.is_dir()]  # Folders for every student in path
    Student.list_of_emotions = emotions_recognizer.emotions
    Student.detector = faces_recognizer
    Student.group_initialization(class_name)
    Student.recognize_all_students[class_name] = False


    for one_student in students_dirs:
        student_photos = []
        for file_name in os.listdir(one_student.path):
            print('Getting photo .... : ' + one_student.path + '/' + file_name)
            image = cv2.imread(one_student.path + '/' + file_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            student_photos.append(image)
        Student(student_photos, one_student.name, class_name)

    # Create a VideoCapture object and read from input file
    bounding_box = {'top': 10, 'left': 100, 'width': 1800, 'height': 1020}
    sct = mss()
    sct_frame = sct.grab(bounding_box)
    frame = np.array(sct_frame)[:,:,:3].copy()
    fps = 15
    out_fps = 15
    frame_step = fps//out_fps
    pose_estimator = PoseEstimation()
    text_displayer = DisplayText(frame.shape)

    if save_video:
        # Create a VideoCapture object and read from input file
        fourcc = cv2.VideoWriter_fourcc('F', 'M', 'P', '4')
        out_video = cv2.VideoWriter('./testing/output_video_5.mp4', fourcc, fps//frame_step, (text_displayer.full_frame_shape[1],
                                                                                    text_displayer.full_frame_shape[0]))


    stride = 1  # we will recognize only every 'stride' frame for saving time
    i = 0  # counter
    text = 'hi'

    for _ in tqdm(range(10000)):

        flag = False
        if i % stride == 0:
            flag = True

        # Capture frame-by-frame
        sct_frame = sct.grab(bounding_box)
        frame = np.array(sct_frame)[:, :, :3].copy()
        if  i % frame_step == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if flag:
                faces_recognizer(frame, Student, class_name)
                emotions_recognizer(Student, class_name)
                frame = pose_estimator(frame, Student, class_name)
                    
                Student.logging_of_group(class_name)
                ## This IF block is for real statistic window initialization and
                ## sending data to it.
                if real_time_stat == True and i == 0:
                    queue_for_logs,stat_window = get_real_time_stat_window(Student.get_frame_log(class_name))
                elif real_time_stat and i>0:
                    queue_for_logs.put(Student.get_frame_log(class_name))

            write_caption(frame, Student, class_name)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            recommendation, attendance = Student.get_recommendation(class_name)
            full_frame = text_displayer.show_text(frame, recommendation, attendance)

            if save_video:
                out_video.write(full_frame)

            if show_video:
                scale = 2
                cv2.imshow('meeting', cv2.resize(full_frame, (full_frame.shape[1]//scale, full_frame.shape[0]//scale)))

            # Press Q on keyboard to  exit
            if cv2.waitKey(2) & 0xFF == ord('q'):
                break
        i += 1

    cv2.destroyAllWindows()
    
    if real_time_stat == False:
        window_queue,window_process = get_real_time_stat_window(Student.get_group_log(class_name))



if __name__ == '__main__':
    #main(video_path=0, photos_path='../../er_test/photos_4', show_video=True, save_video=True)
    # main(video_path='../../er_test/innopolismeeting.mp4', photos_path='../../er_test/photos_zuzan', show_video=True, save_video=True)
    # main(video_path='../../er_test/test_video.mp4', photos_path='../../er_test/photos_4', show_video=True, save_video=True)
    # main(video_path='../../test_simpleem.mp4', photos_path='../../er_test/photos_3', show_video=True, save_video=True)
    main(photos_path='../../er_test/photos_4', show_video=True, save_video=False, real_time_stat=True)


