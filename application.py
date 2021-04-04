from flask import Flask, request, jsonify, send_file
import numpy as np
from src.faces_recognition import FacesRecognition
from src.emotion_recognition import EmotionsRecognition
from src.students import Student
from src.pose_estimation import PoseEstimation
from src.display_text import write_caption
from PIL import Image
from collections import deque
import io
import cv2
import time


application = Flask(__name__)

faces_recognizer = FacesRecognition(resize=0.5, max_face_tilt=10)
emotions_recognizer = EmotionsRecognition()
emotions_list = emotions_recognizer.emotions
pose_estimator = PoseEstimation()
student_initialization_dict = {}
class_image = {}
process_time = {'fps': None, 'time_deque': deque([]), 'num_frames': 5}


@application.route('/')
def index():
    return 'Simpleem_app'

@application.route('/initialization', methods=['POST', 'GET'])
def initialization():

    if request.method == 'POST':
        name = request.files.get('name').read().decode('utf8')
        class_name = request.files.get('class_name').read().decode('utf8')
        file = request.files.get('file')
        img_bytes = file.read()
        image = np.array(Image.open(io.BytesIO(img_bytes)), dtype='uint8')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if class_name not in student_initialization_dict:
            student_initialization_dict[class_name] = {}

        if name not in student_initialization_dict[class_name]:
            student_initialization_dict[class_name][name] = [image]
        else:
            student_initialization_dict[class_name][name].append(image)

        return f'{name}\'s photo is added'

    if request.method == 'GET':
        class_name = request.files.get('class_name').read().decode('utf8')
        for name in student_initialization_dict[class_name]:
            Student(student_initialization_dict[class_name][name], name, class_name, faces_recognizer, emotions_list)

        return f'{class_name} class has been initialized'


@application.route('/run', methods=['POST'])
def start_lesson():
    file = request.files.get('file')
    img_bytes = file.read()
    frame = np.array(Image.open(io.BytesIO(img_bytes)), dtype='uint8')
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    class_name = request.files.get('class_name').read().decode('utf8')

    time_start = time.time()

    faces_recognizer(frame, Student, class_name)
    emotions_recognizer(Student, class_name)
    Student.logging_of_group(class_name)
    recommendation, attendance = Student.get_recommendation(class_name)
    frame = pose_estimator(frame, Student, class_name)
    write_caption(frame, Student, class_name)

    time_end = time.time()

    if process_time['fps'] is None:
        process_time['time_deque'] = deque([time_end - time_start]*process_time['num_frames'])
    else:
        process_time['time_deque'].popleft()
        process_time['time_deque'].append(time_end - time_start)

    process_time['fps'] = process_time['num_frames']/np.sum(process_time['time_deque'])

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    class_image[class_name] = frame

    students_condition = {}
    for name in Student.group[class_name]:
        students_condition[name] = {}
        student = Student.group[class_name][name]
        emotion = student.current_emotion
        angles = student.angles
        students_condition[name]['emotion'] = emotion
        students_condition[name]['angles'] = angles

    return jsonify({'recommendation': recommendation[0], 'attendance': attendance[0],
                    'students_condition': students_condition})


@application.route('/end_lesson', methods=['GET'])
def end_lesson():
    class_name = request.files.get('class_name').read().decode('utf8')
    student_initialization_dict.pop(class_name, None)
    Student.group.pop(class_name, None)
    Student.names.pop(class_name, None)
    Student.embeddings.pop(class_name, None)
    Student._logging_time.pop(class_name, None)
    Student.start_lesson.pop(class_name, None)
    Student.time_length_const.pop(class_name, None)
    class_image.pop(class_name, None)
    return f'End of the lesson for {class_name} class'

@application.route('/classes', methods=['GET'])
def classes():

    return jsonify({'classes': list(Student.group.keys())})


@application.route('/get_image', methods=['GET'])
def get_image():
    class_name = request.files.get('class_name').read().decode('utf8')
    image = class_image[class_name]
    _, buffer = cv2.imencode('.png', image)
    return send_file(io.BytesIO(buffer), mimetype='image/png')


@application.route('/fps', methods=['GET'])
def frames_per_second():
    fps = round(process_time['fps'], 2)
    return f'fps - {fps}'


if __name__ == "__main__":
    # application.run(host='192.168.43.73')
    application.run()
