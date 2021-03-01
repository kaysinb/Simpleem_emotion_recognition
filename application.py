from flask import Flask, request, jsonify, render_template, send_file, make_response
import numpy as np
from src.faces_recognition import FacesRecognition
from src.emotion_recognition import EmotionsRecognition
from src.students import Student
from src.pose_estimation import PoseEstimation
from src.display_text import write_signature
from PIL import Image
import io
import cv2


app = Flask(__name__)

faces_recognizer = FacesRecognition(resize=0.5, max_face_tilt=10)
emotions_recognizer = EmotionsRecognition()
emotions_list = emotions_recognizer.emotions
pose_estimator = PoseEstimation()
student_initialization_dict = {}


@app.route('/initialization_dict', methods=['POST'])
def initialization_dict():
    file = request.files.get('file')
    name = request.files.get('name').read().decode('utf8')
    img_bytes = file.read()
    image = np.array(Image.open(io.BytesIO(img_bytes)), dtype='uint8')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if name not in student_initialization_dict:
        student_initialization_dict[name] = [image]
    else:
        student_initialization_dict[name].append(image)

    return jsonify({'ok': name})


@app.route('/initialization',  methods=['GET'])
def initialization():
    for name in student_initialization_dict:
        Student(student_initialization_dict[name], name, faces_recognizer, emotions_list)
    return jsonify({'initialization': list(Student.group.keys())})


@app.route('/run',  methods=['POST'])
def run():
    file = request.files.get('file')
    img_bytes = file.read()
    frame = np.array(Image.open(io.BytesIO(img_bytes)), dtype='uint8')

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces_recognizer(frame, Student)
    emotions_recognizer(Student)
    Student.logging_of_group()
    recommendation, attendance = Student.get_recommendation()
    frame = pose_estimator(frame, Student)
    write_signature(frame, Student)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imwrite('testing/frame.png', frame)

    return jsonify({'recommendation': recommendation[0], 'attendance': attendance[0]})


if __name__ == "__main__":
    app.run(host='192.168.43.73')