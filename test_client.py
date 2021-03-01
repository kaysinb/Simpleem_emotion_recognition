import requests
import os
import cv2
from tqdm import tqdm


photos_path = '../../er_test/photos'
students_dirs = [f for f in os.scandir(photos_path) if f.is_dir()]
for one_student in students_dirs:
    student_photos = []
    for file_name in os.listdir(one_student.path):
        print('Getting photo .... : ' + one_student.path + '/' + file_name)
        image = cv2.imread(one_student.path + '/' + file_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        _, buffer = cv2.imencode('.png', image)

        resp = requests.post("http://localhost:5000/initialization_dict",
                             files={'file': buffer, 'name': one_student.name.encode('utf8')})


resp = requests.get("http://localhost:5000/initialization")

video_path = 0

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

if video_path == 0:
    cap.set(3, 1280)
    cap.set(4, 720)

# Check if video opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")

firs_frame = 0
cap.set(1, firs_frame)  # read from firs_frame

for _ in tqdm(range(1000)):

    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _, buffer = cv2.imencode('.png', frame)
        resp = requests.post("http://localhost:5000/run", files={'file': buffer})

        print(resp.text)
