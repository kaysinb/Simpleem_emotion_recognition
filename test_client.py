import requests
import numpy as np
from PIL import Image
import io
import os
import cv2
from tqdm import tqdm

# this is the aws server address
entry_point = 'http://simapp-env.eba-rxssbrpe.eu-west-3.elasticbeanstalk.com'

# this is a unique name for each class
class_name = 'first'

# this is the path to students' photos for initialization
photos_path = '../../er_test/photos'

# the next block initializes the class
students_dirs = [f for f in os.scandir(photos_path) if f.is_dir()]
for one_student in students_dirs:
    student_photos = []
    for file_name in os.listdir(one_student.path):
        print('Getting photo .... : ' + one_student.path + '/' + file_name)
        image = cv2.imread(one_student.path + '/' + file_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        _, buffer = cv2.imencode('.png', image)

        resp = requests.post(f"{entry_point}/initialization",
                             files={'file': buffer, 'name': one_student.name.encode('utf8'),
                                    'class_name': class_name.encode('utf8')})
        print(resp.text)

resp = requests.get(f"{entry_point}/initialization", files={'class_name': class_name.encode('utf8')})
print(resp.text)


# in the next block, we read the video frame by frame and send it to the server for processing
video_path = 0  # if video_path = 0 it will be video from your webcam
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

if not cap.isOpened():
    print("Error opening video stream or file")

firs_frame = 0
cap.set(1, firs_frame)  # read from the firs_frame

for _ in tqdm(range(50)):  # here we use tqdm for speed estimation, but it is not necessary

    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        scale = 2  # here we can resize our frame to speed up the process
        frame = cv2.resize(frame, (frame.shape[1] // scale, frame.shape[0] // scale))
        _, buffer = cv2.imencode('.png', frame)

        # Post request. The response contains the recommendation and condition of the students
        resp = requests.post(f"{entry_point}/run", files={'file': buffer, 'class_name': class_name.encode('utf8')})
        print(resp.text)

        resp = requests.get(f"{entry_point}/fps")  # optionally, if you want to estimate the speed of the model ...
        print(resp.text)  # ... excluding the time of sending images

        # optionally, if you want to see the result of image processing
        resp = requests.get(f"{entry_point}/get_image", files={'class_name': class_name.encode('utf8')})
        frame = np.array(Image.open(io.BytesIO(resp.content)), dtype='uint8')
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow('meeting', frame)
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break
cv2.destroyAllWindows()

# and this request deletes all information about the class after the end of the lesson (!!! it is necessary !!!)
resp = requests.get(f"{entry_point}/end_lesson", files={'class_name': class_name.encode('utf8')})
print(resp.text)




