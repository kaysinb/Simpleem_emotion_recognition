import torch.hub
from torchvision import transforms
from torch.nn import Softmax
from models.mini_xception import *
from PIL import Image
import numpy as np

class EmotionsRecognition:

    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.emotions = ['angry', 'scared', 'happy', 'neutral', 'sad']
        model_path = 'models/public_model_89_63.t7'
        self.net = Model(num_classes=len(self.emotions))
        checkpoint = torch.load(model_path, map_location=torch.device(self.device))
        self.net.load_state_dict(checkpoint['net'])
        self.net.to(self.device).eval()
        self.softmax = Softmax(dim=1)
        self.n_pre = 5  # average smoothing by looking on n_pre emotions
        self.transforms = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(48),
            transforms.CenterCrop(44),
            transforms.ToTensor()
        ])

    def __call__(self, student, class_name):
        self.emotion_recognition(student, class_name)

    def emotion_recognition(self, student, class_name):

        faces = []
        names = []
        for name in student.group[class_name]:
            face = student.group[class_name][name].face_image
            if face is not None:
                face = Image.fromarray(face)
                face_input = self.transforms(face)
                faces.append(face_input)
                names.append(name)

        if len(names) > 0:
            inputs = torch.stack(faces).to(self.device)
            output = self.softmax(self.net(inputs)).detach().cpu().numpy()

        for name in student.group[class_name]:
            person = student.group[class_name][name]
            if name in names:
                student_index = names.index(name)
                person.emotions = output[student_index].tolist()
                if person.current_emotion is None:
                    person.current_emotion = self.emotions[int(np.argmax(person.emotions))]
                else:
                    mask = person._stud_is_on_frame[-self.n_pre:]
                    person.current_emotion = self.emotions[int(np.argmax(
                        np.mean(np.array(person._emotion_logg[-self.n_pre:])[mask], axis=0)))]
            else:
                person.emotions = [None] * len(self.emotions)
                person.current_emotion = None
