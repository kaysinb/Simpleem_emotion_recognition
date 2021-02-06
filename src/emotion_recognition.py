import torch.hub
from torchvision import transforms
from torch.nn import Softmax
from models.mini_xception import *
from PIL import Image


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
        self.transforms = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(48),
            transforms.CenterCrop(44),
            transforms.ToTensor()
        ])

    def __call__(self, student):
        self.emotion_recognition(student)

    def emotion_recognition(self, student):

        faces = []
        names = []
        for name in student.group:
            face = student.group[name].face_image
            if face is not None:
                face = Image.fromarray(face)
                face_input = self.transforms(face)
                faces.append(face_input)
                names.append(name)

        if len(names) > 0:
            inputs = torch.stack(faces).to(self.device)
            output = self.softmax(self.net(inputs)).detach().cpu().numpy()

        for name in student.group:
            if name in names:
                student_index = names.index(name)
                student.group[name].emotions = output[student_index].tolist()

            else:
                student.group[name].emotions = [None] * len(self.emotions)
