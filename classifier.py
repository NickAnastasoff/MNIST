import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
# define the CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(128 * 7 * 7, 1024)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.pool(self.relu1(self.conv1(x)))
        x = self.pool(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

def image_transform(imagepath):
    transformer = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
    ])
    
    image = Image.open(imagepath)
    image = transformer(image).unsqueeze(0)
    return image

def predict(imagepath, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        image = image_transform(imagepath).to(device)
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return predicted

def classify(imagepath):
    model_path = 'model.pth'
    model = CNN()
    model.load_state_dict(torch.load(model_path))
    print('Class:', predict(imagepath, model).item())
