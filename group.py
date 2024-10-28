import time
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
import imutils
from math import *
import random
import xml.etree.ElementTree as ET
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from torchvision import datasets, models, transforms
import hydra
from omegaconf import DictConfig
from torch.utils.data import Dataset
import cv2
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class Transforms():
    def __init__(self):
        pass

    def crop_face(self, image, landmarks, crops):
        top = int(crops['top'])
        left = int(crops['left'])
        height = int(crops['height'])
        width = int(crops['width'])

        image = TF.crop(image, top, left, height, width)

        img_shape = np.array(image).shape
        landmarks = torch.tensor(landmarks) - torch.tensor([[left, top]])
        landmarks = landmarks / torch.tensor([img_shape[1], img_shape[0]])
        return image, landmarks

    def resize(self, image, landmarks, img_size):
        image = TF.resize(image, img_size)
        return image, landmarks

    def color_jitter(self, image, landmarks):
        # ranNum = random.random()
        color_jitter = transforms.ColorJitter(brightness=random.random(),
                                              contrast=random.random(),
                                              saturation=random.random(),
                                              hue=random.uniform(0, 0.5))
        image = color_jitter(image)
        return image, landmarks

    def rotate(self, image, landmarks, angle):
        angle = random.uniform(-angle, +angle)

        transformation_matrix = torch.tensor([
            [+cos(radians(angle)), -sin(radians(angle))],
            [+sin(radians(angle)), +cos(radians(angle))]
        ])

        image = imutils.rotate(np.array(image), angle)

        landmarks = landmarks - 0.5
        new_landmarks = np.matmul(landmarks, transformation_matrix)
        new_landmarks = new_landmarks + 0.5
        return Image.fromarray(image), new_landmarks

    def __call__(self, image, landmarks, crops):
        image = Image.fromarray(image)
        image, landmarks = self.crop_face(image, landmarks, crops)
        image, landmarks = self.resize(image, landmarks, (224, 224))
        image, landmarks = self.color_jitter(image, landmarks)
        image, landmarks = self.rotate(image, landmarks, angle=random.randint(-30, 30))

        image = TF.to_tensor(image)
        image = TF.normalize(image, [0.5], [0.5])
        return image, landmarks
class FaceLandmarksDataset(Dataset):

    def __init__(self, transform=None):
        tree = ET.parse('data/ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train.xml')
        root = tree.getroot()

        self.image_filenames = []
        self.landmarks = []
        self.crops = []
        self.transform = transform
        self.root_dir = 'data/ibug_300W_large_face_landmark_dataset'

        for filename in root[2]:
            self.image_filenames.append(os.path.join(self.root_dir, filename.attrib['file']))

            self.crops.append(filename[0].attrib)

            landmark = []
            for num in range(68):
                x_coordinate = int(filename[0][num].attrib['x'])
                y_coordinate = int(filename[0][num].attrib['y'])
                landmark.append([x_coordinate, y_coordinate])
            self.landmarks.append(landmark)

        self.landmarks = np.array(self.landmarks).astype('float32')

        assert len(self.image_filenames) == len(self.landmarks)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        image = cv2.imread(self.image_filenames[index], 0)
        landmarks = self.landmarks[index]

        if self.transform:
            image, landmarks = self.transform(image, landmarks, self.crops[index])

        landmarks = landmarks - 0.5

        return image, landmarks
class ResNet50(nn.Module):
    def __init__(self, num_classes=136):
        super().__init__()
        self.model_name = 'resnet50'
        self.model = models.resnet50()
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x

best_network = ResNet50()
best_network.load_state_dict(torch.load("checkpoint/trainRS50_120.pth",weights_only=True))  # Đúng
best_network.to(device)
best_network.eval()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

dataset = FaceLandmarksDataset(Transforms())
len_test_set = int(0.2*len(dataset))
len_train_set = len(dataset) - len_test_set

print("The length of Train set is {}".format(len_train_set))
print("The length of Test set is {}".format(len_test_set))

train_dataset , test_dataset = torch.utils.data.random_split(dataset , [len_train_set, len_test_set])

# setting batch sizes and shuffle the data
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=6)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=6)

def detect_and_draw_landmarks_image(image_path):
    """
    Phát hiện khuôn mặt và vẽ landmark từ một ảnh.
    """
    # Đọc ảnh mới
    image = cv2.imread(image_path, 1)

    # Phát hiện khuôn mặt
    faces = face_cascade.detectMultiScale(image, 1.1, 4)

    # Tiền xử lý ảnh
    for (x, y, w, h) in faces:
        face_image_gray = cv2.cvtColor(image[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
        face_image = Image.fromarray(face_image_gray)
        face_image = transforms.Resize((224, 224))(face_image)
        face_image = transforms.ToTensor()(face_image)
        face_image = transforms.Normalize([0.5], [0.5])(face_image)
        face_image = face_image.unsqueeze(0).to(device)

        # Dự đoán landmark
        with torch.no_grad():
            predictions = best_network(face_image)
        predictions = (predictions.cpu() + 0.5) * 224
        predictions = predictions.view(-1, 68, 2).numpy()

        scale_x = w / 224
        scale_y = h / 224

        # Chuyển đổi tọa độ landmark về ảnh gốc
        for i in range(68):
            landmark_x = int(predictions[0, i, 0] * scale_x + x)
            landmark_y = int(predictions[0, i, 1] * scale_y + y)

            # Vẽ landmark lên ảnh gốc
            cv2.circle(image, (landmark_x, landmark_y), 2, (0, 255, 0), -1)

        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Tạo thư mục results nếu chưa tồn tại
    results_dir = 'Results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Lưu kết quả với tên file là ngày giờ
    output_filename = f"{current_datetime}.jpg"
    output_path = os.path.join(results_dir, output_filename)
    cv2.imwrite(output_path, image)
    # Hiển thị kết quả
    cv2.imshow('Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def detect_and_draw_landmarks_video(video_path):
    """
    Phát hiện khuôn mặt và vẽ landmark từ video.
    """
    cap = cv2.VideoCapture(video_path)

    # Lấy thông tin về video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Tạo VideoWriter để lưu video kết quả
    current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{current_datetime}_output.mp4"
    output_path = os.path.join('Results', output_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        # Phát hiện khuôn mặt
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=8)

        # Tiền xử lý ảnh và dự đoán landmark cho từng khuôn mặt
        for (x, y, w, h) in faces:
            face_image_gray = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
            face_image = Image.fromarray(face_image_gray)
            face_image = transforms.Resize((224, 224))(face_image)
            face_image = transforms.ToTensor()(face_image)
            face_image = transforms.Normalize([0.5], [0.5])(face_image)
            face_image = face_image.unsqueeze(0).to(device)

            # Dự đoán landmark
            with torch.no_grad():
                predictions = best_network(face_image)
            predictions = (predictions.cpu() + 0.5) * 224
            predictions = predictions.view(-1, 68, 2).numpy()

            # Tính toán tỷ lệ resize
            scale_x = w / 224
            scale_y = h / 224

            # Chuyển đổi tọa độ landmark về ảnh gốc
            for i in range(68):
                landmark_x = int(predictions[0, i, 0] * scale_x + x)
                landmark_y = int(predictions[0, i, 1] * scale_y + y)

                # Vẽ landmark lên ảnh gốc
                cv2.circle(frame, (landmark_x, landmark_y), 2, (0, 255, 0), -1)

        # Ghi khung hình vào video kết quả
        out.write(frame)

        # Hiển thị khung hình
        cv2.imshow('Video Filter', frame)

        # Thoát khi nhấn phím 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng camera và đóng cửa sổ
    cap.release()
    out.release()
    cv2.destroyAllWindows()
def detect_and_draw_landmarks_realtime():
    """
    Phát hiện khuôn mặt và vẽ landmark trong thời gian thực từ camera.
    """
    cap = cv2.VideoCapture(0)

    while (True):
        # Đọc một khung hình từ camera
        ret, frame = cap.read()

        # Phát hiện khuôn mặt
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=8)

        # Tiền xử lý ảnh và dự đoán landmark cho từng khuôn mặt
        for (x, y, w, h) in faces:
            face_image_gray = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
            face_image = Image.fromarray(face_image_gray)
            face_image = transforms.Resize((224, 224))(face_image)
            face_image = transforms.ToTensor()(face_image)
            face_image = transforms.Normalize([0.5], [0.5])(face_image)
            face_image = face_image.unsqueeze(0).to(device)

            # Dự đoán landmark
            with torch.no_grad():
                predictions = best_network(face_image)
            predictions = (predictions.cpu() + 0.5) * 224
            predictions = predictions.view(-1, 68, 2).numpy()

            # Tính toán tỷ lệ resize
            scale_x = w / 224
            scale_y = h / 224

            # Chuyển đổi tọa độ landmark về ảnh gốc
            for i in range(68):
                landmark_x = int(predictions[0, i, 0] * scale_x + x)
                landmark_y = int(predictions[0, i, 1] * scale_y + y)

                # Vẽ landmark lên ảnh gốc
                cv2.circle(frame, (landmark_x, landmark_y), 2, (0, 255, 0), -1)

        # Hiển thị khung hình
        cv2.imshow('Realtime Filter', frame)
        # Thoát khi nhấn phím 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng camera và đóng cửa sổ
    cap.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def print_overwrite(step, total_step, loss, operation):
    sys.stdout.write('\r')
    if operation == 'train':
        sys.stdout.write("Train Steps: %d/%d  Loss: %.4f " % (step, total_step, loss))
    else:
        sys.stdout.write("Valid Steps: %d/%d  Loss: %.4f " % (step, total_step, loss))

    sys.stdout.flush()
def euclidean_distance(predictions, landmarks):
    """
    Tính khoảng cách Euclidean giữa các landmark dự đoán và thực tế.

    Args:
      predictions: Các landmark dự đoán, có dạng (batch_size, 68, 2).
      landmarks: Các landmark thực tế, có dạng (batch_size, 68, 2).

    Returns:
      Khoảng cách Euclidean trung bình giữa các landmark.
    """
    return torch.mean(torch.sqrt(torch.sum((predictions - landmarks) ** 2, dim=2)))
def train_test(model, epochs, learning_rate, train_loader, test_loader):
    torch.autograd.set_detect_anomaly(True)
    network = model
    # network.load_state_dict("checkpoint/progress2.pth")
    network.load_state_dict(torch.load("checkpoint/train_rn50_120+80rt.pth"))  # Đúng
    network.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    num_epochs = epochs
    loss_min = np.inf

    train_loss_record = []
    test_loss_record = []

    start_time = time.time()
    for epoch in range(1, num_epochs + 1):

        loss_train = 0
        loss_test = 0
        running_loss = 0

        train_accuracy = 0
        test_accuracy = 0

        network.train()
        for step in range(1, len(train_loader) + 1):
            images, landmarks = next(iter(train_loader))

            images = images.to(device)
            landmarks = landmarks.view(landmarks.size(0), -1).to(device)

            predictions = network(images)

            # clear all the gradients before calculating them
            optimizer.zero_grad()

            # find the loss for the current step
            loss_train_step = criterion(predictions, landmarks)

            # loss_valid_step = criterion(predictions.logits, landmarks)

            # calculate the gradients
            loss_train_step.backward()

            # update the parameters
            optimizer.step()

            loss_train = loss_train + loss_train_step.item()
            running_loss = loss_train / step

            print_overwrite(step, len(train_loader), running_loss, 'train')

            # Tính toán độ chính xác trên tập huấn luyện
            predictions = (predictions.view(-1, 68, 2).cpu() + 0.5) * 224
            landmarks = (landmarks.view(-1, 68, 2).cpu() + 0.5) * 224
            train_accuracy += euclidean_distance(predictions, landmarks)

        network.eval()
        with torch.no_grad():

            for step in range(1, len(test_loader) + 1):
                images, landmarks = next(iter(test_loader))

                images = images.to(device)
                landmarks = landmarks.view(landmarks.size(0), -1).to(device)

                predictions = network(images)

                # find the loss for the current step
                loss_test_step = criterion(predictions, landmarks)
                # loss_valid_step = criterion(predictions.logits, landmarks)

                loss_test = loss_test + loss_test_step.item()
                running_loss = loss_test / step

                print_overwrite(step, len(test_loader), running_loss, 'test')

                # Tính toán độ chính xác trên tập kiểm tra
                predictions = (predictions.view(-1, 68, 2).cpu() + 0.5) * 224
                landmarks = (landmarks.view(-1, 68, 2).cpu() + 0.5) * 224
                test_accuracy += euclidean_distance(predictions, landmarks)

        loss_train = loss_train / len(train_loader)
        loss_test = loss_test / len(test_loader)

        train_accuracy = train_accuracy / len(train_loader)
        test_accuracy = test_accuracy / len(test_loader)

        print('\n--------------------------------------------------')
        print('Epoch: {}  Train Loss: {:.6f}  Test Loss: {:.6f}'.format(epoch, loss_train, loss_test))
        print('Train Accuracy: {:.6f}  Test Accuracy: {:.6f}'.format(train_accuracy, test_accuracy))
        print('--------------------------------------------------')

        train_loss_record.append(loss_train)
        test_loss_record.append(loss_test)

        if loss_test < loss_min:
            loss_min = loss_test
            torch.save(network.state_dict(), 'checkpoint/test.pth')
            print("\nMinimum Test Loss of {:.6f} at epoch {}/{}".format(loss_min, epoch, num_epochs))
            print('Model Saved\n')

    print('Training Complete')
    print("Total Elapsed Time : {} s".format(time.time() - start_time))

    plt.subplot(1, 2, 1)
    plt.title("Train Loss")
    plt.plot(train_loss_record)
    plt.xticks(range(1, len(train_loss_record) + 1, 1))
    plt.ylabel('MSE Loss')
    plt.xlabel('Epochs')

    plt.subplot(1, 2, 2)
    plt.title("Test Loss")
    plt.plot(test_loss_record)
    plt.xticks(range(1, len(test_loss_record) + 1, 1))
    plt.ylabel('MSE Loss')
    plt.xlabel('Epochs')
def show_result(model, test_loader):
    start_time = time.time()

    with torch.no_grad():
        best_network = model
        best_network.to(device)
        best_network.load_state_dict(
            torch.load('/home/phong/PycharmProjects/Filter project/notebook/checkpoint/train_rn50_120+80rt.pth',weights_only=True))
        best_network.eval()

        images, landmarks = next(iter(test_loader))

        images = images.to(device)
        landmarks = (landmarks + 0.5) * 224

        predictions = (best_network(images).cpu() + 0.5) * 224
        predictions = predictions.view(-1, 68, 2)

        plt.figure(figsize=(10, 40))

        for img_num in range(10):
            plt.subplot(1, 10, img_num + 1)
            plt.imshow(images[img_num].cpu().numpy().transpose(1, 2, 0).squeeze(), cmap='gray')
            plt.scatter(predictions[img_num, :, 0], predictions[img_num, :, 1], c='r', s=5)
            plt.scatter(landmarks[img_num, :, 0], landmarks[img_num, :, 1], c='g', s=5)

    print('Total number of test images: {}'.format(len(test_dataset)))

    end_time = time.time()
    print("Elapsed Time : {}".format(end_time - start_time))
@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # train_test(best_network,1,0.0001,train_loader,test_loader)
    # show_result(best_network,test_loader)
    # detect_and_draw_landmarks_image('test3.jpg')
    # detect_and_draw_landmarks_video("test1.mp4")
    detect_and_draw_landmarks_realtime()
if __name__ == "__main__":
    main()