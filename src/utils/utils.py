import sys
import time
import matplotlib.pyplot as plt
import torch
import cv2
import datetime
import os
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
from src.models.resnet50 import ResNet50

import warnings

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
def detect_and_draw_landmarks_image(image_path,best_network,device):
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
def detect_and_draw_landmarks_video(video_path,best_network,device):
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
def detect_and_draw_landmarks_realtime(best_network,device):
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
def train_test(model, epochs, learning_rate, train_loader, test_loader,best_network,device):
    torch.autograd.set_detect_anomaly(True)
    network = model
    # network.load_state_dict(torch.load("checkpoint/train_rn50_120+80rt.pth"))  # Đúng
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