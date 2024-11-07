import csv

import cv2
import mediapipe as mp
import numpy as np


# Đọc file csv và trả về các điểm
def readCSV(file):
    landmarks = {}
    ids = []
    coordinates = []
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if (line_count != 0):
                landmarks[line_count] = {'id': int(row[0]),
                                         'x': int(row[1]),
                                         'y': int(row[2])}
                ids.append(int(row[0]))
                coordinates.append([int(row[1]), int(row[2])])
            line_count += 1
    return landmarks, ids, coordinates


def nhan_dien_khuon_mat(source):
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    cap = None
    image = None
    image_processed = False  # Cờ để xử lý ảnh một lần

    if source == "webcam":
        cap = cv2.VideoCapture(0)
    elif source == "video":
        video_path = input("Nhập đường dẫn đến video: ")
        cap = cv2.VideoCapture(video_path)
    elif source == "image":
        image_path = input("Nhập đường dẫn đến ảnh: ")
        image = cv2.imread(image_path)
        image_processed = True  # Đánh dấu đã xử lý ảnh ngay từ đầu

    with mp_face_mesh.FaceMesh(
            max_num_faces=4,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:

        while True:
            face_count = 0
            landmarks_coordinates = []
            # Đọc khung hình từ webcam hoặc video
            if cap and not image_processed:
                success, image = cap.read()
                if not success:
                    print("Video đã kết thúc.")
                    image_processed = True
                    continue

            # Xử lý ảnh hoặc khung hình video
            if image is not None:
                image.flags.writeable = False
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image_rgb)

                image.flags.writeable = True
                image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                height, width = image.shape[:2]
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        landmarks_coordinate = []
                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=drawing_spec,
                            connection_drawing_spec=drawing_spec)
                        for landmark_of_interest in ids:
                            y = int(face_landmarks.landmark[landmark_of_interest].y * height)
                            x = int(face_landmarks.landmark[landmark_of_interest].x * width)
                            landmarks_coordinate.append([x, y])
                        landmarks_coordinates.append(landmarks_coordinate)
                        face_count += 1
                    # Call warp function to apply homography with the face orientation
                    #   and mask image
                    im_src = cv2.imread(mask_filenames[0], cv2.IMREAD_UNCHANGED)
                    im_out = []
                    pts_src = np.array(mask_coordinates, dtype=float)
                    for i in range(face_count):
                        pts_dst = np.array(landmarks_coordinates[i], dtype=float)
                        # print(pts_src)
                        h, status = cv2.findHomography(pts_src, pts_dst)
                        im_out.append(cv2.warpPerspective(im_src, h, (image.shape[1], image.shape[0])))

                    dst = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA).astype(float) / 255.0
                    for img in im_out:
                        src = img.astype(float)
                        src = src / 255
                        alpha_foreground = src[:, :, 3]
                        for color in range(0, 3):
                            dst[:, :, color] = alpha_foreground * src[:, :, color] + (1 - alpha_foreground) * dst[:, :,
                                                                                                              color]

                    dst[:, :, :] = cv2.erode(dst[:, :, :], (5, 5), 0)
                    dst[:, :, :] = cv2.GaussianBlur(dst[:, :, :], (3, 3), 0)

                    output = dst
                else:
                    output = image
                cv2.imshow('MediaPipe Face Mesh', image)
                cv2.imshow("Live", output)
                if (source == "image"): image = None
            # Kiểm tra phím nhấn
            key = cv2.waitKey(5) & 0xFF
            if key == 27:  # Nhấn Esc để thoát
                break
            elif key == ord('w'):  # Nhấn 'w' để chuyển sang webcam
                source = "webcam"
                cap = cv2.VideoCapture(0)
                image = None
                image_processed = False
            elif key == ord('v'):  # Nhấn 'v' để chuyển sang video
                source = "video"
                video_path = input("Nhập đường dẫn đến video: ")
                cap = cv2.VideoCapture(video_path)
                image = None
                image_processed = False
            elif key == ord('i'):  # Nhấn 'i' để chuyển sang ảnh
                source = "image"
                image_path = input("Nhập đường dẫn đến ảnh: ")
                cap = cv2.imread(image_path)
                image = None
                image_processed = True

    if cap:
        cap.release()
    cv2.destroyAllWindows()

mask_filenames = ['../../../data/dataFilter/batman_1.png', '../../../data/dataFilter/batman_2.png', \
                  '../../../data/dataFilter/iron_man_2.png', '../../../data/dataFilter/none.png']
csv_filenames = ['../../../data/dataFilter/batman_1.csv', '../../../data/dataFilter/batman_2.csv', \
                 '../../../data/dataFilter/iron_man_2.csv']
landmarks, ids, mask_coordinates = readCSV(csv_filenames[0])
height, width = 480, 640
mask_width = 70
mask_height = 10
mask = cv2.imread(mask_filenames[0], cv2.IMREAD_UNCHANGED)
mask = cv2.resize(mask, (mask_width, mask_height))
mask = mask / 255.01
if __name__ == "__main__":
    print("Chọn nguồn đầu vào:")
    print("1. Webcam")
    print("2. Video")
    print("3. Ảnh")
    choice = input("Nhập lựa chọn (1, 2, hoặc 3): ")

    if choice == "1":
        nhan_dien_khuon_mat("webcam")
    elif choice == "2":
        nhan_dien_khuon_mat("video")
    elif choice == "3":
        nhan_dien_khuon_mat("image")
    else:
        print("Lựa chọn không hợp lệ.")
