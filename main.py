import hydra
from omegaconf import DictConfig
import time
import cv2
import os
import warnings
import torch
from src.dataloaders.face_landmarks_dataset import FaceLandmarksDataset, Transforms
from src.models.resnet50 import ResNet50
from src.utils.utils import *

warnings.filterwarnings("ignore", category=DeprecationWarning)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Khởi tạo model
    model = ResNet50()
    model.load_state_dict(torch.load(cfg.model_path, weights_only=True))
    model.to(device)
    model.eval()

    # Khởi tạo face detector


    # Khởi tạo dataset và dataloader
    dataset = FaceLandmarksDataset(Transforms())
    len_test_set = int(cfg.test_size * len(dataset))
    len_train_set = len(dataset) - len_test_set

    print("The length of Train set is {}".format(len_train_set))
    print("The length of Test set is {}".format(len_test_set))

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len_train_set, len_test_set])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)

    # Chạy chức năng mong muốn dựa trên config
    if cfg.mode == "train":
        train_test(model, cfg.epochs, cfg.learning_rate, train_loader, test_loader,model,device)
    elif cfg.mode == "image":
        detect_and_draw_landmarks_image(cfg.image_path, model,device)
    elif cfg.mode == "video":
        detect_and_draw_landmarks_video(cfg.video_path, model,device)
    elif cfg.mode == "realtime":
        detect_and_draw_landmarks_realtime(model,device)

if __name__ == "__main__":
    main()