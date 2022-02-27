import math
import time
import os

import cv2
import numpy as np
import torch
from progress.bar import Bar
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchsummary import summary


from model import DnCnn
from dataloader import ZurichDataset, ToTensor

BATCH_SIZE = 8
DATASET_DIR = "/storage/zurich-dataset"
INPUT_DIR = "/storage/dncnn_2021_6_22_21"

if __name__ == '__main__':
    # Output folder creation
    output_dir = os.path.join(INPUT_DIR, "apply")
    os.makedirs(output_dir, exist_ok=True)

    # Initiate log file
    # log_file = open(f"{output_dir}/log.txt", "w")

    # Device definition
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    text = f"Using device: {device}\n"
    print(text)
    # log_file.write(f"{text}\n")


    # Datasets definition
    test_dataset = ZurichDataset(
        DATASET_DIR,
        transform=transforms.Compose([
            ToTensor()
        ]),
        train=False
    )

    # Dataloader definition
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model initialization
    model_path = os.path.join(INPUT_DIR, "dncnn.pth")
    model = torch.load(model_path)
    model.eval()

    # Applying model
    text = f"Applying model..."
    print(text)
    # log_file.write(f"{text}\n")

    with torch.no_grad():
        (test_x, test_y) = next(iter(test_dataloader))
        print("Processing", test_x.shape, "elements")
        predict_y = model(test_x.float().to(device)).detach()
        print(predict_y.shape)
        out = predict_y.cpu().numpy().transpose([0, 2, 3, 1]) * 255
        print(out.shape)
        for idx, image in enumerate(out):
            print(idx, image.shape)
            cv2.imwrite(os.path.join(output_dir, f"output{idx}.png"), image)
            ground = test_y[idx].cpu().numpy().transpose([1, 2, 0]) * 255
            print(ground.shape)
            cv2.imwrite(os.path.join(output_dir, f"groundtruth{idx}.png"), ground)

    # log_file.close()
