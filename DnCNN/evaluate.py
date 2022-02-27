import math
import time
import os

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
TEST_SIZE = 1204
BATCHES = math.ceil(TEST_SIZE / BATCH_SIZE)
DATASET_DIR = "/storage/zurich-dataset"
INPUT_DIR = "/storage/dncnn_2021_6_22_21"

if __name__ == '__main__':
    # Output folder creation
    output_dir = os.path.join(INPUT_DIR, "evaluation")
    os.makedirs(output_dir, exist_ok=True)

    # Initiate log file
    log_file = open(f"{output_dir}/log.txt", "w")

    # Device definition
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    text = f"Using device: {device}\n"
    print(text)
    log_file.write(f"{text}\n")


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

    # Cost function
    cost = MSELoss()

    # Evaluate the model
    text = f"Evaluating model..."
    print(text)
    log_file.write(f"{text}\n")
    loss_mse_eval = 0
    loss_psnr_eval = 0

    with torch.no_grad():
        for idx, (test_x, test_y) in enumerate(test_dataloader):
            print("Processing", test_x.shape, "elements")
            predict_y = model(test_x.float().to(device)).detach()
            loss_mse_temp = cost(predict_y, test_y.float().to(device))
            print("mse loss", loss_mse_temp)
            loss_mse_eval += loss_mse_temp
            loss_psnr_eval += 20 * math.log10(1.0 / math.sqrt(loss_mse_temp))

        loss_mse_eval = loss_mse_eval / BATCHES
        loss_psnr_eval = loss_psnr_eval / BATCHES

        text = f"Mean Evaluation Loss: {loss_mse_eval}"
        print(text)
        log_file.write(f"{text}\n")
        text = f"Mean PSNR Score: {loss_psnr_eval}"
        print(text)
        log_file.write(f"{text}\n")
        text = "============================="
        print(text)
        log_file.write(f"{text}\n")
    log_file.flush()
