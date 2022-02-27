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
TRAIN_SIZE = 46839
TEST_SIZE = 1204
DATASET_DIR = "/storage/zurich-dataset"

if __name__ == '__main__':
    # Output folder creation
    year, month, day, hour, min = map(int, time.strftime("%Y %m %d %H %M").split())
    output_dir = f"/storage/dncnn_{year}_{month}_{day}_{hour}"
    os.makedirs(output_dir, exist_ok=True)

    # Initiate log file
    log_file = open(f"{output_dir}/log.txt", "w")

    # Device definition
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    text = f"Using device: {device}\n"
    print(text)
    log_file.write(f"{text}\n")

    # Datasets definition
    train_dataset = ZurichDataset(
        DATASET_DIR,
        transform=transforms.Compose([
            ToTensor()
        ]),
    )
    test_dataset = ZurichDataset(
        DATASET_DIR,
        transform=transforms.Compose([
            ToTensor()
        ]),
        train=False
    )

    # Dataloader definition
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = DnCnn().to(device)

    optimizer = Adam(params=model.parameters(), lr=5e-5)
    cost = MSELoss()

    epoch = 50
    for _epoch in range(epoch):
        # Train model
        torch.cuda.empty_cache()

        text = f"=====\nEpoch {_epoch}\n====="
        print(text)
        log_file.write(f"{text}\n")
        bar = Bar('Progress', suffix='%(percent)d%%', max=len(train_dataloader))
        start = time.time()
        total_loss = 0
        for idx, (train_x, train_y) in enumerate(train_dataloader):
            optimizer.zero_grad()

            predict_y = model(train_x.to(device))

            loss = cost(predict_y, train_y.to(device))
            total_loss += loss

            loss.backward()
            optimizer.step()
            bar.next()
        bar.finish()

        end = time.time()

        text = f"Mean EPOCH Training loss: {total_loss/TRAIN_SIZE}"
        print(text)
        log_file.write(f"{text}\n")
        text = f"Epoch computing time: {end-start}"
        print(text)
        log_file.write(f"{text}\n")
        text = "==============================="
        print(text)
        log_file.write(f"{text}\n")
        log_file.flush()

        if (_epoch + 1) % 10 == 0:
            # Evaluate the model
            text = f"Evaluating model for epoch {_epoch+1}"
            print(text)
            log_file.write(f"{text}\n")
            loss_mse_eval = 0
            loss_psnr_eval = 0

            model.eval()
            with torch.no_grad():
                for idx, (test_x, test_y) in enumerate(test_dataloader):
                    predict_y = model(test_x.float().to(device)).detach()
                    loss_mse_temp = cost(predict_y, test_y.float().to(device))

                    loss_mse_eval += loss_mse_temp
                    loss_psnr_eval += 20 * math.log10(1.0 / math.sqrt(loss_mse_temp))

                loss_mse_eval = loss_mse_eval / TEST_SIZE
                loss_psnr_eval = loss_psnr_eval / TEST_SIZE

                text = f"Mean EPOCH Evaluation Loss: {loss_mse_eval}"
                print(text)
                log_file.write(f"{text}\n")
                text = f"Mean EPOCH PSNR Score: {loss_psnr_eval}"
                print(text)
                log_file.write(f"{text}\n")
                text = "============================="
                print(text)
                log_file.write(f"{text}\n")
            log_file.flush()

        model.train()

    text = "Saving final model..."
    print(text)
    log_file.write(f"{text}\n")
    log_file.close()
    torch.save(model, f'{output_dir}/dncnn.pth')
