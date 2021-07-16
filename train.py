import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from model import UNET
# Image.MAX_IMAGE_PIXELS = 933120000            #--Use this when pixels exceed maximmum size, but not recommended
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_train_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cpu"                     # "cuda" if torch.cuda.is_available() else "cpu"    --uncomment this if you have GPU
BATCH_SIZE = 15
NUM_EPOCHS = 200
NUM_WORKERS = 2
IMAGE_HEIGHT = 512 
IMAGE_WIDTH = 512 
PIN_MEMORY = True
LOAD_MODEL = True
TRAIN_IMG_DIR = "data/train_images/"
TRAIN_MASK_DIR = "data/train_masks/"
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/"

def train_fn(loader, model, batch_size, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    running_loss = 0
    train_fn.training_loss = []

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
        running_loss+= loss.item()  * batch_size
        train_loss = running_loss/ len(loader)
        train_fn.training_loss.append(float(train_loss))

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=train_loss)


def train():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_train_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("model_checkpoint.tar"), model)


    check_accuracy(val_loader, model, BATCH_SIZE, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    epoch = NUM_EPOCHS
    n = 1
    while n <= epoch:
        print('start of epoch ' + str(n))
        train_fn(train_loader, model, BATCH_SIZE, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, BATCH_SIZE, device=DEVICE)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE
        )
        print('end of epoch ' + str(n))
        n = n + 1


    # I trie plotting the val loss and train loss autommatically but code was not working

    # epochs = []
    # for epoch in range(NUM_EPOCHS):
    #     epochs.append(epoch + 1)
    # fig, ax = plt.subplots(figsize=(20, 15))
    # plt.rcParams['font.size'] = '20'
    # ax.plot(epoch, np.array(check_accuracy.validation_loss), color='orange', label='Validation loss')
    # print(check_accuracy.validation_loss)
    # print(train_fn.training_loss)
    # # ax.plot(epoch, np.array(train_fn.training_loss), color='blue', label='Training loss')
    # ax.set_title('Validation loss vs Training loss', fontsize=30)
    # ax.set_xlabel('Number of epochs', fontsize = 25)
    # ax.set_ylabel('Validation and Training loss', fontsize = 25)
    # leg = ax.legend(prop={"size":20})
    # plt.savefig('losses')




if __name__ == "__main__":
    train()