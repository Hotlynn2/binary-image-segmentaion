import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import UNET
from utils import (
    load_checkpoint,
    get_pred_loader,
    make_prediction_on_new_imgs
)

# Hyperparameters
DEVICE = "cpu"                   # "cuda" if torch.cuda.is_available() else "cpu"      --uncomment this if you have GPU
BATCH_SIZE = 50                  # number of images to predict at a time               --50 predictions at a go or less
NUM_WORKERS = 2
IMAGE_HEIGHT = 512               # you can change to desired height if you are predicting just one image
IMAGE_WIDTH = 512                # you can change to desired width if you are predicting just one image
PIN_MEMORY = True
LOAD_MODEL = True
PRED_IMG_DIR = "data/pred_images/"

def predict():
    pred_transforms = A.Compose(
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

    pred_loader = get_pred_loader(
        PRED_IMG_DIR,
        BATCH_SIZE,
        pred_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("model_checkpoint.tar"), model)


    # print some examples to a folder
    make_prediction_on_new_imgs(
        pred_loader, model, folder="predicted_images/", device=DEVICE
    )
    

if __name__ == "__main__":
    predict()