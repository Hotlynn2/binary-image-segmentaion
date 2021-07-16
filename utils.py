import torch
import torchvision
from dataset import EYTrainDataset
from dataset import EYPredDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from PIL import Image
# Image.MAX_IMAGE_PIXELS = 933120000            #--Use this when pixels exceed maximmum size, but not recommended

def save_checkpoint(state, filename="model_checkpoint.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_train_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = EYTrainDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = EYTrainDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def get_pred_loader(
    pred_dir,
    batch_size,
    pred_transform,
    num_workers=4,
    pin_memory=True,
):

    pred_ds = EYPredDataset(
        image_dir=pred_dir,
        transform=pred_transform,
    )

    pred_loader = DataLoader(
        pred_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return pred_loader

loss_fn = nn.BCEWithLogitsLoss()

def check_accuracy(loader, model, batch_size, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()
    running_loss = 0
    check_accuracy.validation_loss = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            pred = torch.sigmoid(model(x))
            preds = pred
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )
            loss = loss_fn(pred, y)
            running_loss+= loss.item()  *batch_size
            val_loss = running_loss/ len(loader)
            print('Val_loss: ' + str(val_loss))
            check_accuracy.validation_loss.append(val_loss)

    print(f"Dice score: {dice_score/len(loader)}")
    model.train()

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/predicted_imgs.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/original_imgs.png") 

    model.train()

def make_prediction_on_new_imgs(
    loader, model, folder="predicted_images/", device="cuda"
):
    model.eval()
    
    for idx, x in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/predicted_img(s).png"
        )

    model.train()
