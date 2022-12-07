import pandas as pd
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import torch 
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from torch.utils.data import DataLoader

from utils.dataset_utils import SegmentationDataset
from utils.train_utils import train_fn, eval_fn
from utils.plot_utils import show_image, dice_coef

from models.model_backboned import SegmentationModel
from models.own_model import build_unet



CSV_FILE = "../ai_ready/silo_only.csv" 
DATA_DIR = "../ai_ready"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 10
LR = 0.006
BATCH_SIZE = 16
IMG_SIZE = 256
ENCODER = "timm-efficientnet-b0"
WEIGHTS = "imagenet"

def train_model(test_size: float, model_name: str): 
    """ 
    This function is used to trigger the full pipeline.

    Parameters : 
    - test_size : corresponds to the size of the test set
    - model_name : corresponds to the .pt file you want to save
    """
    ## Read the path df and keep only the images where there is a silo
    df = pd.read_csv("../ai_ready/x-ai_data.csv")
    df = df.loc[df["class"] == 1]
    df.reset_index(inplace=True, drop=True)

    ## Load the train and validation dataset

    train_df, valid_df = train_test_split(df, test_size = 0.2, random_state=42)

    trainset = SegmentationDataset(train_df)
    validset = SegmentationDataset(valid_df)

    print(f'Size of trainset {len(trainset)}')
    print(f'Size of validset {len(validset)}')

    ## Feed them into the loader 

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    validloader = DataLoader(validset, batch_size=BATCH_SIZE)

    print(f"Total no of batches in trainloader: {len(trainloader)}")
    print(f"Total no of batches in validloader: {len(validloader)}")

    ## Load the model and send it to the CPU/GPU instance

    model =SegmentationModel()

    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    ## train the model and save it in the saved_models file 

    train_loss_ls = []
    val_loss_ls = []

    best_loss = np.Inf

    for i in range(EPOCHS):
        train_loss = train_fn(trainloader, model, optimizer)
        valid_loss = eval_fn(validloader, model)

        train_loss_ls.append(train_loss)
        val_loss_ls.append(valid_loss)

        if valid_loss < best_loss :
            torch.save(model.state_dict(), "saved_models/" + model_name)
            print('SAVED_MODEL')
            best_loss = valid_loss
            print(f"Epoch : {i+1} Train Loss : {train_loss} Valid Loss : {valid_loss}")


def plot_performance(idx, df):
    """ 
    This function computes and plots the Dice Score of the model for one image and its predicted mask again.
    """
    model =SegmentationModel()
    model.load_state_dict(torch.load('saved_models/second_model.pt'))

    df.loc[:, 'images'] = 'images/' + df.loc[:, 'filename']
    df.loc[:, 'masks'] = 'masks/' + df.loc[:, 'filename']
    df.reset_index(inplace=True, drop=True)
    
    testset = SegmentationDataset(df)

    image, mask = testset[idx]

    logits_mask = model(image.to(DEVICE).unsqueeze(0)) #(c, h,w) --> (batch, channel, h, w)
    pred_mask = torch.sigmoid(logits_mask)
    pred_mask = (pred_mask > 0.5) * 1

    mask = mask.to("cpu")
    pred_mask =pred_mask.to("cpu")

    mask = mask.type(torch.int64)
    mask

    dice= dice_coef(pred_mask, mask).numpy()

    show_image(image, mask, pred_mask.detach().cpu().squeeze(0), round(dice, 2))

    return dice_coef


def make_predictions(idx: int, df: pd.DataFrame, path: str):
    """ 
    This function is used to generate the final predictions file on a dataset of images 
    """

    model =SegmentationModel()
    model.load_state_dict(torch.load('saved_models/second_model.pt'))

    df.loc[:, 'images'] = 'images/' + df.loc[:, 'filename']
    df.loc[:, 'masks'] = 'masks/' + df.loc[:, 'filename']
    df.reset_index(inplace=True, drop=True)
    testset = SegmentationDataset(df)

    for i in df.index:
        image, mask = testset[i]
        logits_mask = model(image.to(DEVICE).unsqueeze(0)) #(c, h,w) --> (batch, channel, h, w)
        pred_mask = torch.sigmoid(logits_mask)
        pred_mask = (pred_mask > 0.5) * 1
        plt.imshow(pred_mask.detach().cpu().squeeze(0).permute(1,2,0).squeeze(),cmap = 'gray')
        cv2.imwrite(f"/Images/{df.loc[i, 'filename']}", pred_mask.detach().cpu().squeeze(0).permute(1,2,0).squeeze().numpy() * 255)
    