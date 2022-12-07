import torch
import torchvision
import multiprocessing

from torch import nn 
from PIL import Image
from pathlib import Path
from torchvision import transforms 
from sklearn.metrics import roc_curve, auc, roc_auc_score
from tqdm.auto import tqdm

BATCH_SIZE = 32
EPOCHS = 20
NUM_WORKERS = multiprocessing.cpu_count()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_step(model,
               dataloader,
               loss_fn,
               optimizer,
               device,):

  """
    Trains a PyTorch model for a single epoch.

    Parameters:
    - model: A PyTorch model to be trained.
    - dataloader: A DataLoader instance for the model to be trained on.
    - loss_fn: A PyTorch loss function to minimize.
    - optimizer: A PyTorch optimizer to help minimize the loss function.
    - device: A target device to compute on (e.g. "cuda" or "cpu").

  """
  
  model.train()

  epoch_loss = 0
  epoch_auc = 0

  for batch, (X, y, _) in enumerate(dataloader):
    X, y = X.to(device), y.to(device)

    y_pred_logits = model(X)

    loss = loss_fn(y_pred_logits, y.unsqueeze(dim=1).to(torch.float32))
    epoch_loss += loss 

    fpr, tpr, thresholds = roc_curve( y.unsqueeze(dim=1).cpu().detach().numpy() , torch.sigmoid(y_pred_logits).cpu().detach().numpy() , pos_label=1)
    epoch_auc += auc(fpr, tpr) 

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    del X
    del y
    torch.cuda.empty_cache()
    

  epoch_loss = epoch_loss / len(dataloader)
  epoch_auc = epoch_auc / len(dataloader)
  return epoch_loss, epoch_auc


def test_step(model,
               dataloader,
              loss_fn,
              device):

"""
  Test a PyTorch model after a training epoch.

  Parameters:
  - model: A PyTorch model to be trained.
  - dataloader: A DataLoader instance for the model to be trained on.
  - loss_fn: A PyTorch loss function to minimize.
  - device: A target device to compute on (e.g. "cuda" or "cpu").

  """
  
  model.eval()

  epoch_loss = 0
  epoch_auc = 0

  with torch.inference_mode():
    for batch, (X, y, _) in enumerate(dataloader):
      X, y = X.to(device), y.to(device)
      y_pred_logits = model(X)
      
      loss = loss_fn(y_pred_logits, y.unsqueeze(dim=1).to(torch.float32))
      epoch_loss += loss 

      fpr, tpr, thresholds = roc_curve( y.unsqueeze(dim=1).cpu().detach().numpy() , torch.sigmoid(y_pred_logits).cpu().detach().numpy() , pos_label=1)
      epoch_auc += auc(fpr, tpr) 

    del X
    del y
    torch.cuda.empty_cache()

  epoch_loss = epoch_loss / len(dataloader)
  epoch_auc = epoch_auc / len(dataloader)
  return epoch_loss, epoch_auc

def train(model, 
          train_dataloader,
          test_dataloader,
          loss_fn,
          optimizer,
          epochs=EPOCHS,
          device=device,
):
  """
    Trains and tests a PyTorch model.


    Parameters:
    - model: A PyTorch model to be trained and tested.
    - train_dataloader: A DataLoader instance for the model to be trained on.
    - test_dataloader: A DataLoader instance for the model to be tested on.
    - optimizer: A PyTorch optimizer to help minimize the loss function.
    - loss_fn: A PyTorch loss function to calculate loss on both datasets.
    - epochs: An integer indicating how many epochs to train for.
    - device: A target device to compute on (e.g. "cuda" or "cpu").
  """

  train_loss_ls = []
  test_loss_ls = []

  train_auc_ls = []
  test_auc_ls = []

  for epoch in tqdm(range(epochs)):
    train_loss, train_auc = train_step(model, train_dataloader, loss_fn, optimizer, device)
    test_loss, test_auc = test_step(model, test_dataloader, loss_fn, device)

    train_loss_ls.append(train_loss.item())
    train_auc_ls.append(train_auc)
    test_loss_ls.append(test_loss.item())
    test_auc_ls.append(test_auc)

    print(
        f"Epoch: {epoch+1} | "
        f"train_loss: {train_loss:.4f} | "
        f"train_acc: {train_auc:.4f} | "
        f"test_loss: {test_loss:.4f} | "
        f"test_acc: {test_auc:.4f}"
    )

  return {"train_loss": train_loss_ls,
             "train_auc": train_auc_ls,
             "test_loss": test_loss_ls,
             "test_auc": test_auc_ls} 
