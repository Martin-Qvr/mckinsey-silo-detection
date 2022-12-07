import torch
import pandas as pd
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from typing import Tuple, Dict, List


class SilosImage(Dataset):
  
  def __init__(self, img_path, label_path=None, val=False, transform=None):


    self.transform = transform
    self.paths = list(iter(img_path.glob('*.png')))

    if label_path:
      self._typ = 'train'
      self.label = pd.read_csv(label_path).set_index('filename').to_dict('index')
      if val:
        self.paths = [img for img in self.paths if self.label[img.stem+'.png']['split']=='validation']
      else:
        self.paths = [img for img in self.paths if self.label[img.stem+'.png']['split']=='train']
    else:
      self._typ = 'test'

    self.classes = [0, 1]


  def load_image(self, index: int) -> Image.Image:
    "Opens an image via a path and returns it."
    image_path = self.paths[index]
    return Image.open(image_path) 

  def __len__(self) -> int:
    "Returns the total number of samples."
    return len(self.paths)

  def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
    "Returns one sample of data, data and label (X, y, filename)."
    img = self.load_image(index)
    filename =  self.paths[index].stem + '.png'
    class_name = -1

    if self._typ == 'train':
      class_name  = self.label[filename]['class'] 
  
    if self.transform:
        return self.transform(img), class_name, filename 
    else:
        return img, class_name, filename 
           