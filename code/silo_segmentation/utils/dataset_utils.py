from torch.utils.data import Dataset
from PIL import Image
from typing import Tuple


class SegmentationDataset(Dataset):
  def __init__(self, df):
    self.df = df
            
  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    row = self.df.iloc[idx]

    image_path = DATA_DIR + "/" + row.images
    mask_path = DATA_DIR +"/" + row.masks

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB )

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) /255  # format is (h, w)

    mask = np.expand_dims(mask, axis=-1) #(h, w, channel)
    
    image = np.transpose(image, (2, 0, 1)).astype(np.float32) # (channel, h, w)
    mask = np.transpose(mask, (2, 0, 1)).astype(np.float32) # (channel, h, w)

    image = torch.Tensor(image) / 255.0
    mask = torch.round(torch.Tensor(mask)) # / 255.0

    return image, mask