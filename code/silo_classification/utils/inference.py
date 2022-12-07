from utils.dataset import *
import multiprocessing
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch 


BATCH_SIZE = 32
NUM_WORKERS = multiprocessing.cpu_count()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_transform = transforms.Compose([
    transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
])
def inference(model, test_path, targ_path,  device=device, data_transform=data_transform, ):
  # Create test dataset
  test_dataset = SilosImage(test_path, val=False, transform=data_transform)

  model = model.to(device)
  # Get data into dataloader
  test_dataloader = DataLoader(dataset=test_dataset,
                                batch_size=BATCH_SIZE,
                                num_workers=NUM_WORKERS,
                                shuffle=False)
  model.eval()
  res_dict = {'filenames': [], 'class': []}
  with torch.inference_mode():
    for X, _ , filenames in test_dataloader:
      y_pred_label = torch.round(torch.sigmoid(model(X.to(device)))).squeeze()

      res_dict['filenames'] += filenames
      res_dict['class'] += y_pred_label.cpu() 
  
  df_res = pd.DataFrame.from_dict(res_dict )
  df_res['class'] = df_res['class'].map(lambda x: int(x.item()))

  df_res.to_csv(targ_path, index=False)
  return df_res 

