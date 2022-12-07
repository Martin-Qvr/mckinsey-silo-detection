from torch import nn
class CNNModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.cnn_block = nn.Sequential(
        nn.Conv2d(3, 20, kernel_size=(11, 11), stride=(4, 4), padding=(1, 1), bias=True),
        nn.ReLU(),
        nn.BatchNorm2d(20),
        nn.MaxPool2d((3, 3), stride=(2, 2)),
        nn.Conv2d(20, 20, kernel_size=(5, 5), stride=(1, 1), padding='same', bias=True),
        nn.ReLU(),
        nn.BatchNorm2d(20),
        nn.MaxPool2d((3, 3), stride=(2, 2)),
        nn.MaxPool2d((3, 3), stride=(2, 2)),
        nn.Conv2d(20, 2, kernel_size=(3, 3), stride=(1, 1), padding='same', bias=True),
        nn.BatchNorm2d(2),
        nn.Flatten(),
            
        nn.Linear(in_features=72, out_features=40, bias=True),
        nn.Dropout(p=0.5, inplace=True), 
        nn.Linear(in_features=40, out_features=1, bias=True),


        
    )
  def forward(self, x):
    return self.cnn_block(x)