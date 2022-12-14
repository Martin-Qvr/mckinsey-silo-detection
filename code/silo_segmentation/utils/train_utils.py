from tqdm import tqdm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_fn(dataloader, model, optimizer):
  """ 
  Train function to feed in the pipeline.
  """
  model.train() # Turn on dropout, batchnorm etc.

  total_loss = 0

  for images, masks in tqdm(dataloader):
    images = images.to(DEVICE)
    masks = masks.to(DEVICE)

    optimizer.zero_grad()
    logits, loss = model(images, masks)
    loss.backward()
    optimizer.step()

    total_loss += loss.item()



  return total_loss / len(dataloader)



def eval_fn(dataloader, model):
  """ 
  Validation function to feed in the training pipeline.
  """
  model.eval() # Turn OFF dropout, batchnorm etc.

  total_loss = 0
  
  for images, masks in tqdm(dataloader):
    images = images.to(DEVICE)
    masks = masks.to(DEVICE)

    logits, loss = model(images, masks)
    loss.backward()

    total_loss += loss.item()



  return total_loss / len(dataloader)