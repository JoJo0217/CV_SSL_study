import torch
from tqdm import tqdm

def test_model(model,dataloader,device):
  acc=0
  for idx, data in tqdm(enumerate(dataloader,start=0)):
    inputs, labels=data[0].to(device), data[1].to(device)
    with torch.no_grad():
        outputs=model(inputs)
    output=torch.argmax(outputs,dim=1)
    acc+=torch.sum(output==labels).item()
  return acc/len(dataloader.dataset)
