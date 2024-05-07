import torch
from module.dataset import load_dataloader
import sys
from module.test import test_model

if __name__=='__main__':
    if len(sys.argv) < 2:
      print("사용법: python3 ./module/test.py [경로]")
      sys.exit(1)
    path = sys.argv[1]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _,testloader=load_dataloader(train=False,test=True,batch_size=32)

    #model 불러오기
    model_loaded=torch.load(path)
    model_loaded.to(device)


    print('acc: ',test_model(model_loaded,testloader,device))