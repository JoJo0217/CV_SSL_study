import sys
import argparse

import torch
from tqdm import tqdm

from utils.dataset import load_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    return args


def eval_model(model, dataloader, device):
    acc = 0
    for idx, data in tqdm(enumerate(dataloader, start=0)):
        inputs, labels = data[0].to(device), data[1].to(device)
        with torch.no_grad():
            outputs = model(inputs)
        output = torch.argmax(outputs, dim=1)
        acc += torch.sum(output == labels).item()
    return acc / len(dataloader.dataset)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python3 evaluate.py [경로]")
        sys.exit(1)
    path = sys.argv[1]

    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    testloader = load_data(name=args.dataset, root=args.data_path,
                           train=False, batch_size=args.batch_size)

    model_loaded = torch.load(path)  # model 불러오기
    model_loaded.to(device)

    print("acc: ", eval_model(model_loaded, testloader, device))
