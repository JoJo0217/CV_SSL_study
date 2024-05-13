import argparse

import torch

from model import load_model
from dataset import load_data
from train import train
from optim import load_optimizer, load_criterion


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="lenet5")
    parser.add_argument("--output", type=str, default="./model.pth")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--criterion", type=str, default="cross_entropy")
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--logdir", type=str, default="./logs")
    parser.add_argument("--logging_step", type=int, default=None)
    args = parser.parse_args()
    return args


def main():
    args=parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    
    #load dataset
    trainloader = load_data(
        args.dataset, 
        root=args.data_path, 
        batch_size=args.batch_size, 
        train=True
        )
    testloader = load_data(
        args.dataset, 
        root=args.data_path, 
        batch_size=args.batch_size, 
        train=False
        )

    model = load_model(args.model)
    model = model.to(device)
    criterion = load_criterion(args.criterion)
    optimizer = load_optimizer(args.optimizer, model, lr=args.lr)

    #train
    print("start training")
    model = train(
        model, 
        criterion, 
        optimizer, 
        args.epoch, 
        trainloader, 
        testloader, 
        logging_step=args.logging_step,
        log_dir=args.logdir,
        )
    print("finish training")

    print("saving...")
    torch.save(model, args.output)
    print("save success")
    return None


if __name__ == "__main__":
    main()