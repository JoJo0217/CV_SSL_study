import argparse

import torch

from logger import Logger
from model import load_model
from dataset import load_data
from train import train
from optim import load_optimizer, load_criterion, load_scheduler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="lenet5")
    parser.add_argument("--output", type=str, default="./model.pth")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--criterion", type=str, default="cross_entropy")
    parser.add_argument("--scheduler", type=str, default=None)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--T_max", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--logdir", type=str, default="./logs")
    parser.add_argument("--logging_step", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    torch.manual_seed(args.seed)

    # load dataset
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

    model = load_model(args.model, class_num=trainloader.dataset.class_num)
    model = model.to(device)
    criterion = load_criterion(args.criterion)
    optimizer = load_optimizer(
        args.optimizer, model, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, T_max=args.T_max)
    scheduler = load_scheduler(args.scheduler, optimizer, args)
    # train
    logger = Logger(args.logdir)
    logger.log(0, is_tensor_board=False, **vars(args))
    print("start training")
    model = train(
        model,
        criterion,
        optimizer,
        args.epoch,
        trainloader,
        testloader,
        logging_step=args.logging_step,
        logger=logger,
        scheduler=scheduler,
        scheduer_type=args.scheduler,
    )
    print("finish training")

    print("saving...")
    torch.save(model, args.output)
    print("save success")
    return None


if __name__ == "__main__":
    main()
