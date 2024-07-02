from __future__ import print_function

import argparse
import os
import string
import glob

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DistributedSampler
from torchvision import datasets, transforms
from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook as powerSGD
from pathlib import Path
from typing import Callable, Optional, Union, List

class ProxyFashionMNIST(datasets.FashionMNIST):

    mirrors = ["http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"]

    def __init__(
        self,
        root: Union[str, Path],
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        mirrors: Optional[List[str]] = None,
    ) -> None:

        # 如果 mirrors 参数为空，则使用默认的 mirrors 值
        if mirrors is None:
            self.mirrors = datasets.FashionMNIST.mirrors
        else:
            self.mirrors = mirrors

        super(ProxyFashionMNIST, self).__init__(root, transform=transform, target_transform=target_transform, download=download)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, epoch, writer):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for batch_idx, (data, target) in enumerate(train_loader):
        # Attach tensors to the device.
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tloss={:.4f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            niter = epoch * len(train_loader) + batch_idx
            writer.add_scalar("loss", loss.item(), niter)


def test(model, device, test_loader, writer, epoch):
    model.eval()

    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # Attach tensors to the device.
            data, target = data.to(device), target.to(device)

            output = model(data)
            # Get the index of the max log-probability.
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    print("\naccuracy={:.4f}\n".format(float(correct) / len(test_loader.dataset)))
    writer.add_scalar("accuracy", float(correct) / len(test_loader.dataset), epoch)

def save_model(model, epoch, save_path="model_checkpoint.pth"):
    """
    保存模型状态到文件。
    
    参数:
    - model: 要保存的模型。
    - epoch: 当前epoch数，用于文件命名。
    - save_path: 保存文件的路径。
    """
    if dist.get_rank() == 0:  # 只有rank 0进程保存模型
        # 检查save_path中是否包含目录路径，并创建
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        
        # 保存模型
        torch.save(model.state_dict(), f"{save_path}_{epoch}.pth")
        print(f"Model saved at epoch {epoch}")

def load_model(model, load_path="model_checkpoint.pth"):
    """
    尝试加载模型状态。
    
    参数:
    - model: 要加载状态的模型。
    - load_path: 加载文件的路径，支持通配符。
    """
    if dist.get_rank() == 0:  # 只有rank 0进程尝试加载模型
        # 获取所有匹配的文件
        files = glob.glob(f"{load_path}_*.pth")
        if files:
            # 按epoch排序，选择最新的模型
            latest_file = max(files, key=os.path.getmtime)
            try:
                model.load_state_dict(torch.load(latest_file))
                print(f"Model loaded from {latest_file}")
                return latest_file
            except Exception as e:
                print(f"Error loading model: {e}")
        else:
            print(f"No checkpoint found at {load_path}")

def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch FashionMNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )

    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )

    parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,
        metavar="M",
        help="SGD momentum (default: 0.5)",
    )

    parser.add_argument(
        "--no-cuda",
        action="store_true",
        default=False,
        help="disables CUDA training",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        metavar="S",
        help="random seed (default: 1)",
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )

    parser.add_argument(
        "--dir",
        default="logs",
        metavar="L",
        help="directory where summary logs are stored",
    )

    parser.add_argument(
        "--backend",
        type=str,
        help="Distributed backend",
        choices=[dist.Backend.GLOO, dist.Backend.NCCL, dist.Backend.MPI],
        default=dist.Backend.GLOO,
    )

    parser.add_argument(
        "--dataset-mirror",
        type=str,
        default="",
        help="Dataset mirror",
    )

    parser.add_argument(
        "--ckpt-path",
        type=str,
        default="checkpoint/model_checkpoint",
        help="Checkpoint path",
    )

    parser.add_argument(
        "--use-powersdg-hook",
        action="store_true",
        default=False,
        help="use powerSDG hook",
    )

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        print("Using CUDA")
        if args.backend != dist.Backend.NCCL:
            print(
                "Warning. Please use `nccl` distributed backend for the best performance using GPUs"
            )

    writer = SummaryWriter(args.dir)

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    # Attach model to the device.
    model = Net().to(device)

    print("Using distributed PyTorch with {} backend".format(args.backend))
    # Set distributed training environment variables to run this training script locally.
    if "WORLD_SIZE" not in os.environ:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "1234"

    print(f"World Size: {os.environ['WORLD_SIZE']}. Rank: {os.environ['RANK']}")

    dist.init_process_group(backend=args.backend)

    model = nn.parallel.DistributedDataParallel(model)

    if args.use_powersdg_hook:
        state = powerSGD.PowerSGDState(process_group=None, matrix_approximation_rank=1, start_powerSGD_iter=10, min_compression_rate=0.5)
        model.register_comm_hook(state, powerSGD.powerSGD_hook)

    latest_checkpoint = load_model(model=model, load_path=args.ckpt_path)
    if latest_checkpoint:
        # 如果加载成功，可以从相应的epoch开始训练
        start_epoch = int(os.path.basename(latest_checkpoint).split("_")[-1].split(".")[0]) + 1
    else:
        # 如果没有找到模型或加载失败，从第一个epoch开始
        start_epoch = 1

    # Get FashionMNIST train and test dataset.
    # train_ds = datasets.FashionMNIST(
    train_ds = ProxyFashionMNIST(
        "./data",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
        mirrors = None if args.dataset_mirror == '' else [args.dataset_mirror],
    )
    # test_ds = datasets.FashionMNIST(
    test_ds = ProxyFashionMNIST(
        "./data",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
        mirrors = None if args.dataset_mirror == '' else [args.dataset_mirror],
    )
    # Add train and test loaders.
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=DistributedSampler(train_ds),
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.test_batch_size,
        sampler=DistributedSampler(test_ds),
    )

    for epoch in range(start_epoch, args.epochs + 1):
        train(args, model, device, train_loader, epoch, writer)
        # if args.save_model and dist.get_rank() ==0:
        #     print("torch save model, patch: {}".format(args.ckpt_path))
        #     torch.save(model.state_dict(), args.ckpt_path)
        test(model, device, test_loader, writer, epoch)
        save_model(model, epoch, save_path=args.ckpt_path)

if __name__ == "__main__":
    main()