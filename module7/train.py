import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from fire import Fire
from model import ConvNet
import numpy as np


def train():
    ## Let's setup the dataloaders
    size = (128, 128)
    train_transform = torchvision.transforms.Compose([torchvision.transforms.RandomResizedCrop(size=size, antialias=True),
                                                      torchvision.transforms.RandomHorizontalFlip(),
                                                      torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

    valid_transform = torchvision.transforms.Compose([torchvision.transforms.Resize(size),
                                                      torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_dataset = torchvision.datasets.Flowers102("./flowers", "train", transform=train_transform, download=True)
    valid_dataset = torchvision.datasets.Flowers102("./flowers", "val", transform=valid_transform, download=True)
    # test_dataset = torchvision.datasets.Flowers102("./flowers", "test", transform=transform, download=True)

    writer = SummaryWriter(log_dir="./logs")
    writer.add_graph(ConvNet(channels_l0=32, n_blocks=4), torch.zeros(1, 3, *size))
    writer.add_images("train_images", torch.stack([train_dataset[i][0] for i in range(32)]))
    writer.flush()

    net = ConvNet(channels_l0=32, n_blocks=4)
    if torch.cuda.is_available():
        net.cuda()
    optim = torch.optim.AdamW(net.parameters(), lr=0.005, weight_decay=1e-4)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=8)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=256, num_workers=8)

    global_step = 0
    for epoch in range(500):

        net.train()
        train_accuracy = []
        for data, label in train_loader:
            if torch.cuda.is_available():
                data, label = data.cuda(), label.cuda()
            output = net(data)
            loss = torch.nn.functional.cross_entropy(output, label)

            train_accuracy.extend((output.argmax(dim=-1) == label).cpu().detach().float().numpy())

            optim.zero_grad()
            loss.backward()
            optim.step()

            writer.add_scalar("train/loss", loss.item(), global_step=global_step)
            global_step += 1

        writer.add_scalar("train/accuracy", np.mean(train_accuracy), epoch)

        net.eval()
        valid_accuracy = []
        for data, label in valid_loader:
            if torch.cuda.is_available():
                data, label = data.cuda(), label.cuda()
            with torch.inference_mode():
                output = net(data)

            valid_accuracy.extend((output.argmax(dim=-1) == label).cpu().detach().float().numpy())

        writer.add_scalar("valid/accuracy", np.mean(valid_accuracy), epoch)

        writer.flush()

        ## Early stopping
        if epoch % 10 == 0:
            torch.save(net.state_dict(), f"model_{epoch}.pth")


if __name__ == "__main__":
    Fire(train)