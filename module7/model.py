import torch

class ConvNet(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, in_channels: int, out_channels: int, stride: int):
            super().__init__()
            kernel_size = 3
            padding = (kernel_size-1)//2

            self.c1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            self.n1 = torch.nn.GroupNorm(1, out_channels)
            self.c2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)
            self.n2 = torch.nn.GroupNorm(1, out_channels)
            self.relu1 = torch.nn.ReLU()
            self.relu2 = torch.nn.ReLU()

            self.skip = torch.nn.Conv2d(in_channels, out_channels, 1, stride, 0) if in_channels != out_channels else torch.nn.Identity()



        def forward(self, x0):
            x = self.relu1(self.n1(self.c1(x0)))
            x = self.relu2(self.n2(self.c2(x)))
            return self.skip(x0) + x


    def __init__(self, channels_l0 = 64, n_blocks = 4):
        super().__init__()
        cnn_layers: list[torch.nn.Module] = [
            torch.nn.Conv2d(3, channels_l0, kernel_size=11, stride=2, padding=5),
            torch.nn.ReLU(),
        ]
        c1 = channels_l0
        for _ in range(n_blocks):
            c2 = c1 * 2
            cnn_layers.append(self.Block(in_channels=c1, out_channels=c2, stride=2))
            c1 = c2
        cnn_layers.append(torch.nn.Conv2d(c1, 102, kernel_size=1))
        self.network = torch.nn.Sequential(*cnn_layers)

    def forward(self, x):
        return self.network(x).mean(dim=-1).mean(dim=-1)
