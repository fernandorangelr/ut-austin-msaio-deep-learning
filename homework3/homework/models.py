from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class ClassificationLoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.weight = weight

    def forward(self, logits: torch.Tensor, target: torch.LongTensor) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(logits, target, weight=self.weight)



class RegressionLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.LongTensor) -> torch.Tensor:
        """
        Regression loss

        Args:
            logits: tensor (b, c) logits, where c is the number of classes
            target: tensor (b,) labels

        Returns:
            tensor, scalar loss
        """
        return torch.nn.functional.l1_loss(logits, target)


class Classifier(nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, in_channels: int, out_channels: int, stride: int):
            super().__init__()
            kernel_size = 3
            padding = (kernel_size - 1) // 2
            self.model = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(),
            )  # Add a layer before the residual connection

            # Validate the number of input channels matches the number of output channels for the residual connections
            if in_channels != out_channels:
                self.skip = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                            padding)  # Add a convolutional layer to change the shape and match the output
            else:
                self.skip = torch.nn.Identity()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x) + self.skip(x)  # By adding `x`, we have added a residual connection

    def __init__(
            self,
            in_channels: int = 3,
            num_classes: int = 6,
            channels_l0: int = 64,
            n_blocks: int = 3
    ):
        """
        A convolutional network for image classification.

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        cnn_layers = [
            torch.nn.Conv2d(in_channels, channels_l0, kernel_size=11, stride=2, padding=5),
            torch.nn.ReLU()
        ]
        c1 = channels_l0

        for _ in range(n_blocks):
            c2 = c1 * 2
            cnn_layers.append(self.Block(c1, c2, stride=2))
            c1 = c2

        cnn_layers.append(torch.nn.Conv2d(c1, num_classes, kernel_size=1))
        cnn_layers.append(torch.nn.AdaptiveAvgPool2d(1))  # Pool everything together and average the outputs
        self.network = torch.nn.Sequential(*cnn_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, h, w) image

        Returns:
            tensor (b, num_classes) logits
        """
        # optional: normalizes the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        logits = self.network(z)  # shape: (b, num_classes, 1, 1)

        return logits.squeeze(-1).squeeze(-1)  # shape: (b, num_classes)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Used for inference, returns class labels
        This is what the AccuracyMetric uses as input (this is what the grader will use!).
        You should not have to modify this function.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            pred (torch.LongTensor): class labels {0, 1, ..., 5} with shape (b, h, w)
        """
        return self(x).argmax(dim=1)


class Detector(torch.nn.Module):
    class DownBlock(torch.nn.Module):
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.model = torch.nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),  # downsample
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            )  # Add a layer before the residual connection

            self.skip = nn.Identity()
            if in_channels != out_channels:
                self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x) + self.skip(x)  # By adding `x`, we have added a residual connection

    class UpBlock(torch.nn.Module):
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            self.conv = nn.Sequential(
                nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )

        def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
            x = self.up(x)
            if x.shape != skip.shape:
                diffY = skip.size(2) - x.size(2)
                diffX = skip.size(3) - x.size(3)
                x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2])
            x = torch.cat([x, skip], dim=1)
            return self.conv(x)

    def __init__(
            self,
            in_channels: int = 3,
            num_classes: int = 3,
            channels_l0=48,
            depth=4
    ):
        """
        A single model that performs segmentation and depth regression

        Args:
            in_channels: int, number of input channels
            num_classes: int
            channels_l0: int
            n_blocks: int, must be an even number
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.skip_channels = []

        c_in = in_channels
        for i in range(depth):
            c_out = channels_l0 * (2 ** i)
            self.down_blocks.append(self.DownBlock(c_in, c_out))
            self.skip_channels.append(c_out)
            c_in = c_out

        for i in reversed(range(depth - 1)):
            c_out = channels_l0 * (2 ** i)
            self.up_blocks.append(self.UpBlock(c_in, c_out))
            c_in = c_out

        self.final_up = nn.ConvTranspose2d(c_in, channels_l0, kernel_size=2, stride=2)

        self.segmentation_head = nn.Sequential(
            nn.Conv2d(channels_l0, channels_l0, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels_l0),
            nn.ReLU(),
            nn.Conv2d(channels_l0, channels_l0, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels_l0),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.Conv2d(channels_l0, num_classes, kernel_size=1)
        )

        self.depth_head = nn.Sequential(
            nn.Conv2d(channels_l0, channels_l0, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels_l0, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used in training, takes an image and returns raw logits and raw depth.
        This is what the loss functions use as input.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.FloatTensor, torch.FloatTensor):
                - logits (b, num_classes, h, w)
                - depth (b, h, w)
        """
        # optional: normalizes the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        skips = []
        out = z

        for down in self.down_blocks:
            out = down(out)
            skips.append(out)

        out = skips.pop()

        for up in self.up_blocks:
            skip = skips.pop()
            out = up(out, skip)

        out = self.final_up(out)

        logits = self.segmentation_head(out)
        depth = self.depth_head(out).squeeze(1)

        return logits, depth

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used for inference, takes an image and returns class labels and normalized depth.
        This is what the metrics use as input (this is what the grader will use!).

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.LongTensor, torch.FloatTensor):
                - pred: class labels {0, 1, 2} with shape (b, h, w)
                - depth: normalized depth [0, 1] with shape (b, h, w)
        """
        logits, raw_depth = self(x)
        pred = logits.argmax(dim=1)

        # Optional additional post-processing for depth only if needed
        depth = raw_depth

        return pred, depth


MODEL_FACTORY = {
    "classifier": Classifier,
    "detector": Detector,
}


def load_model(
        model_name: str,
        with_weights: bool = False,
        **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def debug_model(batch_size: int = 1):
    """
    Test your model implementation

    Feel free to add additional checks to this function -
    this function is NOT used for grading
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_batch = torch.rand(batch_size, 3, 64, 64).to(device)

    print(f"Input shape: {sample_batch.shape}")

    model = load_model("classifier", in_channels=3, num_classes=6).to(device)
    output = model(sample_batch)

    # should output logits (b, num_classes)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    debug_model()
