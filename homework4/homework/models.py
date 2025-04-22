import inspect
from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


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


class MLPPlanner(nn.Module):
    """
    MLP-based planner that predicts future waypoints from track boundaries.
    """

    class Block(nn.Module):
        """
        A single MLP block: Linear -> ReLU
        """

        def __init__(self, in_dim: int, out_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    def __init__(
            self,
            n_track: int = 10,
            n_waypoints: int = 3,
            hidden_dim: int = 128,
            num_layers: int = 4,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        input_dim = n_track * 2 * 2  # left+right, each has n_track x 2 coords
        layers = []
        in_dim = input_dim

        for _ in range(num_layers):
            layers.append(self.Block(in_dim, hidden_dim))
            in_dim = hidden_dim

        # final layer: no ReLU, no dropout
        layers.append(nn.Linear(hidden_dim, n_waypoints * 2))
        self.mlp = nn.Sequential(*layers)

    def forward(
            self,
            track_left: torch.Tensor,
            track_right: torch.Tensor,
            **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        # track_left/right: (B, n_track, 2)
        B = track_left.size(0)
        x = torch.cat([
            track_left.view(B, -1),
            track_right.view(B, -1)
        ], dim=1)
        out = self.mlp(x)  # (B, n_waypoints*2)
        return out.view(B, self.n_waypoints, 2)


class TransformerPlanner(nn.Module):
    """
    Transformer-based planner using a TransformerDecoder over boundary embeddings.
    """

    class DecoderBlock(nn.Module):
        """
        Wrapper for a single TransformerDecoderLayer
        """

        def __init__(self, d_model: int, nhead: int, dim_feedforward: int):
            super().__init__()
            self.layer = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward
            )

        def forward(self, tgt, memory):
            return self.layer(tgt, memory)

    def __init__(
            self,
            n_track: int = 10,
            n_waypoints: int = 3,
            d_model: int = 128,
            nhead: int = 8,
            num_layers: int = 4,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        self.point_encoder = nn.Linear(2, d_model)
        self.query_embed = nn.Embedding(n_waypoints, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model = d_model,
            nhead = nhead,
            dim_feedforward = 4 * d_model,
            dropout = 0.1,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers = num_layers,
        )
        self.output_proj = nn.Linear(d_model, 2)

    def forward(
            self,
            track_left: torch.Tensor,
            track_right: torch.Tensor,
            **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        B = track_left.size(0)

        x = torch.cat([track_left, track_right], dim=1)  # (B, 2*n_track, 2)

        memory = self.point_encoder(x)  # (B, 2*n_track, d_model)
        memory = memory.permute(1, 0, 2)  # (S=2*n_track, B, d_model)

        idx = torch.arange(self.n_waypoints, device=x.device)
        tgt = self.query_embed(idx)  # (T, d_model)
        tgt = tgt.unsqueeze(1).repeat(1, B, 1)  # (T, B, d_model)

        out = self.transformer_decoder(tgt=tgt, memory=memory)  # (T, B, d_model)

        out = self.output_proj(out)  # (T, B, 2)

        return out.permute(1, 0, 2).contiguous()


class CNNPlanner(torch.nn.Module):
    def __init__(
            self,
            n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        # convolutional backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # compute flattened feature size:
        H, W = (96,128)
        fh, fw = H // 16, W // 16  # 4×stride‑2 reductions
        feat_dim = 128 * fh * fw

        self.head = nn.Sequential(
            nn.Flatten(),  # (B, feat_dim)
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, n_waypoints * 2),
        )

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        x = image
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        feats = self.backbone(x)
        out = self.head(feats)
        B = image.size(0)
        return out.view(B, self.n_waypoints, 2)


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
        model_name: str,
        with_weights: bool = False,
        **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    cls = MODEL_FACTORY[model_name]
    sig = inspect.signature(cls.__init__)
    valid = set(sig.parameters.keys()) - {'self'}
    filtered = {k: v for k, v in model_kwargs.items() if k in valid}
    m = MODEL_FACTORY[model_name](**filtered)

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
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
