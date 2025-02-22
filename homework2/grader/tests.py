import csv
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.utils.tensorboard as tb
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .grader import Case, Grader

# A different split will be used for grading
DATA_SPLIT = "classification_data/val"
INPUT_SHAPE = (1, 3, 64, 64)
LABEL_NAMES = ["background", "kart", "pickup", "nitro", "bomb", "projectile"]


def get_device() -> torch.device:
    if torch.cuda.is_available() and False:
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        warnings.warn(
            "No hardware acceleration found. Using CPU for grading.",
            category=RuntimeWarning,
            stacklevel=1,
        )
        device = torch.device("cpu")

    return device


class DummyFileWriter(tb.FileWriter):
    def __init__(self):
        self.events = []
        self.log_dir = None

    def add_event(self, e, step=None, walltime=None):
        self.events.append((e, step, walltime))


class DummySummaryWriter(tb.SummaryWriter):
    def __init__(self):
        self.log_dir = None
        self.file_writer = self.all_writers = None
        self._get_file_writer()

    def _get_file_writer(self):
        if self.file_writer is None:
            self.file_writer = DummyFileWriter()
            self.all_writers = {None: self.file_writer}
        return self.file_writer


class LogGrader(Grader):
    """Log correctness"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger = DummySummaryWriter()
        self.module.logger.test_logging(logger)
        self.events = logger.file_writer.events

    @staticmethod
    def get_val(events, tag):
        values = {}
        for e, s, _ in events:
            if e.HasField("summary"):
                for v in e.summary.value:
                    if v.tag == tag:
                        values[s] = v.simple_value
        return values

    @Case(score=5)
    def test_train(self):
        """Training loss and accuracy"""
        loss = self.get_val(self.events, "train_loss")
        for step in range(200):
            expect = 0.9 ** (step / 20.0)
            assert step in loss, f"No train_loss found for epoch={step // 20}, iteration={step % 20}"
            got = loss[step]
            assert abs(got - expect) < 1e-2, (
                f"train_loss epoch={step // 20}, iteration={step % 20} " f"expected {expect} got {got}"
            )
        acc = self.get_val(self.events, "train_accuracy")
        for epoch in range(10):
            torch.manual_seed(epoch)
            expect = epoch / 10.0 + torch.mean(torch.cat([torch.randn(10) for i in range(20)]))
            assert (
                20 * epoch + 19 in acc or 20 * epoch + 20 in acc
            ), f"No train_accuracy logging found for epoch {epoch}"
            got = acc[20 * epoch + 19] if 20 * epoch + 19 in acc else acc[20 * epoch + 20]
            assert abs(got - expect) < 1e-2, f"train_accuracy epoch={epoch} expected {expect} got {got}"

    @Case(score=5)
    def test_val(self):
        """Validation accuracy"""
        acc = self.get_val(self.events, "val_accuracy")
        for epoch in range(10):
            torch.manual_seed(epoch)
            expect = epoch / 10.0 + torch.mean(torch.cat([torch.randn(10) for i in range(10)]))
            assert 20 * epoch + 19 in acc or 20 * epoch + 20 in acc, f"No val_accuracy logging found for epoch {epoch}"
            got = acc[20 * epoch + 19] if 20 * epoch + 19 in acc else acc[20 * epoch + 20]
            assert abs(got - expect) < 1e-2, f"val_accuracy epoch={epoch} expected {expect} got {got}"


@dataclass
class TracerStats:
    num_linear_in_shortest_path: int = 0
    num_linear_relu_in_longest_path: int = 0
    has_relu: bool = False

    @classmethod
    def trace(cls, model, input_sample):
        visited = {}
        unknown_layers = set()

        def graph_tracer(node):
            if node in visited:
                # print("DUP", node.kind(), visited[node])
                return visited[node]
            visited[node] = TracerStats(100000, 0, False)

            if node.kind() in ["prim::Param"]:
                return TracerStats(0, 0, True)
            elif node.kind() in ["aten::linear", "aten::_convolution"]:
                stats = graph_tracer(list(node.inputs())[0].node())
                visited[node] = TracerStats(
                    stats.num_linear_in_shortest_path + 1,
                    stats.num_linear_relu_in_longest_path + int(stats.has_relu),
                    False,
                )
                return visited[node]
            elif node.kind() in ["aten::relu_", "aten::relu", "aten::gelu_", "aten::gelu", "aten::elu_", "aten::elu"]:
                stats = graph_tracer(list(node.inputs())[0].node())
                visited[node] = TracerStats(
                    stats.num_linear_in_shortest_path, stats.num_linear_relu_in_longest_path, True
                )
                return visited[node]
            elif node.kind() in ["aten::add_", "aten::add", "aten::cat"]:
                all_stats = [graph_tracer(i.node()) for i in node.inputs() if i.node().kind() != "prim::Constant"]
                visited[node] = TracerStats(
                    min(s.num_linear_in_shortest_path for s in all_stats),
                    max(s.num_linear_relu_in_longest_path for s in all_stats),
                    True,
                )
                return visited[node]
            elif list(node.inputs()):
                unknown_layers.add(node.kind())
                visited[node] = graph_tracer(list(node.inputs())[0].node())
                return visited[node]
            return visited[node]

        traced_model = torch.jit.trace(model, input_sample)
        graph = traced_model.graph
        torch._C._jit_pass_inline(graph)
        stats = graph_tracer(graph.return_node())
        # print(unknown_layers)
        return stats


def is_linear(model: torch.nn.Module) -> bool:
    torch.manual_seed(0)

    a = torch.rand(*INPUT_SHAPE)
    b = torch.ones(*INPUT_SHAPE)
    bias = model(torch.zeros_like(a))

    return torch.allclose(model(a + b), model(a) + model(b) - bias, atol=1e-3)


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path: str):
        """
        Pairs of images and labels (int) for classification
        You won't need to modify this, but all PyTorch datasets must implement these methods
        """
        to_tensor = transforms.ToTensor()

        self.data = []

        with open(Path(dataset_path, "labels.csv"), newline="") as f:
            for fname, label, _ in csv.reader(f):
                if label in LABEL_NAMES:
                    image = Image.open(Path(dataset_path, fname))
                    label_id = LABEL_NAMES.index(label)

                    self.data.append((to_tensor(image), label_id))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_data(dataset_path: str, num_workers: int = 0, batch_size: int = 128, shuffle: bool = False) -> DataLoader:
    return DataLoader(
        SuperTuxDataset(dataset_path), num_workers=num_workers, batch_size=batch_size, shuffle=shuffle, drop_last=True
    )


class ClassificationGrader(Grader):
    """Classification"""

    @Case(score=10, timeout=500)
    def test_classification_loss(self):
        """
        Classification Loss
        """
        loss_func = self.module.models.ClassificationLoss()
        logits = torch.FloatTensor(
            [
                [-1.0, 1.0, 0.0],
                [-1.0, -1.0, 2.0],
            ]
        )
        labels = torch.LongTensor([1, 0])
        loss = loss_func(logits, labels)
        loss_expected = torch.FloatTensor([1.7513])

        assert torch.allclose(loss, loss_expected, atol=1e-3)


class LinearGrader(Grader):
    """Linear"""

    KIND = "linear"
    ACC_RANGE = 0.4, 0.7

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.data = list(load_data(DATA_SPLIT, num_workers=2))
        self.device = get_device()

    @torch.inference_mode()
    def accuracy(
        self, model: torch.nn.Module, min_accuracy: float = 0.5, max_accuracy: float = 1.0
    ) -> tuple[float, str]:
        """
        Returns the accuracy of the model normalized to [0, 1]

        If the accuracy is greater than max_accuracy, you get 1.0 (full score)
        Similarly, if the model's accuracy less than min_accuracy, you get 0.0 (no points)
        """
        model.to(self.device)

        accuracy_list = []

        for img, label in self.data:
            pred = model(img.to(self.device)).detach().cpu()
            acc = (pred.argmax(dim=1).type_as(label) == label).float()
            accuracy_list.extend(acc.numpy())

        accuracy = torch.as_tensor(accuracy_list).mean()
        accuracy_normalized = (accuracy - min_accuracy) / (max_accuracy - min_accuracy)
        msg = f"Accuracy: {accuracy.item():0.2f}, Full Credit @ {max_accuracy:0.2f}"

        return accuracy_normalized.clip(0, 1).item(), msg

    def check_model(self, model):
        assert is_linear(model), "Model is not linear"

    @Case(score=5, timeout=500)
    def test_model(self):
        """Model"""
        model = self.module.load_model(self.KIND, with_weights=False)
        self.check_model(model)

    @Case(score=15, timeout=10000)
    def test_accuracy(self):
        """Accuracy"""
        model = self.module.load_model(self.KIND, with_weights=True)
        self.check_model(model)
        return self.accuracy(model, *self.ACC_RANGE)

    @Case(score=1, timeout=10000, extra_credit=True)
    def test_accuracy_extra(self):
        """Accuracy: Extra Credit"""
        model = self.module.load_model(self.KIND, with_weights=True)
        self.check_model(model)
        # Extra credit if accuracy is 2 points higher than the regular test (rounded)
        return self.accuracy(model, self.ACC_RANGE[1], self.ACC_RANGE[1] + 0.04)


class MLPGrader(LinearGrader):
    """MLP"""

    KIND = "mlp"
    ACC_RANGE = 0.5, 0.8

    def check_model(self, model):
        stats = TracerStats.trace(model, torch.rand(*INPUT_SHAPE))
        assert stats.num_linear_relu_in_longest_path > 1, "Model not deep enough"
        assert (
            stats.num_linear_relu_in_longest_path == stats.num_linear_in_shortest_path
        ), "Model may contain a residual connection"


class DeepMLPGrader(LinearGrader):
    """Deep MLP"""

    KIND = "mlp_deep"
    ACC_RANGE = 0.5, 0.8
    MIN_LAYERS = 4

    def check_model(self, model):
        stats = TracerStats.trace(model, torch.rand(*INPUT_SHAPE))
        assert stats.num_linear_relu_in_longest_path >= self.MIN_LAYERS, "Model not deep enough"
        assert (
            stats.num_linear_relu_in_longest_path == stats.num_linear_in_shortest_path
        ), "Model may contain a residual connection"


class DeepResidualMLPGrader(LinearGrader):
    """Deep Residual MLP"""

    KIND = "mlp_deep_residual"
    ACC_RANGE = 0.5, 0.8
    MIN_LAYERS = 4

    def check_model(self, model):
        stats = TracerStats.trace(model, torch.rand(*INPUT_SHAPE))
        assert stats.num_linear_relu_in_longest_path >= self.MIN_LAYERS, "Model not deep enough"
        assert (
            stats.num_linear_relu_in_longest_path != stats.num_linear_in_shortest_path
        ), "Model should contain a residual connection"
