from typing import Callable

import torch

from .grader import Case, CheckFailed, Grader
from .slow_pytorch_basics import PyTorchBasics as SlowPyTorchBasics


def assert_close(a, b):
    if not isinstance(a, torch.Tensor):
        raise CheckFailed(f"Expected torch.Tensor, got {type(a)}")
    elif not isinstance(b, torch.Tensor):
        raise CheckFailed(f"Expected torch.Tensor, got {type(b)}")
    elif a.shape != b.shape:
        raise CheckFailed(f"Expected shape {b.shape}, got {a.shape}")
    elif not torch.isclose(a, b).all():
        raise CheckFailed(f"Expected {b}, got {a}")


def assert_differentiable(x, y):
    y.mean().backward()

    assert x.grad is not None, "No gradient found."


class PyTorchBasics(Grader):
    """Make it PyTorch"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.answers = self.module.pytorch_basics.PyTorchBasics

    def test_fn(self, fname, *inputs, test_grad=True):
        import warnings

        from torch.jit import TracerWarning

        warnings.filterwarnings("ignore", category=TracerWarning)
        answer_fn = getattr(self.answers, fname)
        slow_fn = getattr(SlowPyTorchBasics, fname)

        t_fun = None

        for inp in inputs:
            # Make sure inputs are tuples
            if not isinstance(inp, (tuple, list)):
                inp = (inp,)

            # Compute the result
            a = answer_fn(*inp)
            b = slow_fn(*inp)

            if not isinstance(a, torch.Tensor):
                raise CheckFailed(f"Expected torch.Tensor, got {type(a)}")

            if a.shape != b.shape:
                raise CheckFailed(f"Expected shape {b.shape}, got {a.shape}")
            if abs(a - b).sum() > 1e-3:
                raise CheckFailed(f"Expected output:\n{b}\nGot:\n{a}\nInput:\n{inp}")

            if t_fun is None:
                t_fun = torch.jit.trace(answer_fn, inp, check_trace=False)

            try:
                a = t_fun(*inp)
            except Exception:
                raise CheckFailed("Function not general enough (might contain loops, special indexing, etc.)") from None

            if a.shape != b.shape:
                raise CheckFailed("Function not general enough (might contain loops, special indexing, etc.)")
            if abs(a - b).sum() > 1e-3:
                raise CheckFailed("Function not general enough (might contain loops, special indexing, etc.)")

    @Case(score=3, timeout=500)
    def test_make_it_pytorch_1(self):
        """Make it pytorch 1"""
        cases = [torch.arange(n).float() for n in [10, 20, 50, 100, 1, 2, 3, 4, 99]]
        self.test_fn("make_it_pytorch_1", *cases)

    @Case(score=3, timeout=500)
    def test_make_it_pytorch_2(self):
        """Make it pytorch 2"""
        g = torch.Generator().manual_seed(2147483647)
        cases = [torch.randn(n // 2, n, n + 4, generator=g) for n in range(4, 20)]
        self.test_fn("make_it_pytorch_2", *cases)

    @Case(score=3, timeout=1000)
    def test_make_it_pytorch_3(self):
        """Make it pytorch 3"""
        g = torch.Generator().manual_seed(2147483647)
        cases = [(torch.randn(n // 2, n, n + 4, generator=g) * 5).int() for n in range(4, 20)]
        self.test_fn("make_it_pytorch_3", *cases)

    @Case(score=3, timeout=500)
    def test_make_it_pytorch_4(self):
        """Make it pytorch 4"""
        g = torch.Generator().manual_seed(2147483647)
        cases = [
            (torch.randn(n // 2, n, n + 4, generator=g), torch.randn(n * 2, n + 4, generator=g)) for n in range(4, 20)
        ]
        self.test_fn("make_it_pytorch_4", *cases)

    @Case(score=3, timeout=500)
    def test_make_it_pytorch_5(self):
        """Make it pytorch 5"""
        g = torch.Generator().manual_seed(2147483647)
        cases = [torch.randn(n, abs(10 - n) + 1, generator=g) for n in range(4, 20)]
        self.test_fn("make_it_pytorch_5", *cases)

    @Case(score=3, timeout=500)
    def test_make_it_pytorch_6(self):
        """Make it pytorch 6"""
        g = torch.Generator().manual_seed(2147483647)
        cases = [torch.randn(n, abs(10 - n) + 1, generator=g) for n in range(4, 20)]
        self.test_fn("make_it_pytorch_6", *cases)

    @Case(score=3, timeout=500)
    def test_make_it_pytorch_7(self):
        """Make it pytorch 7"""
        g = torch.Generator().manual_seed(2147483647)
        cases = [torch.randn(n, abs(10 - n) + 1, generator=g) for n in range(4, 20)]
        self.test_fn("make_it_pytorch_7", *cases)

    @Case(score=3, timeout=500)
    def test_make_it_pytorch_8(self):
        """Make it pytorch 8"""
        g = torch.Generator().manual_seed(2147483647)
        cases = [torch.randn(n, generator=g) for n in torch.linspace(1, 6, 10).int()]
        self.test_fn("make_it_pytorch_8", *cases)

    @Case(score=4, timeout=500)
    def test_make_it_pytorch_9(self):
        """Make it pytorch 9"""
        g = torch.Generator().manual_seed(2147483647)
        cases = [torch.randn(n, 2 * n, generator=g) for n in torch.linspace(1, 6, 10).int()]
        self.test_fn("make_it_pytorch_9", *cases)

    @Case(score=4, timeout=500)
    def test_make_it_pytorch_10(self):
        """Make it pytorch 10"""
        g = torch.Generator().manual_seed(2147483647)
        cases = [
            (torch.randn(n, 2 * n, generator=g), torch.randn(1, generator=g)) for n in torch.linspace(1, 6, 10).int()
        ]
        self.test_fn("make_it_pytorch_10", *cases)

    @Case(score=4, timeout=500)
    def test_make_it_pytorch_11(self):
        """Make it pytorch 11"""
        g = torch.Generator().manual_seed(2147483647)
        cases = [
            (torch.randn(n, 2 * n, generator=g), torch.randn(1, generator=g)) for n in torch.linspace(1, 6, 10).int()
        ]
        self.test_fn("make_it_pytorch_11", *cases)

    @Case(score=4, timeout=500)
    def test_make_it_pytorch_12(self):
        """Make it pytorch 12"""
        g = torch.Generator().manual_seed(2147483647)
        cases = [
            (torch.randn(n, 2 * n, generator=g), torch.randn(n, 2 * n, generator=g) > 0)
            for n in torch.linspace(1, 6, 10).int()
        ]
        self.test_fn("make_it_pytorch_12", *cases)

    @Case(score=1, timeout=500, extra_credit=True)
    def make_it_pytorch_extra_1(self):
        """Make it pytorch extra 1"""
        g = torch.Generator().manual_seed(2147483647)
        cases = [(torch.randn(n, generator=g), torch.randn(2 * n, generator=g)) for n in torch.linspace(1, 6, 10).int()]
        self.test_fn("make_it_pytorch_extra_1", *cases)

    @Case(score=1, timeout=500, extra_credit=True)
    def make_it_pytorch_extra_2(self):
        """Make it pytorch extra 2"""

        def make_example(n, g) -> tuple[torch.Tensor, torch.Tensor]:
            x = torch.randn(n, generator=g)
            y = torch.randn(2 * n, generator=g)
            m = torch.randn(n, generator=g) < 0
            y[:n][m] = x[m]
            return x, y

        g = torch.Generator().manual_seed(2147483647)
        cases = [make_example(n, g) for n in torch.linspace(1, 6, 10).int()]
        self.test_fn("make_it_pytorch_extra_2", *cases)


class NearestNeighborGrader(Grader):
    """NN Grader"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.nn = self.module.nearest_neighbor_classifier.NearestNeighborClassifier

    def test_fn(
        self,
        fname: str | Callable,
        inputs,
        outputs: list[torch.Tensor],
    ):
        if isinstance(fname, str):
            answer_fn = getattr(self.nn, fname)
        else:
            answer_fn = fname

        for inp, out in zip(inputs, outputs, strict=True):
            if not isinstance(inp, (tuple, list)):
                inp = (inp,)

            a = answer_fn(*inp)
            b = out

            if isinstance(b, tuple):
                if not isinstance(a, tuple):
                    raise CheckFailed(f"Expected tuple, got {type(a)}")
            else:
                a = (a,)
                b = (b,)

            if len(a) != len(b):
                raise CheckFailed(f"Expected {len(b)} outputs, got {len(a)}")

            for _a, _b in zip(a, b):
                if not isinstance(_a, torch.Tensor):
                    raise CheckFailed(f"Expected torch.Tensor, got {type(_a)}")
                if _a.dtype != _b.dtype:
                    raise CheckFailed(f"Expected dtype {_b.dtype}, got {_a.dtype}")
                if _a.shape != _b.shape:
                    raise CheckFailed(f"Expected shape {_b.shape}, got {_a.shape}")
                if abs(_a - _b).sum() > 1e-3:
                    raise CheckFailed(f"Expected\nOutput:\n{_b}\nGot:\n{_a}\nInputs:\n{inp}\n")

    @Case(score=2, timeout=100)
    def test_1_make_data_single_element(self):
        """Test with single-element input lists."""
        inputs = [
            ([[1.0]], [0.0]),
            ([[-0.5]], [1.0]),
            ([[1000.0]], [0.0]),
        ]

        outputs = [
            (torch.FloatTensor([[1.0]]), torch.FloatTensor([0.0])),
            (torch.FloatTensor([[-0.5]]), torch.FloatTensor([1.0])),
            (torch.FloatTensor([[1000.0]]), torch.FloatTensor([0.0])),
        ]

        self.test_fn("make_data", inputs, outputs)

    @Case(score=3, timeout=100)
    def test_2_make_data_multiple_elements(self):
        """Test with multiple elements in input lists."""

        inputs = [
            ([[1.0, 2.0, 3.0]], [0.0, 0.0, 1.0]),
            ([[-1.0, 0.0, 10.0]], [1.0, 1.0, 0.0]),
            ([[0.1, 0.2, 0.3, 0.4, 0.5]], [1.0, 1.0, 0.0, 0.0, 1.0]),
        ]

        outputs = [
            (torch.FloatTensor([[1.0, 2.0, 3.0]]), torch.FloatTensor([0.0, 0.0, 1.0])),
            (torch.FloatTensor([[-1.0, 0.0, 10.0]]), torch.FloatTensor([1.0, 1.0, 0.0])),
            (torch.FloatTensor([[0.1, 0.2, 0.3, 0.4, 0.5]]), torch.FloatTensor([1.0, 1.0, 0.0, 0.0, 1.0])),
        ]

        self.test_fn("make_data", inputs, outputs)

    @Case(score=5, timeout=100)
    def test_3_compute_data_statistics_simple(self):
        """2D tensor statistics with different shapes."""

        inputs = [
            (torch.tensor([[0.0, 0.0], [0.0, 0.0]])),
            (torch.tensor([[1.0, 2.0], [4.0, 5.0], [7.0, 8.0]])),
            (torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])),
        ]

        outputs = [
            (torch.tensor([[0.0, 0.0]]), torch.tensor([[0.0, 0.0]])),
            (torch.tensor([[4.0, 5.0]]), torch.tensor([[3.0, 3.0]])),
            (torch.tensor([[2.5, 3.5, 4.5]]), torch.tensor([[2.1213, 2.1213, 2.1213]])),
        ]

        self.test_fn("compute_data_statistics", inputs, outputs)

    @Case(score=5)
    def test_4_input_normalization(self):
        """Input normalization"""

        def callback(_x, _y):
            nn = self.nn(_x, _y)
            ret = nn.input_normalization(_x)

            return ret

        inputs = [
            (torch.FloatTensor([[0.0, 1.0], [-2.0, -2.0], [-1.0, 0.0]]), torch.FloatTensor([0.0, 1.0, 0.0])),
            (torch.FloatTensor([[1.0], [-1.0], [0.0], [-3.0]]), torch.FloatTensor([0.0, 0.0, 1.0, 1.0])),
            (
                torch.FloatTensor([[3.0, -3.0], [-2.0, 1.0], [4.0, -1.0], [3.0, 0.0], [0.0, 4.0]]),
                torch.FloatTensor([0.0, 0.0, 0.0, 0.0, 1.0]),
            ),
        ]

        outputs = [
            torch.FloatTensor([[1.0, 0.8729], [-1.0, -1.0911], [0.0, 0.2182]]),
            torch.FloatTensor([[1.0247], [-0.1464], [0.4392], [-1.3175]]),
            torch.FloatTensor(
                [[0.5578, -1.2363], [-1.4343, 0.3091], [0.9562, -0.4636], [0.5578, -0.0773], [-0.6375, 1.4681]]
            ),
        ]

        self.test_fn(callback, inputs, outputs)

    @Case(score=5)
    def test_5_nearest_neighbors(self):
        """Nearest Neighbors"""

        def callback(_x, _y, _z):
            nn = self.nn(_x, _y)
            ret = nn.get_nearest_neighbor(_z)

            return ret

        inputs = [
            (
                torch.FloatTensor([[2.0, -1.0], [0.0, 1.0], [-3.0, 2.0], [0.0, 2.0]]),
                torch.FloatTensor([1.0, 0.0, 1.0, 1.0]),
                torch.FloatTensor([[1.0, -1.0]]),
            ),
            (
                torch.FloatTensor([[-1.0], [-3.0], [0.0], [-2.0], [-1.0]]),
                torch.FloatTensor([1.0, 1.0, 1.0, 0.0, 1.0]),
                torch.FloatTensor([[1.0]]),
            ),
        ]

        outputs = [
            (torch.FloatTensor([2.0, -1.0]), torch.as_tensor(1.0)),
            (torch.FloatTensor([0.0]), torch.as_tensor(1.0)),
        ]

        self.test_fn(callback, inputs, outputs)

    @Case(score=5)
    def test_6_knn(self):
        """K-Nearest Neighbors"""

        def callback(_x, _y, _z, _k):
            nn = self.nn(_x, _y)
            ret = nn.get_k_nearest_neighbor(_z, _k)

            return ret

        inputs = [
            (
                torch.FloatTensor([[0.0, 1.0], [-2.0, -2.0], [-1.0, 0.0]]),
                torch.FloatTensor([0.0, 1.0, 0.0]),
                torch.FloatTensor([[0.0, 0.0]]),
                3,
            ),
            (
                torch.FloatTensor([[1.0, 2.0], [-2.0, -1.0], [1.0, 3.0], [0.0, 2.0]]),
                torch.FloatTensor([0.0, 1.0, 1.0, 0.0]),
                torch.FloatTensor([[-2.0, -1.0]]),
                3,
            ),
            (
                torch.FloatTensor(
                    [[4.0, 2.0, 2.0], [0.0, -2.0, 0.0], [-4.0, -3.0, -2.0], [3.0, 2.0, -2.0], [0.0, 3.0, 4.0]]
                ),
                torch.FloatTensor([1.0, 1.0, 1.0, 0.0, 1.0]),
                torch.FloatTensor([[-3.0, -3.0, -2.0]]),
                1,
            ),
        ]

        outputs = [
            (torch.FloatTensor([[0.0, 1.0], [-1.0, 0.0], [-2.0, -2.0]]), torch.FloatTensor([0.0, 0.0, 1.0])),
            (torch.FloatTensor([[-2.0, -1.0], [0.0, 2.0], [1.0, 2.0]]), torch.FloatTensor([1.0, 0.0, 0.0])),
            (torch.FloatTensor([[-4.0, -3.0, -2.0]]), torch.FloatTensor([1.0])),
        ]

        self.test_fn(callback, inputs, outputs)

    @Case(score=5)
    def test_7_knn_regression(self):
        """K-Nearest Neighbors Regression"""

        def callback(_x, _y, _z, _k):
            nn = self.nn(_x, _y)
            ret = nn.knn_regression(_z, _k)

            return ret

        inputs = [
            (
                torch.FloatTensor([[0.0, 1.0], [-2.0, -2.0], [-1.0, 0.0]]),
                torch.FloatTensor([0.0, 1.0, 0.0]),
                torch.FloatTensor([[0.0, 0.0]]),
                3,
            ),
            (
                torch.FloatTensor([[1.0, 2.0], [-2.0, -1.0], [1.0, 3.0], [0.0, 2.0]]),
                torch.FloatTensor([0.0, 1.0, 1.0, 0.0]),
                torch.FloatTensor([[-2.0, -1.0]]),
                3,
            ),
            (
                torch.FloatTensor(
                    [[4.0, 2.0, 2.0], [0.0, -2.0, 0.0], [-4.0, -3.0, -2.0], [3.0, 2.0, -2.0], [0.0, 3.0, 4.0]]
                ),
                torch.FloatTensor([1.0, 1.0, 1.0, 0.0, 1.0]),
                torch.FloatTensor([[-3.0, -3.0, -2.0]]),
                1,
            ),
        ]

        outputs = [torch.as_tensor(0.3333), torch.as_tensor(0.3333), torch.as_tensor(1.0)]

        self.test_fn(callback, inputs, outputs)


class WeatherForecastGrader(Grader):
    """Weather Forecast Grader"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.inputs = [
            torch.FloatTensor(
                [
                    [74.8, 88.4, 54.4, 56.6, 65.3, 81.7, 74.5, 94.8, 72.7, 81.6],
                    [67.4, 70.0, 51.1, 58.4, 64.6, 75.9, 84.8, 90.0, 58.0, 64.1],
                ]
            ),
            torch.FloatTensor(
                [
                    [84.0, 95.7, 69.8, 93.7, 70.9, 77.6, 97.6, 51.8, 59.2, 68.6],
                    [65.2, 96.6, 58.7, 63.4, 57.5, 51.5, 60.4, 96.4, 86.1, 87.1],
                    [76.3, 62.1, 79.2, 51.6, 56.9, 62.1, 90.7, 89.6, 63.9, 74.0],
                    [90.9, 99.8, 84.9, 78.3, 91.7, 60.2, 79.6, 55.6, 57.6, 62.0],
                    [86.3, 85.0, 60.1, 82.5, 88.7, 71.8, 75.9, 80.7, 90.5, 99.0],
                    [55.7, 65.8, 84.8, 95.7, 96.7, 97.0, 79.9, 53.2, 77.2, 59.3],
                    [51.7, 97.2, 94.0, 50.0, 79.6, 70.7, 70.8, 63.5, 84.6, 60.1],
                    [84.1, 87.6, 92.8, 84.3, 50.2, 58.7, 87.4, 80.2, 55.4, 60.6],
                    [98.5, 91.8, 64.0, 68.7, 51.1, 74.5, 56.1, 55.7, 73.6, 78.7],
                    [64.7, 89.8, 59.7, 97.6, 92.1, 53.9, 68.7, 76.1, 78.6, 80.9],
                    [84.8, 76.4, 62.8, 86.8, 51.0, 60.1, 68.7, 62.8, 66.2, 54.5],
                    [69.6, 80.3, 58.7, 73.7, 92.8, 72.4, 75.6, 72.8, 80.0, 90.8],
                    [98.6, 90.8, 98.7, 73.1, 52.5, 63.1, 92.0, 74.8, 62.5, 55.8],
                    [51.6, 53.8, 69.9, 88.7, 88.5, 50.8, 90.5, 55.4, 69.7, 64.8],
                    [70.1, 70.0, 52.5, 53.4, 71.0, 75.3, 63.6, 84.4, 52.4, 73.3],
                    [96.9, 64.8, 97.5, 84.0, 52.4, 90.8, 72.1, 63.8, 94.9, 54.7],
                    [77.6, 69.7, 92.8, 81.9, 87.0, 83.8, 68.9, 69.7, 54.3, 88.5],
                    [94.8, 92.1, 57.3, 76.1, 57.3, 61.2, 60.4, 83.5, 60.1, 74.4],
                    [76.0, 91.1, 56.1, 57.8, 60.4, 92.4, 66.0, 96.0, 84.0, 78.1],
                    [74.8, 70.0, 78.1, 69.2, 74.8, 78.1, 55.4, 61.8, 95.1, 54.7],
                    [73.2, 99.7, 84.0, 75.7, 53.3, 87.3, 57.1, 67.9, 66.6, 71.2],
                    [75.2, 95.6, 78.1, 97.3, 90.2, 59.1, 86.2, 57.3, 64.4, 82.3],
                    [83.2, 93.7, 66.9, 75.0, 87.8, 50.8, 93.0, 54.3, 75.3, 70.7],
                    [61.8, 78.3, 95.6, 67.6, 60.1, 65.7, 50.2, 86.2, 62.9, 58.3],
                    [60.5, 89.3, 88.2, 94.1, 84.0, 66.6, 68.0, 82.3, 95.5, 81.7],
                    [63.1, 63.2, 51.3, 80.4, 60.9, 52.7, 96.9, 58.7, 72.1, 82.1],
                    [75.7, 58.1, 54.7, 94.9, 79.0, 95.7, 66.6, 82.3, 69.2, 73.8],
                    [59.7, 83.4, 82.9, 74.4, 69.3, 59.5, 92.2, 56.3, 85.2, 66.5],
                    [62.9, 79.4, 62.0, 80.7, 79.9, 56.4, 79.1, 85.6, 84.8, 71.8],
                    [54.5, 71.1, 83.6, 65.8, 84.4, 91.6, 61.9, 75.2, 85.3, 76.9],
                ]
            ),
        ]
        self.weather_forecast = self.module.weather_forecast.WeatherForecast

    def test_fn(
        self,
        fname: str | Callable,
        inputs,
        outputs: list[torch.Tensor],
    ):
        if isinstance(fname, str):
            answer_fn = getattr(self.weather_forecast, fname)
        else:
            answer_fn = fname

        for inp, out in zip(inputs, outputs, strict=True):
            if not isinstance(inp, (tuple, list)):
                inp = (inp,)

            a = answer_fn(*inp)
            b = out

            if isinstance(b, tuple):
                if not isinstance(a, tuple):
                    raise CheckFailed(f"Expected tuple, got {type(a)}")
            else:
                a = (a,)
                b = (b,)

            if len(a) != len(b):
                raise CheckFailed(f"Expected {len(b)} outputs, got {len(a)}")

            for _a, _b in zip(a, b):
                if not isinstance(_a, torch.Tensor):
                    raise CheckFailed(f"Expected torch.Tensor, got {type(_a)}")
                if _a.dtype != _b.dtype:
                    raise CheckFailed(f"Expected dtype {_b.dtype}, got {_a.dtype}")
                if _a.shape != _b.shape:
                    raise CheckFailed(f"Expected shape {_b.shape}, got {_a.shape}")
                if abs(_a - _b).sum() > 1e-3:
                    raise CheckFailed(f"Expected\nOutput:\n{_b}\nGot:\n{_a}\nInputs:\n{inp}\n")

    @Case(score=5)
    def test_1_find_min_and_max(self):
        """Find min and max per day"""

        def callback(_x):
            weather_forecast = self.weather_forecast(_x)
            ret = weather_forecast.find_min_and_max_per_day()

            return ret

        outputs = [
            (torch.FloatTensor([54.4, 51.1]), torch.FloatTensor([94.8, 90.0])),
            (
                torch.FloatTensor(
                    [
                        51.8,
                        51.5,
                        51.6,
                        55.6,
                        60.1,
                        53.2,
                        50.0,
                        50.2,
                        51.1,
                        53.9,
                        51.0,
                        58.7,
                        52.5,
                        50.8,
                        52.4,
                        52.4,
                        54.3,
                        57.3,
                        56.1,
                        54.7,
                        53.3,
                        57.3,
                        50.8,
                        50.2,
                        60.5,
                        51.3,
                        54.7,
                        56.3,
                        56.4,
                        54.5,
                    ]
                ),
                torch.FloatTensor(
                    [
                        97.6,
                        96.6,
                        90.7,
                        99.8,
                        99.0,
                        97.0,
                        97.2,
                        92.8,
                        98.5,
                        97.6,
                        86.8,
                        92.8,
                        98.7,
                        90.5,
                        84.4,
                        97.5,
                        92.8,
                        94.8,
                        96.0,
                        95.1,
                        99.7,
                        97.3,
                        93.7,
                        95.6,
                        95.5,
                        96.9,
                        95.7,
                        92.2,
                        85.6,
                        91.6,
                    ]
                ),
            ),
        ]

        self.test_fn(callback, self.inputs, outputs)

    @Case(score=5)
    def test_2_largest_drop(self):
        """Find largest average temperature drop"""

        def callback(_x):
            weather_forecast = self.weather_forecast(_x)
            ret = weather_forecast.find_the_largest_drop()

            return ret

        outputs = [torch.as_tensor(-6.05), torch.as_tensor(-12.88)]

        self.test_fn(callback, self.inputs, outputs)

    @Case(score=5)
    def test_3_most_extreme_day(self):
        """Find most extreme day"""

        def callback(_x):
            weather_forecast = self.weather_forecast(_x)
            ret = weather_forecast.find_the_most_extreme_day()

            return ret

        outputs = [
            torch.FloatTensor([94.8, 90.0]),
            torch.FloatTensor(
                [
                    51.8,
                    96.6,
                    90.7,
                    99.8,
                    60.1,
                    53.2,
                    97.2,
                    50.2,
                    98.5,
                    53.9,
                    86.8,
                    58.7,
                    52.5,
                    90.5,
                    84.4,
                    52.4,
                    54.3,
                    94.8,
                    96.0,
                    95.1,
                    99.7,
                    57.3,
                    50.8,
                    95.6,
                    60.5,
                    96.9,
                    95.7,
                    92.2,
                    56.4,
                    54.5,
                ]
            ),
        ]

        self.test_fn(callback, self.inputs, outputs)

    @Case(score=5)
    def test_4_max_last_k_days(self):
        """Max temperature"""

        def callback(_x, _k):
            weather_forecast = self.weather_forecast(_x)
            ret = weather_forecast.max_last_k_days(_k)

            return ret

        extra_inputs = [1, 10]
        inputs = [(case,) + (extra,) for case, extra in zip(self.inputs, extra_inputs)]
        outputs = [
            torch.FloatTensor([90.0]),
            torch.FloatTensor([99.7, 97.3, 93.7, 95.6, 95.5, 96.9, 95.7, 92.2, 85.6, 91.6]),
        ]

        self.test_fn(callback, inputs, outputs)

    @Case(score=5)
    def test_5_predict_temperature(self):
        """Predict the temperature"""

        def callback(_x, _k):
            weather_forecast = self.weather_forecast(_x)
            ret = weather_forecast.predict_temperature(_k)

            return ret

        extra_inputs = [1, 10]
        inputs = [(case,) + (extra,) for case, extra in zip(self.inputs, extra_inputs)]
        outputs = [torch.as_tensor(68.43), torch.as_tensor(74.23)]

        self.test_fn(callback, inputs, outputs)

    @Case(score=5)
    def test_6_what_day_is_this_from(self):
        """What day is this from?"""

        def callback(_x, _z):
            weather_forecast = self.weather_forecast(_x)
            ret = weather_forecast.what_day_is_this_from(_z)

            return ret

        extra_inputs = [
            torch.FloatTensor([74.8, 88.4, 54.4, 56.6, 65.3, 81.7, 74.5, 94.8, 72.7, 81.6]),
            torch.FloatTensor([64.7, 89.8, 59.7, 97.6, 92.1, 53.9, 68.7, 76.1, 78.6, 80.9]),
        ]
        inputs = [(case,) + (extra,) for case, extra in zip(self.inputs, extra_inputs)]
        outputs = [torch.as_tensor(0), torch.as_tensor(9)]

        self.test_fn(callback, inputs, outputs)
