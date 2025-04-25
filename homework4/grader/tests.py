"""
Do not modify unless you know what you are doing!
"""

import warnings

import numpy as np
import torch

from .datasets import road_dataset
from .grader import Case, Grader
from .metrics import PlannerMetric

# A hidden test split will be used for grading
DATA_SPLIT = "drive_data/val"


def get_device() -> torch.device:
    if torch.cuda.is_available():
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


def normalized_score(val: float, low: float, high: float, lower_is_better=False):
    """
    Normalizes and clips the value to the range [0, 1]
    """
    score = np.clip((val - low) / (high - low), 0, 1)
    return 1 - score if lower_is_better else score


class BaseGrader(Grader):
    """
    Helper for loading models and checking their correctness
    """

    METRIC = PlannerMetric
    MODEL_NAME: str = None
    TRANSFORM_PIPELINE: str = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.device = get_device()

        # some grading might still be non-deterministic
        np.random.seed(2024)
        torch.manual_seed(2024)

        self.data = road_dataset.load_data(
            DATA_SPLIT,
            num_workers=1,
            batch_size=64,
            shuffle=False,
            transform_pipeline=self.TRANSFORM_PIPELINE,
        )

        self._model = None
        self._metrics_computed = False
        self._metric_computer = self.METRIC()

    @property
    def model(self):
        """
        Lazily loads the model
        """
        if self._model is None:
            self._model = self.module.load_model(self.MODEL_NAME, with_weights=True)
            self._model.to(self.device)

        return self._model

    @property
    def metrics(self):
        """
        Runs the model on the data and computes metrics only once
        """
        if not self._metrics_computed:
            self.compute_metrics()
            self._metrics_computed = True

        return self._metric_computer.compute()

    @torch.inference_mode()
    def compute_metrics(self):
        """
        Implemented by subclasses depending on the model
        """
        raise NotImplementedError


class MLPPlannerGrader(BaseGrader):
    """MLP Planner"""

    TRANSFORM_PIPELINE = "state_only"
    MODEL_NAME = "mlp_planner"
    LON_ERROR = 0.16, 0.2, 0.3
    LAT_ERROR = 0.5, 0.6, 0.7

    @torch.inference_mode()
    def compute_metrics(self):
        self.model.eval()

        for batch in self.data:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            track_left = batch["track_left"]
            track_right = batch["track_right"]
            waypoints = batch["waypoints"]
            waypoints_mask = batch["waypoints_mask"]

            pred = self.model(track_left, track_right)

            self._metric_computer.add(pred, waypoints, waypoints_mask)

    @Case(score=5, timeout=10000)
    def test_model(self):
        """Test Output Shape"""
        model = self.module.load_model(self.MODEL_NAME, with_weights=False).to(self.device)

        batch_size = 16
        n_track = 10
        n_waypoints = 3

        dummy_track_left = torch.rand(batch_size, n_track, 2).to(self.device)
        dummy_track_right = torch.rand(batch_size, n_track, 2).to(self.device)
        output = model(dummy_track_left, dummy_track_right)
        output_expected_shape = (batch_size, n_waypoints, 2)

        assert output.shape == output_expected_shape, f"Expected shape {output_expected_shape}, got {output.shape}"

    @Case(score=10, timeout=10000)
    def test_longitudinal_error(self):
        """Longitudinal Error"""
        key = "longitudinal_error"
        val = self.metrics[key]
        score = normalized_score(val, self.LON_ERROR[1], self.LON_ERROR[2], lower_is_better=True)

        return score, f"{key}: {val:.3f}, required < {self.LON_ERROR[1]}"

    @Case(score=1, extra_credit=True)
    def test_longitudinal_error_extra(self):
        """Longitudinal Error: Extra Credit"""
        key = "longitudinal_error"
        val = self.metrics[key]
        score = normalized_score(val, self.LON_ERROR[0], self.LON_ERROR[1], lower_is_better=True)

        return score

    @Case(score=10)
    def test_lateral_error(self):
        """Lateral Error"""
        key = "lateral_error"
        val = self.metrics[key]
        score = normalized_score(val, self.LAT_ERROR[1], self.LAT_ERROR[2], lower_is_better=True)

        return score, f"{key}: {val:.3f}, required < {self.LAT_ERROR[1]}"

    @Case(score=1, extra_credit=True)
    def test_lateral_error_extra(self):
        """Lateral Error: Extra Credit"""
        key = "lateral_error"
        val = self.metrics[key]
        score = normalized_score(val, self.LAT_ERROR[0], self.LAT_ERROR[1], lower_is_better=True)

        return score

    @Case(score=10, timeout=20000)
    def test_driving_performance(self, track_name="lighthouse"):
        """Driving Performance"""
        try:
            import pystk  # noqa
            from .supertux_utils.evaluate import Evaluator
        except ImportError as e:
            print(e)
            return 0.0, "Skipping test (pystk not installed)."

        max_distance = 0.0
        total_track_distance = float("inf")

        max_distance_list = []
        evaluator = Evaluator(self.model, device=self.device)

        # Take best of 3 runs since pystk is non-deterministic
        for _ in range(3):
            max_distance, total_track_distance = evaluator.evaluate(
                track_name=track_name,
                max_steps=500,
                frame_skip=4,
                disable_tqdm=True,
            )

            max_distance_list.append(max_distance)

        # Must finish 50% of the track for full score
        track_coverage = max(max_distance_list) / total_track_distance
        score = normalized_score(track_coverage, 0.25, 0.5)

        return score, f"track coverage: {track_coverage:.3f}, required > 0.5"


class TransformerPlannerGrader(MLPPlannerGrader):
    """Transformer Planner"""

    MODEL_NAME = "transformer_planner"
    LON_ERROR = 0.16, 0.2, 0.3
    LAT_ERROR = 0.5, 0.6, 0.7


class CNNPlannerGrader(BaseGrader):
    """CNN Planner"""

    TRANSFORM_PIPELINE = "default"
    MODEL_NAME = "cnn_planner"
    LON_ERROR = 0.2, 0.3, 0.4
    LAT_ERROR = 0.3, 0.45, 0.6

    @torch.inference_mode()
    def compute_metrics(self):
        self.model.eval()

        for batch in self.data:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            image = batch["image"]
            waypoints = batch["waypoints"]
            waypoints_mask = batch["waypoints_mask"]

            pred = self.model(image)

            self._metric_computer.add(pred, waypoints, waypoints_mask)

    @Case(score=5, timeout=10000)
    def test_model(self):
        """Test Output Shape"""
        model = self.module.load_model(self.MODEL_NAME, with_weights=False).to(self.device)

        batch_size = 4
        h, w = 96, 128
        n_waypoints = 3

        dummy_image = torch.rand(batch_size, 3, h, w).to(self.device)
        output = model(dummy_image)
        output_expected_shape = (batch_size, n_waypoints, 2)

        assert output.shape == output_expected_shape, f"Expected shape {output_expected_shape}, got {output.shape}"

    @Case(score=10, timeout=10000)
    def test_longitudinal_error(self):
        """Longitudinal Error"""
        key = "longitudinal_error"
        val = self.metrics[key]
        score = normalized_score(val, self.LON_ERROR[1], self.LON_ERROR[2], lower_is_better=True)

        return score, f"{key}: {val:.3f}, required < {self.LON_ERROR[1]}"

    @Case(score=1, extra_credit=True)
    def test_longitudinal_error_extra(self):
        """Longitudinal Error: Extra Credit"""
        key = "longitudinal_error"
        val = self.metrics[key]
        score = normalized_score(val, self.LON_ERROR[0], self.LON_ERROR[1], lower_is_better=True)

        return score

    @Case(score=10)
    def test_lateral_error(self):
        """Lateral Error"""
        key = "lateral_error"
        val = self.metrics[key]
        score = normalized_score(val, self.LAT_ERROR[1], self.LAT_ERROR[2], lower_is_better=True)

        return score, f"{key}: {val:.3f}, required < {self.LAT_ERROR[1]}"

    @Case(score=1, extra_credit=True)
    def test_lateral_error_extra(self):
        """Lateral Error: Extra Credit"""
        key = "lateral_error"
        val = self.metrics[key]
        score = normalized_score(val, self.LAT_ERROR[0], self.LAT_ERROR[1], lower_is_better=True)

        return score

    @Case(score=5, timeout=20000)
    def test_driving_performance(self, track_name="lighthouse"):
        """Driving Performance"""
        try:
            import pystk  # noqa
            from .supertux_utils.evaluate import Evaluator
        except ImportError:
            return 0.0, "Skipping test (pystk not installed)."

        max_distance = 0.0
        total_track_distance = float("inf")

        max_distance_list = []
        evaluator = Evaluator(self.model, device=self.device)

        # Take best of 3 runs since pystk is non-deterministic
        for _ in range(3):
            max_distance, total_track_distance = evaluator.evaluate(
                track_name=track_name,
                max_steps=500,
                frame_skip=4,
                disable_tqdm=True,
            )

            max_distance_list.append(max_distance)

        # Must finish 50% of the track for full score
        track_coverage = max(max_distance_list) / total_track_distance
        score = normalized_score(track_coverage, 0.25, 0.5)

        return score, f"track coverage: {track_coverage:.3f}, required > 0.5"
