from pathlib import Path

import numpy as np
import pystk
import torch
from tqdm import tqdm

from ..datasets.road_transforms import EgoTrackProcessor
from ..datasets.road_utils import Track
from .video_visualization import VideoVisualizer, save_video

MAPS = [
    "cornfield_crossing",
    "hacienda",
    "lighthouse",
    "snowmountain",
    "zengarden",
]


class BasePlanner:
    """
    Base class for learning-based planners.
    """

    ALLOWED_INFORMATION = []

    def __init__(
        self,
        model: torch.nn.Module,
        device: str,
    ):
        self.model = model
        self.model.to(device).eval()

        self.debug_info = {}

    @torch.inference_mode()
    def act(self, batch: dict) -> dict:
        allowed_info = {k: batch.get(k) for k in self.ALLOWED_INFORMATION}
        pred = self.model(**allowed_info)

        speed = np.linalg.norm(batch["velocity"].squeeze(0).cpu().numpy())
        steer, acceleration, brake = self.get_action(pred, speed)

        return {
            "steer": steer,
            "acceleration": acceleration,
            "brake": brake,
        }

    def get_action(
        self,
        waypoints: torch.Tensor,
        speed: torch.Tensor,
        target_speed: float = 5.0,
        idx: int = 2,
        p_gain: float = 10.0,
        constant_acceleration: float = 0.2,
    ) -> tuple[float, float, bool]:
        """
        Turns model predictions into steering, acceleration, and brake actions.

        Args:
            waypoints (torch.Tensor): predictions for a single sample (n, 2) or (1, n, 2)

        Returns:
            steer (float) from -1 to 1
            acceleration (float) from 0 to 1
            brake (bool) whether to brake
        """
        # make sure waypoints are (n, 2)
        waypoints = waypoints.squeeze(0).cpu().numpy()

        # steering angle is proportional to the angle of the target waypoint
        angle = np.arctan2(waypoints[idx, 0], waypoints[idx, 1])
        steer = p_gain * angle

        # very simple speed control, never brake!
        acceleration = constant_acceleration if target_speed > speed else 0.0
        brake = False

        # NOTE: you can modify use this and the visualizer to debug your model
        self.debug_info.update(
            {
                "waypoints": waypoints,
                "steer": steer,
                "speed": speed,
            }
        )

        # clip to valid range
        steer = float(np.clip(steer, -1, 1))
        acceleration = float(np.clip(acceleration, 0, 1))

        return steer, acceleration, brake


class TrackPlanner(BasePlanner):
    """
    Planner that uses track information to predict future waypoints.
    """

    ALLOWED_INFORMATION = ["track_left", "track_right"]


class ImagePlanner(BasePlanner):
    """
    Planner that drives from raw image data.
    """

    ALLOWED_INFORMATION = ["image"]


class RaceManager:
    """Singleton wrapper around pystk.Race"""
    race = None
    initialized = False

    @classmethod
    def get_instance(
        cls,
        track_name: str = "lighthouse",
        step_size: float = 0.1,
    ) -> "pystk.Race":
        if not cls.initialized:
            try:
                cfg = pystk.GraphicsConfig.ld()
                cfg.screen_width = 128
                cfg.screen_height = 96
                pystk.init(cfg)
                cls.initialized = True
            except ValueError as e:
                raise ValueError("Restart runtime if using a notebook") from e

        if cls.race is not None:
            cls.race.stop()
            del cls.race

        if track_name not in MAPS:
            raise ValueError(f'Track "{track_name}" not in {MAPS}')

        race_cfg = pystk.RaceConfig(track=track_name, step_size=step_size, seed=0)
        race_cfg.num_kart = 1
        cls.race = pystk.Race(race_cfg)

        return cls.race


class Evaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        visualizer: VideoVisualizer | None = None,
        device: str | None = None,
    ):
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        model_type = model.__class__.__name__
        model_to_planner = {
            "MLPPlanner": TrackPlanner,
            "TransformerPlanner": TrackPlanner,
            "CNNPlanner": ImagePlanner,
        }

        if model_type not in model_to_planner:
            raise ValueError(f"Model {model_type} not supported")

        self.planner = model_to_planner[model_type](model, self.device)
        self.visualizer = visualizer

        # lazily intialize the track later
        self.track = None
        self.track_transform = None

    @torch.inference_mode()
    def step(self, sample: dict):
        track_info = self.track_transform.from_frame(**sample)

        sample.update(track_info)
        sample["image"] = np.float32(sample["image_raw"]).transpose(2, 0, 1) / 255.0

        # turn all numpy into batch size=1 tensors
        batch = torch.utils.data.default_collate([sample])

        # hack: torch upcasts distance_down_track on some OS
        batch["distance_down_track"] = batch["distance_down_track"].float()
        batch = {k: v.to(self.device) for k, v in batch.items()}

        action = self.planner.act(batch)

        # optionally save/visualize frame info
        if self.visualizer is not None:
            self.visualizer.process(sample, self.planner.debug_info)

        return action

    def evaluate(
        self,
        track_name: str = "lighthouse",
        max_steps: int = 100,
        frame_skip: int = 4,
        step_size: float = 0.1,
        warmup: int = 10,
        disable_tqdm: bool = True,
    ):
        race = RaceManager.get_instance(track_name, step_size)
        race.start()

        state = pystk.WorldState()
        action = pystk.Action()
        track = pystk.Track()
        track.update()

        for _ in range(warmup):
            race.step(action)
            state.update()

        self.track = Track(
            path_distance=track.path_distance,
            path_nodes=track.path_nodes,
            path_width=track.path_width,
        )
        self.track_transform = EgoTrackProcessor(self.track)

        # keep track of how far the kart has gone
        max_distance = 0.0
        track_length = float(track.path_distance[-1][0])

        for _ in tqdm(range(max_steps), disable=disable_tqdm):
            max_distance = max(max_distance, state.karts[0].distance_down_track)
            sample = {
                "location": np.float32(state.karts[0].location),
                "front": np.float32(state.karts[0].front),
                "velocity": np.float32(state.karts[0].velocity),
                "distance_down_track": float(state.karts[0].distance_down_track),
                "image_raw": np.uint8(race.render_data[0].image),
            }

            action_dict = self.step(sample)
            action.steer = action_dict["steer"]
            action.acceleration = action_dict["acceleration"]
            action.brake = action_dict["brake"]

            for _ in range(frame_skip):
                race.step(action)
                state.update()

        return max_distance, track_length


def main():
    """
    Example Usage:
        python3 -m homework.supertux_utils.evaluate --model mlp_planner --track lighthouse
        python3 -m homework.supertux_utils.evaluate --model transformer_planner --track snowmountain
        python3 -m homework.supertux_utils.evaluate --model cnn_planner --track cornfield_crossing
    """
    import argparse

    from ..models import load_model

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--track", type=str, default="lighthouse", choices=MAPS)
    parser.add_argument("--max-steps", type=int, default=100, help="Number of steps to run")
    parser.add_argument("--disable_tqdm", action="store_true", help="Disable tqdm progress bar")

    args = parser.parse_args()

    model = load_model(args.model, with_weights=True)

    # Each frame will be passed to the visualizer and saved
    visualizer = VideoVisualizer()
    evaluator = Evaluator(model, visualizer=visualizer)
    evaluator.evaluate(
        track_name=args.track,
        max_steps=args.max_steps,
        disable_tqdm=args.disable_tqdm,
    )

    output_path = Path("videos") / f"{args.model}_{args.track}.mp4"
    output_path.parent.mkdir(exist_ok=True)

    # Output the saved frames as a video
    save_video(visualizer.frames, str(output_path))


if __name__ == "__main__":
    main()
