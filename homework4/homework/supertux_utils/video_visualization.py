import matplotlib.pyplot as plt
import numpy as np


def save_video(
    images: list,
    filename: str = "video.mp4",
    fps: int = 20,
):
    """
    Save image sequence as video.
    """
    try:
        import imageio
    except ImportError:
        raise ImportError("pip install imageio imageio-ffmpeg")

    with imageio.get_writer(filename, fps=fps, macro_block_size=1) as writer:
        for img in images:
            writer.append_data(img)

    print(f"{len(images)} frames saved to {filename} @ {fps}fps")


class VideoVisualizer:
    def __init__(self):
        fig, axes = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=(10, 5),
        )

        self.fig = fig
        self.axes = axes
        self.frames = []

        # Don't automatically show the plot in notebooks
        plt.close(fig)

    def process(self, sample: dict, debug_info: dict | None = None):
        """
        A simple visualization of the RGB image, track, and predicted waypoints.
        """
        fig, axes = self.fig, self.axes

        for ax in axes:
            ax.clear()

        axes[0].imshow(sample["image_raw"])
        axes[0].set_title("RGB")
        axes[0].axis("off")

        track_left = sample["track_left"]
        track_right = sample["track_right"]
        axes[1].plot(track_left[:, 0], track_left[:, 1], "ro-")
        axes[1].plot(track_right[:, 0], track_right[:, 1], "bo-")
        axes[1].set_xlim(-10, 10)
        axes[1].set_ylim(-5, 15)

        if debug_info is not None:
            waypoints = debug_info["waypoints"]

            axes[1].plot(waypoints[:, 0], waypoints[:, 1], "g-o")
            axes[1].set_title(f"Steer: {debug_info['steer']:.2f} Speed: {debug_info['speed']:.2f}")

        # turn the matplotlib figure into a numpy image
        s, (width, height) = fig.canvas.print_to_buffer()
        viz = np.frombuffer(s, np.uint8).reshape((height, width, 4))[:, :, :3]

        self.frames.append(viz)
