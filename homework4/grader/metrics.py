import numpy as np
import torch


class PlannerMetric:
    """
    Computes longitudinal and lateral errors for a planner
    """

    def __init__(self):
        self.l1_errors = []
        self.total = 0

    def reset(self):
        self.l1_errors = []
        self.total = 0

    @torch.no_grad()
    def add(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor,
        labels_mask: torch.Tensor,
    ):
        """
        Args:
            preds (torch.Tensor): (b, n, 2) float tensor with predicted waypoints
            labels (torch.Tensor): (b, n, 2) ground truth waypoints
            labels_mask (torch.Tensor): (b, n) bool mask for valid waypoints
        """
        error = (preds - labels).abs()
        error_masked = error * labels_mask[..., None]

        # sum across batch and waypoints
        error_sum = error_masked.sum(dim=(0, 1)).cpu().numpy()

        self.l1_errors.append(error_sum)
        self.total += labels_mask.sum().item()

    def compute(self) -> dict[str, float]:
        error = np.stack(self.l1_errors, axis=0)
        longitudinal_error = error[:, 0].sum() / self.total
        lateral_error = error[:, 1].sum() / self.total
        l1_error = longitudinal_error + lateral_error

        return {
            "l1_error": float(l1_error),
            "longitudinal_error": float(longitudinal_error),
            "lateral_error": float(lateral_error),
            "num_samples": self.total,
        }
