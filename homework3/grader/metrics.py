import torch


class AccuracyMetric:
    def __init__(self):
        self.correct = 0
        self.total = 0

    def reset(self):
        """
        should be called before each epoch
        """
        self.correct = 0
        self.total = 0

    @torch.no_grad()
    def add(self, preds: torch.Tensor, labels: torch.Tensor):
        """
        Updates using predictions and ground truth labels

        Args:
            preds (torch.LongTensor): (b,) or (b, h, w) tensor with class predictions
            labels (torch.LongTensor): (b,) or (b, h, w) tensor with ground truth class labels
        """
        self.correct += (preds.type_as(labels) == labels).sum().item()
        self.total += labels.numel()

    def compute(self) -> dict[str, float]:
        return {
            "accuracy": self.correct / (self.total + 1e-5),
            "num_samples": self.total,
        }


class DetectionMetric:
    """
    Computes iou and depth metrics
    """

    def __init__(self, num_classes: int = 3):
        self.confusion_matrix = ConfusionMatrix(num_classes)
        self.avg_depth_errors = []

        self.tp_depth_error_sum = 0
        self.tp_depth_error_n = 0

    def reset(self):
        self.confusion_matrix.reset()
        self.avg_depth_errors.clear()
        self.tp_depth_error_sum = 0
        self.tp_depth_error_n = 0

    @torch.no_grad()
    def add(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor,
        depth_preds: torch.Tensor,
        depth_labels: torch.Tensor,
    ):
        """
        Args:
            pred (torch.LongTensor): (b, h, w) with class predictions
            labels (torch.LongTensor): (b, h, w) with ground truth class labels
            depth_preds (torch.FloatTensor): (b, h, w) with depth predictions
            depth_labels (torch.FloatTensor): (b, h, w) with ground truth depth
        """
        depth_error = (depth_preds - depth_labels).abs()

        # only consider matches on road
        tp_mask = ((preds == labels) & (labels > 0)).float()
        tp_depth_error = depth_error * tp_mask

        self.confusion_matrix.add(preds, labels)
        self.avg_depth_errors.append(depth_error.mean().item())

        self.tp_depth_error_sum += tp_depth_error.sum().item()
        self.tp_depth_error_n += tp_mask.sum().item()

    def compute(self) -> dict[str, float]:
        """
        Returns:
            dict of metrics
        """
        metrics = self.confusion_matrix.compute()
        metrics["abs_depth_error"] = sum(self.avg_depth_errors) / (len(self.avg_depth_errors) + 1e-5)
        metrics["tp_depth_error"] = self.tp_depth_error_sum / (self.tp_depth_error_n + 1e-5)

        return metrics


class ConfusionMatrix:
    """
    Metric for computing mean IoU and accuracy

    Sample usage:
    >>> batch_size, num_classes = 8, 6
    >>> preds = torch.randint(0, num_classes, (batch_size,))   # (b,)
    >>> labels = torch.randint(0, num_classes, (batch_size,))   # (b,)
    >>> cm = ConfusionMatrix(num_classes=num_classes)
    >>> cm.add(preds, labels)
    >>> # {'iou': 0.125, 'accuracy': 0.25}
    >>> metrics = cm.compute()
    >>> # clear the confusion matrix before the next epoch
    >>> cm.reset()
    """

    def __init__(self, num_classes: int = 3):
        """
        Builds and updates a confusion matrix.

        Args:
            num_classes: number of label classes
        """
        self.matrix = torch.zeros(num_classes, num_classes)
        self.class_range = torch.arange(num_classes)

    @torch.no_grad()
    def add(self, preds: torch.Tensor, labels: torch.Tensor):
        """
        Updates using predictions and ground truth labels

        Args:
            preds (torch.LongTensor): (b,) or (b, h, w) tensor with class predictions
            labels (torch.LongTensor): (b,) or (b, h, w) tensor with ground truth class labels
        """
        if preds.dim() > 1:
            preds = preds.view(-1)
            labels = labels.view(-1)

        preds_one_hot = (preds.type_as(labels).cpu()[:, None] == self.class_range[None]).int()
        labels_one_hot = (labels.cpu()[:, None] == self.class_range[None]).int()
        update = labels_one_hot.T @ preds_one_hot

        self.matrix += update

    def reset(self):
        """
        Resets the confusion matrix, should be called before each epoch
        """
        self.matrix.zero_()

    def compute(self) -> dict[str, float]:
        """
        Computes the mean IoU and accuracy
        """
        true_pos = self.matrix.diagonal()
        class_iou = true_pos / (self.matrix.sum(0) + self.matrix.sum(1) - true_pos + 1e-5)
        mean_iou = class_iou.mean().item()
        accuracy = (true_pos.sum() / (self.matrix.sum() + 1e-5)).item()

        return {
            "iou": mean_iou,
            "accuracy": accuracy,
        }
