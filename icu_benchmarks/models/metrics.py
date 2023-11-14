import logging
from collections import Counter
from typing import Callable, Tuple, cast

import accelerate.utils as accutils
import ignite.distributed as idist
import numpy as np
import torch
from ignite.exceptions import NotComputableError
from ignite.metrics import EpochMetric
from ignite.metrics.metric import Metric, reinit__is_reduced


class AccelerateEpochMetric(EpochMetric):
    """
    Adapted Ignite EpochMetric to work in a distributed
    setting using Huggingface Accelerate

    Base Implementation:
        https://github.com/pytorch/ignite/blob/5d8d6bf4af59943da7e6373315000b515263c8c1/ignite/metrics/epoch_metric.py
    """

    def __init__(
        self,
        *args,
        allow_distributed: bool = True,
        gather_on_gpu: bool = True,
        **kwargs,
    ) -> None:

        self.pad_index = -2
        self.allow_distributed = allow_distributed
        self.gather_on_gpu = gather_on_gpu and self.allow_distributed
        self.ws = idist.get_world_size()
        logging.info(
            f"[{self.__class__.__name__}] distributed {self.ws}: {self.allow_distributed}, gather on gpu: {self.gather_on_gpu}"
        )
        if not self.allow_distributed:
            logging.info(f"[{self.__class__.__name__}] without distributed `compute`")

        assert (
            "accelerator" in kwargs
        ), f"{self.__class__.__name__} requires accelerator to be passed"
        self.accelerator = kwargs["accelerator"]
        del kwargs["accelerator"]

        super(AccelerateEpochMetric, self).__init__(*args, **kwargs)

    # @reinit__is_reduced
    # def update(self, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
    #     self._check_shape(output)

    #     y_pred, y = output[0].detach().contiguous(), output[1].detach().contiguous()
    #     if self.ws > 1 and self.allow_distributed:
    #         y_pred, y = self.accelerator.gather_for_metrics((y_pred, y))

    #     if y_pred.ndimension() == 2 and y_pred.shape[1] == 1:
    #         y_pred = y_pred.squeeze(dim=-1)

    #     if y.ndimension() == 2 and y.shape[1] == 1:
    #         y = y.squeeze(dim=-1)

    #     y_pred = y_pred.clone().to(self._device)
    #     y = y.clone().to(self._device)

    #     self._check_type((y_pred, y))
    #     self._predictions.append(y_pred)
    #     self._targets.append(y)

    #     # Check once the signature and execution of compute_fn
    #     if len(self._predictions) == 1 and self._check_compute_fn:
    #         try:
    #             self.compute_fn(self._predictions[0], self._targets[0])
    #         except Exception as e:
    #             logging.warning(f"Probably, there can be a problem with `compute_fn`:\n {e}.")

    def compute(self) -> float:
        if len(self._predictions) < 1 or len(self._targets) < 1:
            raise NotComputableError(
                "EpochMetric must have at least one example before it can be computed."
            )

        if self._result is None:

            _prediction_tensor = torch.cat(self._predictions, dim=0)
            _target_tensor = torch.cat(self._targets, dim=0)

            # ws = idist.get_world_size()
            if self.ws > 1 and self.allow_distributed:

                # All gather across all processes
                if self.gather_on_gpu:

                    # logging.info(f"[{self.accelerator.process_index}] end of dataloader: {self.accelerator.gradient_state.end_of_dataloader}")
                    # logging.info(f"[{self.accelerator.process_index}] gradient state remainder: {self.accelerator.gradient_state.remainder}")
                    # logging.info(f"[{self.accelerator.process_index}] label length: {len(_target_tensor)}")
                    # logging.info(f"[{self.accelerator.process_index}] labels: {Counter([tensor.item() for tensor in _target_tensor])}")
                    # logging.info(f"[{self.accelerator.process_index}] index: {torch.arange(len(_target_tensor))[_target_tensor == -1]}")

                    _target_tensor = self.accelerator.pad_across_processes(
                        _target_tensor.to(self.accelerator.device),
                        pad_index=self.pad_index,
                    )
                    _prediction_tensor = self.accelerator.pad_across_processes(
                        _prediction_tensor.to(self.accelerator.device),
                        pad_index=self.pad_index,
                    )

                    recv_msg = self.accelerator.gather(
                        (_prediction_tensor, _target_tensor)
                    )
                    _prediction_tensor, _target_tensor = tuple(
                        tens.cpu() for tens in recv_msg
                    )

                    _prediction_tensor = _prediction_tensor[
                        _target_tensor != self.pad_index
                    ]
                    _target_tensor = _target_tensor[_target_tensor != self.pad_index]

                else:

                    _prediction_tensor = cast(
                        torch.Tensor, idist.all_gather(_prediction_tensor)
                    )
                    _target_tensor = cast(
                        torch.Tensor, idist.all_gather(_target_tensor)
                    )

            self._result = 0.0
            # if idist.get_rank() == 0:
            if self.accelerator.is_main_process:
                # Run compute_fn on zero rank only
                # logging.info(f"[{self.accelerator.process_index}] has gathered {len(_prediction_tensor)} predictions, {_prediction_tensor.dtype}")
                # logging.info(f"[{self.accelerator.process_index}] has gathered {_prediction_tensor} predictions")
                # logging.info(f"[{self.accelerator.process_index}] labels: {Counter([tensor.item() for tensor in _target_tensor])}")
                self._result = self.compute_fn(_prediction_tensor, _target_tensor)

            if self.ws > 1 and self.allow_distributed:
                # broadcast result to all processes
                if self.gather_on_gpu:
                    result_msg = torch.tensor(self._result, dtype=torch.float64).to(
                        self.accelerator.device
                    )
                    self._result = float(accutils.broadcast(result_msg, from_process=0))
                    # logging.info(f"[{self.accelerator.process_index}] result {self._result:.2f}")
                else:
                    self._result = cast(float, idist.broadcast(self._result, src=0))

        return self._result


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def balanced_accuracy_compute_fn(
    y_preds: torch.Tensor, y_targets: torch.Tensor
) -> float:
    try:
        from sklearn.metrics import balanced_accuracy_score
    except ImportError:
        raise RuntimeError("This contrib module requires sklearn to be installed.")

    y_true = y_targets.numpy()
    y_pred = np.argmax(y_preds.numpy(), axis=-1)
    return balanced_accuracy_score(y_true, y_pred)


def average_precision_compute_fn(
    y_preds: torch.Tensor, y_targets: torch.Tensor
) -> float:
    try:
        from sklearn.metrics import average_precision_score
    except ImportError:
        raise RuntimeError("This contrib module requires sklearn to be installed.")

    y_true = y_targets.numpy().astype(np.int32)
    y_pred = y_preds.numpy()

    try:
        return average_precision_score(y_true, y_pred)
    except ValueError as e:
        logging.error(e)
        logging.error(f"y_true: {np.unique(y_true)}, {y_true.shape}")
        logging.error(f"Counter: {Counter(y_true)}")


def roc_auc_compute_fn(y_preds: torch.Tensor, y_targets: torch.Tensor) -> float:
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        raise RuntimeError("This contrib module requires sklearn to be installed.")

    y_true = y_targets.numpy().astype(np.int32)
    y_pred = y_preds.numpy()

    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError as e:
        logging.error(e)
        logging.error(f"y_true: {np.unique(y_true)}, {y_true.shape}")
        logging.error(f"Counter: {Counter(y_true)}")


def ece_curve_compute_fn(y_preds: torch.Tensor, y_targets: torch.Tensor) -> float:
    try:
        from sklearn.calibration import calibration_curve
    except ImportError:
        raise RuntimeError("This contrib module requires sklearn to be installed.")

    y_true = y_targets.numpy()
    y_pred = y_preds.numpy()
    return calibration_curve(y_true, y_pred, n_bins=10)


def mae_with_invert_compute_fn(
    y_preds: torch.Tensor, y_targets: torch.Tensor, invert_fn: Callable = lambda x: x
) -> float:
    try:
        from sklearn.metrics import mean_absolute_error
    except ImportError:
        raise RuntimeError("This contrib module requires sklearn to be installed.")

    y_true = invert_fn(y_targets.numpy().reshape(-1, 1))[:, 0]
    y_pred = invert_fn(y_preds.numpy().reshape(-1, 1))[:, 0]
    return float(mean_absolute_error(y_true, y_pred))


class BalancedAccuracy(EpochMetric):
    def __init__(
        self, output_transform: Callable = lambda x: x, check_compute_fn: bool = False
    ) -> None:
        super(BalancedAccuracy, self).__init__(
            balanced_accuracy_compute_fn,
            output_transform=output_transform,
            check_compute_fn=check_compute_fn,
        )


class CustomAveragePrecision(AccelerateEpochMetric):
    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        check_compute_fn: bool = False,
        **kwargs,
    ) -> None:
        super(CustomAveragePrecision, self).__init__(
            average_precision_compute_fn,
            output_transform=output_transform,
            check_compute_fn=check_compute_fn,
            **kwargs,
        )


class CustomROCAUC(AccelerateEpochMetric):
    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        check_compute_fn: bool = False,
        **kwargs,
    ) -> None:
        super(CustomROCAUC, self).__init__(
            roc_auc_compute_fn,
            output_transform=output_transform,
            check_compute_fn=check_compute_fn,
            **kwargs,
        )


class CalibrationCurve(EpochMetric):
    def __init__(
        self, output_transform: Callable = lambda x: x, check_compute_fn: bool = False
    ) -> None:
        super(CalibrationCurve, self).__init__(
            ece_curve_compute_fn,
            output_transform=output_transform,
            check_compute_fn=check_compute_fn,
        )


class MAE(EpochMetric):
    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        check_compute_fn: bool = False,
        invert_transform: Callable = lambda x: x,
    ) -> None:
        super(MAE, self).__init__(
            lambda x, y: mae_with_invert_compute_fn(x, y, invert_transform),
            output_transform=output_transform,
            check_compute_fn=check_compute_fn,
        )


class CustomMAE(AccelerateEpochMetric):
    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        check_compute_fn: bool = False,
        **kwargs,
    ) -> None:
        super(CustomMAE, self).__init__(
            mae_with_invert_compute_fn,
            output_transform=output_transform,
            check_compute_fn=check_compute_fn,
            **kwargs,
        )


class CustomBalancedAccuracy(AccelerateEpochMetric):
    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        check_compute_fn: bool = False,
        **kwargs,
    ) -> None:
        super(CustomBalancedAccuracy, self).__init__(
            balanced_accuracy_compute_fn,
            output_transform=output_transform,
            check_compute_fn=check_compute_fn,
            **kwargs,
        )


# ==============================
# MIMIC-III LOS Kappa
# ==============================
# Get custom bins for LOS Kappa computation
# according to MIMIC-3 benchmarking repository: https://github.com/YerevaNN/mimic3-benchmarks
# see also e.g.: https://www.nature.com/articles/s41597-019-0103-9#Sec3
# https://github.com/YerevaNN/mimic3-benchmarks/blob/220565b5ea3552ae487b41b6dd862f3a619f7619/mimic3models/metrics.py#L149
class CustomBins:
    inf = 1e18
    bins = [
        (-inf, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 14),
        (14, +inf),
    ]
    nbins = len(bins)
    means = [
        11.450379,
        35.070846,
        59.206531,
        83.382723,
        107.487817,
        131.579534,
        155.643957,
        179.660558,
        254.306624,
        585.325890,
    ]


def get_bin_custom(x, nbins, one_hot=False):
    for i in range(nbins):
        a = CustomBins.bins[i][0] * 24.0
        b = CustomBins.bins[i][1] * 24.0
        if a <= x < b:
            if one_hot:
                ret = np.zeros((CustomBins.nbins,))
                ret[i] = 1
                return ret
            return i
    return None


def mimic3_kappa_with_invert_compute_fn(
    y_preds: torch.Tensor, y_targets: torch.Tensor, invert_fn=Callable
) -> float:
    try:
        from sklearn.metrics import cohen_kappa_score
    except ImportError:
        raise RuntimeError("This contrib module requires sklearn to be installed.")

    y_true = invert_fn(y_targets.numpy().reshape(-1, 1))[:, 0]
    y_pred = invert_fn(y_preds.numpy().reshape(-1, 1))[:, 0]

    # print("=================")
    # print("True:", y_true)
    # print("Pred:", y_pred)
    # print("=================")

    # Get bins according to https://github.com/YerevaNN/mimic3-benchmarks
    y_true_bins = [get_bin_custom(x, CustomBins.nbins) for x in y_true]
    prediction_bins = [get_bin_custom(x, CustomBins.nbins) for x in y_pred]

    return cohen_kappa_score(y_true_bins, prediction_bins, weights="linear")


class CustomKappa(AccelerateEpochMetric):
    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        check_compute_fn: bool = False,
        invert_transform: Callable = lambda x: x,
        **kwargs,
    ) -> None:
        super(CustomKappa, self).__init__(
            lambda x, y: mimic3_kappa_with_invert_compute_fn(x, y, invert_transform),
            output_transform=output_transform,
            check_compute_fn=check_compute_fn,
            **kwargs,
        )
