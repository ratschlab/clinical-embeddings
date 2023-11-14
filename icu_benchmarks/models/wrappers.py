import glob
import inspect
import logging
import os
import pickle

import accelerate
import gin
import ignite.distributed as idist
import joblib
import lightgbm
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from ignite.contrib.metrics import (
    ROC_AUC,
    AveragePrecision,
    PrecisionRecallCurve,
    RocCurve,
)
from ignite.metrics import Accuracy, MeanAbsoluteError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    mean_absolute_error,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from icu_benchmarks.models.metrics import (
    MAE,
    BalancedAccuracy,
    CalibrationCurve,
    CustomAveragePrecision,
    CustomROCAUC,
    CustomMAE,
    CustomKappa,
    CustomBalancedAccuracy,
)
from icu_benchmarks.models.utils import load_model_state, save_model

gin.config.external_configurable(
    torch.nn.functional.nll_loss, module="torch.nn.functional"
)
gin.config.external_configurable(
    torch.nn.functional.cross_entropy, module="torch.nn.functional"
)
gin.config.external_configurable(
    torch.nn.functional.mse_loss, module="torch.nn.functional"
)
gin.config.external_configurable(
    torch.nn.functional.binary_cross_entropy_with_logits, module="torch.nn.functional"
)

gin.config.external_configurable(lightgbm.LGBMClassifier, module="lightgbm")
gin.config.external_configurable(lightgbm.LGBMRegressor, module="lightgbm")
gin.config.external_configurable(LogisticRegression)

gin.config.external_configurable(torch.nn.Identity, module="torch.nn")


class MultiOutput_Binary_Metric:
    def __init__(self, metric, num_outputs=4, output_dim=0):
        self.num_outputs = num_outputs
        self.output_dim = output_dim
        self.metric = metric()

    def reset(self):
        self.metric.reset()

    def update(self, output):
        y_pred, y = map(lambda x: x.reshape(-1, self.num_outputs), output)
        output = (y_pred[:, self.output_dim], y[:, self.output_dim])
        self.metric.update(output)

    def compute(self):
        return self.metric.compute()


def l1_reg(embedding_module):
    n_params = sum(
        len(
            p.reshape(
                -1,
            )
        )
        for p in embedding_module.parameters()
    )
    return sum(torch.abs(p).sum() for p in embedding_module.parameters()) / n_params


@gin.configurable("DLWrapper")
class DLWrapper(object):
    def __init__(
        self,
        encoder=gin.REQUIRED,
        optimizer_fn=gin.REQUIRED,
        reg="l1",
        reg_weight=1e-3,
        cluster_reg=0.0,
        aux_label_idx=None,
        aux_label_type="max",
        aux_label_horizon=144,
        aux_label_weight=1.0,
        lr_decay=1.0,
        cluster_splitting: bool = False,
        clustering_tensorboard: str = None,
        clustering_plot_steps: int = 2000,
        clustering_regularizer_type: str = "functional_hard",
    ):

        # use acclerate
        self.accelerator = Accelerator(split_batches=True)
        self.device = self.accelerator.device
        self.accelerator.free_memory()
        self.accelerator.print(f"{AcceleratorState()}")

        if torch.cuda.is_available():
            # device = torch.device('cuda')
            self.pin_memory = True
            self.n_worker = 4
            logging.info(
                f"Model will be trained using GPU Hardware [Accelerate], workers: {self.n_worker}"
            )

        else:
            logging.info(
                "Model will be trained using CPU Hardware. This should be considerably slower"
            )
            self.pin_memory = False
            self.n_worker = 16
            device = torch.device("cpu")

        # self.device = device
        self.encoder = encoder
        # self.encoder.to(device)
        self.optimizer = optimizer_fn(self.encoder.parameters())
        if lr_decay < 1.0:
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=lr_decay
            )
        else:
            self.scheduler = None
        self.scaler = None

        # Regularization set-up
        if reg is None:
            self.reg_fn = None
        elif reg == "l1":
            self.reg_fn = l1_reg
        self.reg_weight = reg_weight

        # Auxiliary task set-up
        self.aux_label_idx = aux_label_idx
        self.aux_label_type = aux_label_type
        self.aux_label_horizon = aux_label_horizon
        self.aux_label_weight = aux_label_weight

        # Clustering Set-Up
        self.cluster_splitting = cluster_splitting
        self.clustering_tensorboard = clustering_tensorboard
        self.clustering_plot_steps = clustering_plot_steps
        self.cluster_reg = cluster_reg
        self.clustering_regularizer_type = clustering_regularizer_type
        assert self.clustering_regularizer_type in [
            "functional_hard",
            "functional_soft",
        ]
        if self.cluster_reg > 0:
            logging.warning(
                f"Clustering Regularization is enabled with weight {self.cluster_reg}"
            )
            logging.info(
                f"[{self.__class__.__name__}] Clustering Regularizer Type: {self.clustering_regularizer_type}"
            )

    def set_logdir(self, logdir):
        logging.info(f"[{self.__class__.__name__}] set log dir: {logdir}")
        self.logdir = logdir

    def set_scaler(self, scaler):
        self.scaler = scaler

    def set_metrics(self, allow_distributed: bool = True, with_kappa: bool = False):
        def softmax_binary_output_transform(output):
            with torch.no_grad():
                y_pred, y = output
                y_pred = torch.softmax(y_pred, dim=1)
                return y_pred[:, -1], y

        def softmax_multi_output_transform(output):
            with torch.no_grad():
                y_pred, y = output
                y_pred = torch.softmax(y_pred, dim=1)
                return y_pred, y

        def softmax_multi_output_binary_transform(output):
            with torch.no_grad():
                y_pred, y = output
                y_pred = torch.sigmoid(y_pred)
                return y_pred, y

        # if isinstance(self.encoder, torch.nn.Module):
        #     encoder = self.encoder
        # else:
        encoder = self.accelerator.unwrap_model(self.encoder)

        # output transform is not applied for contrib metrics so we do our own.
        if encoder.logit.out_features == 2:
            self.task_type = "binary_classification"
            self.output_transform = softmax_binary_output_transform

            self.metrics = {
                "PR": CustomAveragePrecision(
                    accelerator=self.accelerator, allow_distributed=allow_distributed
                ),
                "AUC": CustomROCAUC(
                    accelerator=self.accelerator, allow_distributed=allow_distributed
                ),
            }

            # self.metrics = {'PR': CustomAveragePrecision(), 'AUC': ROC_AUC(),
            #                 'PR_Curve': PrecisionRecallCurve(), 'ROC_Curve': RocCurve()}
            # 'Calibration_Curve': CalibrationCurve()}

        elif encoder.logit.out_features == 1:
            self.task_type = "regression"
            self.output_transform = lambda x: x
            if self.scaler is not None:
                self.metrics = {
                    "MAE": MAE(invert_transform=self.scaler.inverse_transform)
                }
            else:
                # self.metrics = {'MAE': MeanAbsoluteError()}
                if with_kappa:
                    self.metrics = {
                        "MAE": CustomMAE(
                            accelerator=self.accelerator,
                            allow_distributed=allow_distributed,
                        ),
                        "Kappa": CustomKappa(
                            accelerator=self.accelerator,
                            allow_distributed=allow_distributed,
                        ),
                    }
                else:
                    self.metrics = {
                        "MAE": CustomMAE(
                            accelerator=self.accelerator,
                            allow_distributed=allow_distributed,
                        )
                    }

        elif encoder.logit.out_features == 15:  # PHENOTYPING. ugly, to refactor
            self.task_type = "multiclass_classification"
            self.output_transform = softmax_multi_output_transform
            # self.metrics = {'Accuracy': Accuracy(), 'BalancedAccuracy': BalancedAccuracy()}
            self.metrics = {
                "BalancedAccuracy": CustomBalancedAccuracy(
                    accelerator=self.accelerator, allow_distributed=allow_distributed
                )
            }
            logging.info(
                f"[{self.__class__.__name__}] task: {self.task_type}, set metrics: {self.metrics}"
            )

        else:  # MULTIHORIZON
            self.task_type = "multioutput_binary_classification"
            self.output_transform = softmax_multi_output_binary_transform
            metrics = {
                "PR": AveragePrecision,
                "AUC": ROC_AUC,
                "PR_Curve": PrecisionRecallCurve,
                "ROC_Curve": RocCurve,
                "Calibration_Curve": CalibrationCurve,
            }
            self.metrics = {}
            num_outputs = encoder.logit.out_features
            for met in metrics.keys():
                for out in range(num_outputs):
                    self.metrics[met + f"{out}"] = MultiOutput_Binary_Metric(
                        metrics[met], num_outputs=num_outputs, output_dim=out
                    )
        if self.aux_label_idx is not None:
            self.aux_metrics = {
                "MAE_{}".format(str(idx)): MeanAbsoluteError()
                for idx in self.aux_label_idx
            }
        else:
            self.aux_metrics = {}
        logging.info(f"[{self.__class__.__name__}] task type: {self.task_type}")

    def get_auxiliary_labels(self, data):
        HORIZON = self.aux_label_horizon
        IDXS = self.aux_label_idx
        data_4d = data.unsqueeze(0)
        with torch.no_grad():
            padded_batch = torch.nn.functional.pad(
                data_4d[..., IDXS], (0, 0, HORIZON - 1, 0), mode="replicate"
            )[0]
            unfolded = padded_batch.unfold(1, HORIZON, 1)
            if self.aux_label_type == "max":
                labels = unfolded.max(axis=-1)[0]
            elif self.aux_label_type == "min":
                labels = unfolded.min(axis=-1)[0]
            elif self.aux_label_type == "mean":
                labels = unfolded.mean(axis=-1)
        return labels

    def loss_fn(self, output, label, loss_weight):
        """Compute loss based on flattened model output and ground-truth label."""
        if self.task_type == "regression":  # len(label.shape) == 2:
            return torch.nn.functional.mse_loss(output[:, 0], label.float())
        elif self.task_type in ["binary_classification", "multiclass_classification"]:
            return torch.nn.functional.cross_entropy(
                output, label.long(), weight=loss_weight
            )
        else:  # multi-horizon
            return torch.nn.functional.binary_cross_entropy_with_logits(
                output, label, weight=loss_weight
            )

    def step_fn(self, element, loss_weight=None):
        # self.prof.step()
        if len(element) == 2:
            data, labels = element[0], element[1].to(self.device)
            if isinstance(data, list):
                for i in range(len(data)):
                    data[i] = data[i].float().to(self.device)
            else:
                data = data.float().to(self.device)
            mask = torch.ones(labels.shape[:2]).bool()

        elif len(element) == 3:
            data, labels, mask = (
                element[0],
                element[1].to(self.device),
                element[2].to(self.device),
            )
            if isinstance(data, list):
                for i in range(len(data)):
                    data[i] = data[i].float().to(self.device)
            else:
                data = data.float().to(self.device)
        else:
            raise Exception(
                "Loader should return either (data, label) or (data, label, mask)"
            )

        true_length = (
            torch.where(torch.any(data != 0.0, axis=0))[0][-1] + 1
        )  # We need to use data instead of mask
        data = data[:, :true_length]
        labels = labels[:, :true_length]
        mask = mask[:, :true_length]

        out = self.encoder(data)
        if self.aux_label_idx is not None:
            out, aux_out = out
            aux_labels = self.get_auxiliary_labels(data)
            aux_out_flat = torch.masked_select(aux_out, mask.unsqueeze(-1)).reshape(
                -1, aux_out.shape[-1]
            )
            aux_label_flat = torch.masked_select(
                aux_labels, mask.unsqueeze(-1)
            ).reshape(-1, aux_out.shape[-1])
            aux_loss = self.aux_label_weight * torch.nn.functional.mse_loss(
                aux_out_flat, aux_label_flat.float()
            )
            aux_op = [aux_out_flat, aux_label_flat]
        else:
            aux_op = []
            aux_loss = 0.0

        if (
            self.task_type != "multioutput_binary_classification"
        ):  # len(labels.shape) <= 2:
            out_flat = torch.masked_select(out, mask.unsqueeze(-1)).reshape(
                -1, out.shape[-1]
            )
            label_flat = torch.masked_select(labels, mask)
        else:  # multihorizon
            out_flat = torch.masked_select(out, mask.unsqueeze(-1))
            label_flat = torch.masked_select(
                labels, mask.unsqueeze(-1)
            )  # .reshape(-1, labels.shape[-1])

        # avoid nan loss for completely masked samples
        # TODO: why can there be completely masked samples at all?
        if len(out_flat) == 0:
            assert len(out_flat) == len(label_flat)
            # loss = torch.tensor(aux_loss, device=out.device)
            loss = (
                torch.nan_to_num(self.loss_fn(out_flat, label_flat, loss_weight))
                + aux_loss
            )

        else:
            loss = self.loss_fn(out_flat, label_flat, loss_weight) + aux_loss

        # Apply encoder weight regularizer
        if self.reg_fn is not None:
            # loss += self.reg_weight * self.reg_fn(self.encoder.embedding_layer)
            loss += self.reg_weight * self.reg_fn(
                self.accelerator.unwrap_model(self.encoder).embedding_layer
            )

        clustering_loss = 0
        if self.cluster_reg > 0:
            unwrapped_encoder_embedding = self.accelerator.unwrap_model(
                self.encoder
            ).embedding_layer
            cluster_layer = unwrapped_encoder_embedding.clustering

            feature_tokenzier = unwrapped_encoder_embedding.feature_tokenizer
            feature_matrix = (
                unwrapped_encoder_embedding.extract_embeddings_for_clustering(
                    feature_tokenzier
                )
            )

            cl_assignments = unwrapped_encoder_embedding.clustering_assignments
            num_clusters = unwrapped_encoder_embedding.num_clusters_k
            num_feature_vectors = feature_matrix.shape[0]

            if self.clustering_regularizer_type == "functional_hard":
                cl_assignments_matrix = torch.zeros(
                    (num_feature_vectors, num_clusters), device=feature_matrix.device
                )
                cl_assignments_matrix[
                    np.arange(num_feature_vectors), cl_assignments
                ] = 1
                clustering_loss_unscaled = cluster_layer.regularization(
                    feature_matrix, cl_assignments_matrix
                )

            elif self.clustering_regularizer_type == "functional_soft":
                clustering_loss_unscaled = cluster_layer.soft_regularization(
                    feature_matrix
                )

            else:
                raise NotImplementedError(
                    f"Clustering regularizer type not implemented: {self.clustering_regularizer_type}"
                )

            clustering_loss = self.cluster_reg * clustering_loss_unscaled
            loss += clustering_loss

        return loss, out_flat, label_flat, aux_op, clustering_loss

    def _do_training(self, train_loader, weight, metrics):
        # Training epoch
        train_loss = []
        self.encoder.train()
        for t, elem in tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            disable=not self.accelerator.is_local_main_process,
        ):

            loss, preds, target, aux_op, loss_cluster = self.step_fn(elem, weight)

            # loss.backward()
            self.accelerator.backward(loss)

            self.optimizer.step()
            self.optimizer.zero_grad()

            # Do clustering after optimizer step (updated embeddings)
            unwrapped_encoder_embedding = self.accelerator.unwrap_model(
                self.encoder
            ).embedding_layer
            if self.cluster_splitting and self.accelerator.is_main_process:

                cluster_layer = unwrapped_encoder_embedding.clustering
                feature_tokenzier = unwrapped_encoder_embedding.feature_tokenizer
                feature_matrix = (
                    unwrapped_encoder_embedding.extract_embeddings_for_clustering(
                        feature_tokenzier
                    ).detach()
                )
                # print(f"Feature Matrix: {feature_matrix}")

                new_assignments, updated_centroids = cluster_layer(
                    feature_matrix,
                    prev_assignments=unwrapped_encoder_embedding.clustering_assignments,
                )
                unwrapped_encoder_embedding.clustering_assignments = new_assignments

                if updated_centroids:
                    self.cluster_updates += 1

                if self.clustering_tensorboard is not None:

                    cluster_inertia = cluster_layer.compute_inertia(feature_matrix)
                    cluster_ratio = cluster_layer.functional_ratio(feature_matrix)

                    self.clustering_logger.add_scalar(
                        "cluster/inertia", cluster_inertia, self.clustering_logger_steps
                    )
                    self.clustering_logger.add_scalar(
                        "cluster/ratio", cluster_ratio, self.clustering_logger_steps
                    )
                    self.clustering_logger.add_scalar(
                        "cluster/updates",
                        self.cluster_updates,
                        self.clustering_logger_steps,
                    )
                    self.clustering_logger.add_scalar(
                        "cluster/loss", loss_cluster, self.clustering_logger_steps
                    )

                    cluster_size_map = {
                        str(i): size
                        for i, size in enumerate(
                            unwrapped_encoder_embedding.clustering_split_sizes
                        )
                    }
                    self.clustering_logger.add_scalars(
                        "cluster/sizes", cluster_size_map, self.clustering_logger_steps
                    )

                    if self.clustering_logger_steps % self.clustering_plot_steps == 0:
                        cluster_plot = cluster_layer.plot_clusters(feature_matrix)
                        self.clustering_logger.add_figure(
                            "cluster/plot", cluster_plot, self.clustering_logger_steps
                        )

                    self.clustering_logger_steps += 1

            # Sync Clustering Split from main and update local split on each process
            if self.cluster_splitting:
                accelerate.utils.broadcast(
                    unwrapped_encoder_embedding.clustering_assignments, from_process=0
                )
                unwrapped_encoder_embedding.update_clustering_split(
                    unwrapped_encoder_embedding.clustering_assignments
                )

            train_loss.append(loss)
            for name, metric in sorted(metrics.items(), key=lambda x: x[0]):
                if "Curve" not in name:
                    metric.update(self.output_transform((preds, target)))

            if len(aux_op) > 0:
                for i, metric in enumerate(self.aux_metrics.values()):
                    metric.update((aux_op[0][..., i], aux_op[1][..., i]))

            # if t > 100:
            #     break

        # print(f"World size:  {idist.get_world_size()}")

        # self.accelerator.wait_for_everyone()
        train_metric_results = {}
        for name, metric in sorted(metrics.items(), key=lambda x: x[0]):
            self.accelerator.wait_for_everyone()
            if "Curve" not in name:
                train_metric_results[name] = metric.compute()
                metric.reset()
                # print(f"[MAIN {torch.distributed.get_rank()}] compute {name} done: {train_metric_results[name]}")
        for name, metric in self.aux_metrics.items():
            train_metric_results[name] = metric.compute()
            metric.reset()

        train_loss = torch.sum(torch.cat([t.unsqueeze(0) for t in train_loss])) / (
            t + 1
        )
        train_loss = float(torch.mean(self.accelerator.gather(train_loss)))
        return train_loss, train_metric_results

    @gin.configurable(module="DLWrapper")
    def train(
        self,
        train_dataset,
        val_dataset,
        weight,
        epochs=gin.REQUIRED,
        batch_size=gin.REQUIRED,
        patience=gin.REQUIRED,
        min_delta=gin.REQUIRED,
        save_weights=True,
    ):

        self.set_metrics()
        metrics = self.metrics

        # torch.autograd.set_detect_anomaly(True)  # Check for any nans in gradients

        if not train_dataset.h5_loader.on_RAM:
            self.n_worker = 1
            logging.info(
                "Data is not loaded to RAM, thus number of worker has been set to 1"
            )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.n_worker,
            pin_memory=self.pin_memory,
            prefetch_factor=2,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.n_worker,
            pin_memory=self.pin_memory,
            prefetch_factor=2,
        )

        if isinstance(weight, list):
            weight = torch.FloatTensor(weight).to(self.device)
        elif weight == "balanced":
            weight = torch.FloatTensor(train_dataset.get_balance()).to(self.device)

        best_loss = float("inf")
        epoch_no_improvement = 0

        # Setup Tensorboard Logging
        if self.accelerator.is_main_process:
            train_writer = SummaryWriter(
                os.path.join(self.logdir, "tensorboard", "train")
            )
            val_writer = SummaryWriter(os.path.join(self.logdir, "tensorboard", "val"))
        else:
            train_writer, val_writer = None, None

        if self.cluster_splitting and self.clustering_tensorboard is not None:
            self.clustering_logger_steps = 0
            self.cluster_updates = 0

            if self.accelerator.is_main_process:
                self.clustering_logger = SummaryWriter(
                    os.path.join(self.logdir, "tensorboard", "cluster")
                )
            else:
                self.clustering_logger = None

        # self.prof = torch.profiler.profile(
        #         schedule=torch.profiler.schedule(wait=2, warmup=3, active=5, repeat=1),
        #         on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(self.logdir,'tensorboard')),
        #         record_shapes=False,
        #         profile_memory=False,
        #         with_stack=False,
        #         with_flops=True,
        #         with_modules=True)

        logging.info(f"[SCHEDULER] {self.scheduler}")

        (
            self.encoder,
            self.optimizer,
            train_loader,
            val_loader,
            self.scheduler,
        ) = self.accelerator.prepare(
            self.encoder, self.optimizer, train_loader, val_loader, self.scheduler
        )

        if self.cluster_splitting:
            unw_enc_embs = self.accelerator.unwrap_model(self.encoder).embedding_layer
            unw_enc_embs.clustering_assignments = (
                unw_enc_embs.clustering_assignments.to(self.accelerator.device)
            )

        # logging.info(f"[PROC {self.accelerator.process_index}] gloo: {torch.distributed.is_gloo_available()}")
        # logging.info(f"[PROC {self.accelerator.process_index}] nccl: {torch.distributed.is_nccl_available()}")

        for epoch in range(epochs):

            # Train step
            train_loss, train_metric_results = self._do_training(
                train_loader, weight, metrics
            )
            # print(f"[PROC {self.accelerator.process_index}] train epoch: {epoch} done")

            # Validation step
            val_loss, val_metric_results = self.evaluate(val_loader, metrics, weight)

            if self.scheduler is not None:
                self.scheduler.step()

            # Early stopping
            if val_loss <= best_loss - min_delta:
                best_metrics = val_metric_results
                epoch_no_improvement = 0
                if save_weights:
                    self.save_weights(
                        epoch,
                        os.path.join(self.logdir, f"model_e{epoch}.torch"),
                        delete_previous=True,
                    )
                if self.accelerator.is_main_process:
                    logging.info(f"Validation loss improved to {val_loss:.4f} ")
                best_loss = val_loss
            else:
                epoch_no_improvement += 1
                if self.accelerator.is_main_process:
                    logging.info(
                        f"No improvement on loss for {epoch_no_improvement} epochs"
                    )

            if epoch_no_improvement >= patience:
                if self.accelerator.is_main_process:
                    logging.info(
                        f"No improvement on loss for more than {patience} epochs. We stop training"
                    )
                break

            # Logging
            if self.accelerator.is_main_process:
                train_string = "Train Epoch:{}"
                train_values = [epoch + 1]
                for name, value in train_metric_results.items():
                    if "Curve" not in name.split("_")[-1]:
                        train_string += ", " + name + ":{:.4f}"
                        train_values.append(value)
                        train_writer.add_scalar(name, value, epoch)
                train_writer.add_scalar("Loss", train_loss, epoch)

                if self.scheduler is not None:
                    train_writer.add_scalar(
                        "Learning rate", self.scheduler.get_last_lr()[0], epoch
                    )

                val_string = "Val Epoch:{}"
                val_values = [epoch + 1]
                for name, value in val_metric_results.items():
                    if "Curve" not in name.split("_")[-1]:
                        val_string += ", " + name + ":{:.4f}"
                        val_values.append(value)
                        val_writer.add_scalar(name, value, epoch)
                val_writer.add_scalar("Loss", val_loss, epoch)

                logging.info(train_string.format(*train_values))
                logging.info(val_string.format(*val_values))

        if self.accelerator.is_main_process:
            with open(os.path.join(self.logdir, "val_metrics.pkl"), "wb") as f:
                best_metrics["loss"] = best_loss
                pickle.dump(best_metrics, f)

        self.accelerator.wait_for_everyone()
        with self.accelerator.main_process_first():
            self.load_weights(
                glob.glob(os.path.join(self.logdir, "model*.torch"))[0]
            )  # We load back the best iteration
        self.accelerator.free_memory()
        self.accelerator.wait_for_everyone()

    def test(self, dataset, weight, test_filename="test_metrics.pkl"):

        self.set_metrics(allow_distributed=False, with_kappa=True)

        test_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.n_worker,
            pin_memory=self.pin_memory,
        )

        if isinstance(weight, list):
            weight = torch.FloatTensor(weight).to(self.device)
        test_loss, test_metrics = self.evaluate(
            test_loader, self.metrics, weight, distributed=False
        )

        with open(os.path.join(self.logdir, test_filename), "wb") as f:
            test_loss_cpu = float(test_loss.detach().cpu())
            test_metrics["loss"] = float(test_loss_cpu)
            pickle.dump(test_metrics, f)

        for key, value in test_metrics.items():
            if isinstance(value, float):
                logging.info("Test {} :  {}".format(key, value))

    def evaluate(self, eval_loader, metrics, weight, distributed: bool = True):
        self.encoder.eval()
        eval_loss = []

        # print(f"[PROC {self.accelerator.process_index}] val started")
        with torch.no_grad():
            for v, elem in tqdm(
                enumerate(eval_loader),
                total=len(eval_loader),
                disable=not self.accelerator.is_local_main_process,
            ):

                # print(f"[PROC {self.accelerator.process_index}] val started 2")
                loss, preds, target, aux_op, loss_cluster = self.step_fn(elem, weight)
                # print(f"[PROC {self.accelerator.process_index}] val step done")

                eval_loss.append(loss)
                for name, metric in sorted(metrics.items(), key=lambda x: x[0]):
                    metric.update(self.output_transform((preds, target)))

                if len(aux_op) > 0:
                    for i, metric in enumerate(self.aux_metrics.values()):
                        metric.update((aux_op[0][..., i], aux_op[1][..., i]))

                # if v > 10:
                #     break

            eval_metric_results = {}
            for name, metric in sorted(metrics.items(), key=lambda x: x[0]):
                if distributed:
                    self.accelerator.wait_for_everyone()
                eval_metric_results[name] = metric.compute()
                metric.reset()

            for name, metric in self.aux_metrics.items():
                eval_metric_results[name] = metric.compute()
                metric.reset()

        eval_loss = torch.sum(torch.cat([t.unsqueeze(0) for t in eval_loss])) / (v + 1)
        if distributed:
            eval_loss = float(torch.mean(self.accelerator.gather(eval_loss)))

        return eval_loss, eval_metric_results

    def save_weights(self, epoch, save_path, delete_previous: bool = False):

        if delete_previous and self.accelerator.is_main_process:
            directory = os.path.dirname(os.path.abspath(save_path))
            matches = glob.glob(os.path.join(directory, "model*.torch"))
            if len(matches) > 0:
                for match in matches:
                    os.remove(match)

        self.accelerator.wait_for_everyone()
        unwrapped_encoder = self.accelerator.unwrap_model(self.encoder)
        self.accelerator.save(unwrapped_encoder.state_dict(), save_path)

        # save_model(unwrapped_encoder, self.optimizer, epoch, save_path)

    def load_weights(self, load_path):
        logging.info(
            f"[PROC {self.accelerator.process_index}] load weights: {load_path}"
        )
        unwrapped_encoder = self.accelerator.unwrap_model(self.encoder)
        unwrapped_encoder.load_state_dict(torch.load(load_path))

        # unwrapped_encoder = self.accelerator.unwrap_model(self.encoder)
        # load_model_state(load_path, unwrapped_encoder, optimizer=self.optimizer)


@gin.configurable("MLWrapper")
class MLWrapper(object):
    def __init__(self, model=gin.REQUIRED):
        self.model = model
        self.scaler = None

    def set_logdir(self, logdir):
        self.logdir = logdir

    def set_metrics(self, labels):
        if len(np.unique(labels)) == 2:
            if isinstance(self.model, lightgbm.basic.Booster):
                self.output_transform = lambda x: x
            else:
                self.output_transform = lambda x: x[:, 1]
            self.label_transform = lambda x: x

            self.metrics = {"PR": average_precision_score, "AUC": roc_auc_score}

        elif np.all(labels[:10].astype(int) == labels[:10]):
            self.output_transform = lambda x: np.argmax(x, axis=-1)
            self.label_transform = lambda x: x
            self.metrics = {
                "Accuracy": accuracy_score,
                "BalancedAccuracy": balanced_accuracy_score,
            }

        else:
            if (
                self.scaler is not None
            ):  # We invert transform the labels and predictions if they were scaled.
                self.output_transform = lambda x: self.scaler.inverse_transform(
                    x.reshape(-1, 1)
                )
                self.label_transform = lambda x: self.scaler.inverse_transform(
                    x.reshape(-1, 1)
                )
            else:
                self.output_transform = lambda x: x
                self.label_transform = lambda x: x
            self.metrics = {"MAE": mean_absolute_error}

    def set_scaler(self, scaler):
        self.scaler = scaler

    @gin.configurable(module="MLWrapper")
    def train(
        self,
        train_dataset,
        val_dataset,
        weight,
        patience=gin.REQUIRED,
        save_weights=True,
    ):

        train_rep, train_label = train_dataset.get_data_and_labels()
        val_rep, val_label = val_dataset.get_data_and_labels()
        self.set_metrics(train_label)
        metrics = self.metrics

        if "class_weight" in self.model.get_params().keys():  # Set class weights
            self.model.set_params(class_weight=weight)

        if (
            "eval_set" in inspect.getfullargspec(self.model.fit).args
        ):  # This is lightgbm
            self.model.set_params(random_state=np.random.get_state()[1][0])
            self.model.fit(
                train_rep,
                train_label,
                eval_set=(val_rep, val_label),
                early_stopping_rounds=patience,
            )
            val_loss = list(self.model.best_score_["valid_0"].values())[0]
            model_type = "lgbm"
        else:
            model_type = "sklearn"
            self.model.fit(train_rep, train_label)
            val_loss = 0.0

        if "MAE" in self.metrics.keys():
            val_pred = self.model.predict(val_rep)
            train_pred = self.model.predict(train_rep)
        else:
            val_pred = self.model.predict_proba(val_rep)
            train_pred = self.model.predict_proba(train_rep)

        train_metric_results = {}
        train_string = "Train Results :"
        train_values = []
        val_string = "Val Results :" + "loss" + ":{:.4f}"
        val_values = [val_loss]
        val_metric_results = {"loss": val_loss}
        for name, metric in metrics.items():
            train_metric_results[name] = metric(
                self.label_transform(train_label), self.output_transform(train_pred)
            )
            val_metric_results[name] = metric(
                self.label_transform(val_label), self.output_transform(val_pred)
            )
            train_string += ", " + name + ":{:.4f}"
            val_string += ", " + name + ":{:.4f}"
            train_values.append(train_metric_results[name])
            val_values.append(val_metric_results[name])
        logging.info(train_string.format(*train_values))
        logging.info(val_string.format(*val_values))

        if save_weights:
            if model_type == "lgbm":
                self.save_weights(
                    save_path=os.path.join(self.logdir, "model.txt"),
                    model_type=model_type,
                )
            else:
                self.save_weights(
                    save_path=os.path.join(self.logdir, "model.joblib"),
                    model_type=model_type,
                )

        with open(os.path.join(self.logdir, "val_metrics.pkl"), "wb") as f:
            pickle.dump(val_metric_results, f)

    def test(self, dataset, weight):
        test_rep, test_label = dataset.get_data_and_labels()
        self.set_metrics(test_label)
        if "MAE" in self.metrics.keys() or isinstance(
            self.model, lightgbm.basic.Booster
        ):  # If we reload a LGBM classifier
            test_pred = self.model.predict(test_rep)
        else:
            test_pred = self.model.predict_proba(test_rep)

        test_string = "Test Results :"
        test_values = []
        test_metric_results = {}
        for name, metric in self.metrics.items():
            test_metric_results[name] = metric(
                self.label_transform(test_label), self.output_transform(test_pred)
            )
            test_string += ", " + name + ":{:.4f}"
            test_values.append(test_metric_results[name])

        logging.info(test_string.format(*test_values))
        with open(os.path.join(self.logdir, "test_metrics.pkl"), "wb") as f:
            pickle.dump(test_metric_results, f)

    def save_weights(self, save_path, model_type="lgbm"):
        if model_type == "lgbm":
            self.model.booster_.save_model(save_path)
        else:
            joblib.dump(self.model, save_path)

    def load_weights(self, load_path):
        if load_path.split(".")[-1] == "txt":
            self.model = lightgbm.Booster(model_file=load_path)
        else:
            with open(load_path, "rb") as f:
                self.model = joblib.load(f)
