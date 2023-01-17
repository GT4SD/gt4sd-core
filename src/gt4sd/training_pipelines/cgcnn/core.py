#
# MIT License
#
# Copyright (c) 2022 GT4SD team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
"""Cgcnn training utilities."""

import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from random import sample
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch import Tensor
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from ...frameworks.cgcnn.data import CIFData, collate_pool, get_train_val_test_loader
from ...frameworks.cgcnn.model import CrystalGraphConvNet, Normalizer
from ..core import TrainingPipeline, TrainingPipelineArguments

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class CGCNNTrainingPipeline(TrainingPipeline):
    """CGCNN training pipelines for crystals."""

    def train(  # type: ignore
        self,
        training_args: Dict[str, Any],
        model_args: Dict[str, Any],
        dataset_args: Dict[str, Any],
    ) -> None:
        """Generic training function for CGCNN models.

        Args:
            training_args: training arguments passed to the configuration.
            model_args: model arguments passed to the configuration.
            dataset_args: dataset arguments passed to the configuration.

        Raises:
            NotImplementedError: the generic trainer does not implement the pipeline.
        """

        training_args["disable_cuda"] = (
            training_args["disable_cuda"] or not torch.cuda.is_available()
        )

        if training_args["task"] == "regression":
            best_mae_error = 1e10
        else:
            best_mae_error = 0.0

        # load data
        dataset = CIFData(dataset_args["datapath"])
        collate_fn = collate_pool
        train_loader, val_loader, test_loader = get_train_val_test_loader(  # type: ignore
            dataset=dataset,
            collate_fn=collate_fn,
            batch_size=training_args["batch_size"],
            num_workers=training_args["workers"],
            pin_memory=training_args["disable_cuda"],
            train_size=dataset_args["train_size"],
            val_size=dataset_args["val_size"],
            test_size=dataset_args["test_size"],
            return_test=True,
        )

        # obtain target value normalizer
        if training_args["task"] == "classification":
            normalizer = Normalizer(torch.zeros(2))
            normalizer.load_state_dict({"mean": 0.0, "std": 1.0})
        else:
            if len(dataset) < 500:
                logger.warning(
                    "Dataset has less than 500 data points. "
                    "Lower accuracy is expected. "
                )
                sample_data_list = [dataset[i] for i in range(len(dataset))]
            else:
                sample_data_list = [
                    dataset[i] for i in sample(range(len(dataset)), 500)
                ]
            _, sample_target, _ = collate_pool(sample_data_list)
            normalizer = Normalizer(sample_target)

        # build model
        structures, _, _ = dataset[0]
        orig_atom_fea_len = structures[0].shape[-1]
        nbr_fea_len = structures[1].shape[-1]  # type: ignore
        model = CrystalGraphConvNet(
            orig_atom_fea_len,
            nbr_fea_len,
            atom_fea_len=model_args["atom_fea_len"],
            n_conv=model_args["n_conv"],
            h_fea_len=model_args["h_fea_len"],
            n_h=model_args["n_h"],
            classification=True if training_args["task"] == "classification" else False,
        )
        if not training_args["disable_cuda"]:
            model.cuda()

        # define loss func and optimizer
        if training_args["task"] == "classification":
            criterion = nn.NLLLoss()
        else:
            criterion = nn.MSELoss()  # type: ignore

        if training_args["optim"] == "SGD":
            optimizer = optim.SGD(
                model.parameters(),
                training_args["lr"],
                momentum=training_args["momentum"],
                weight_decay=training_args["weight_decay"],
            )
        elif training_args["optim"] == "Adam":
            optimizer = optim.Adam(  # type: ignore
                model.parameters(),
                training_args["lr"],
                weight_decay=training_args["weight_decay"],
            )
        else:
            raise NameError("Only SGD or Adam is allowed as optimizer")

        # optionally resume from a checkpoint
        if training_args["resume"]:
            if os.path.isfile(training_args["resume"]):
                logger.info("loading checkpoint '{}'".format(training_args["resume"]))
                checkpoint = torch.load(training_args["resume"])
                training_args["start_epoch"] = checkpoint["epoch"]
                best_mae_error = checkpoint["best_mae_error"]
                model.load_state_dict(checkpoint["state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                normalizer.load_state_dict(checkpoint["normalizer"])
                logger.info(
                    "loaded checkpoint '{}' (epoch {})".format(
                        training_args["resume"], checkpoint["epoch"]
                    )
                )
            else:
                logger.info(
                    "no checkpoint found at '{}'".format(training_args["resume"])
                )

        scheduler = MultiStepLR(
            optimizer, milestones=[training_args["lr_milestone"]], gamma=0.1
        )

        for epoch in range(training_args["start_epoch"], training_args["epochs"]):
            # train for one epoch
            train(
                train_loader,
                model,
                criterion,
                optimizer,
                epoch,
                normalizer,
                training_args["disable_cuda"],
                training_args["task"],
                training_args["print_freq"],
            )

            # evaluate on validation set
            mae_error = validate(
                val_loader,
                model,
                criterion,
                normalizer,
                training_args["disable_cuda"],
                training_args["task"],
                training_args["print_freq"],
                test=True,
            )

            if mae_error != mae_error:
                raise ValueError("mae_error is NaN")

            scheduler.step()

            # remember the best mae_eror and save checkpoint
            if training_args["task"] == "regression":
                is_best = mae_error < best_mae_error
                best_mae_error = min(mae_error, best_mae_error)
            else:
                is_best = mae_error > best_mae_error
                best_mae_error = max(mae_error, best_mae_error)
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_mae_error": best_mae_error,
                    "optimizer": optimizer.state_dict(),
                    "normalizer": normalizer.state_dict(),
                    "training_args": training_args,
                    "model_args": model_args,
                    "dataset_args": dataset_args,
                },
                is_best,
                training_args["output_path"],
            )

        # test best model
        logger.info("Evaluate Model on Test Set")
        best_checkpoint = torch.load("model_best.pth.tar")
        model.load_state_dict(best_checkpoint["state_dict"])
        validate(
            test_loader,
            model,
            criterion,
            normalizer,
            training_args["disable_cuda"],
            training_args["task"],
            training_args["print_freq"],
            test=True,
        )


def train(
    train_loader: Union[DataLoader[Any], Any],
    model: CrystalGraphConvNet,
    criterion: Union[nn.NLLLoss, nn.MSELoss],
    optimizer: Union[optim.SGD, optim.Adam],
    epoch: int,
    normalizer: Normalizer,
    disable_cuda: bool,
    task: str,
    print_freq: int,
) -> None:
    """Train step for cgcnn models.

    Args:
        train_loader: Dataloader for the training set.
        model: CGCNN model.
        criterion: loss function.
        optimizer: Optimizer to be used.
        epoch: Epoch number.
        normalizer: Normalize.
        disable_cuda: Disable CUDA.
        task: Training task.
        print_freq: Print frequency.
    """

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    mae_errors = AverageMeter()
    accuracies = AverageMeter()
    precisions = AverageMeter()
    recalls = AverageMeter()
    fscores = AverageMeter()
    auc_scores = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if not disable_cuda:
            input_var = (
                Variable(input[0].cuda(non_blocking=True)),
                Variable(input[1].cuda(non_blocking=True)),
                input[2].cuda(non_blocking=True),
                [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]],
            )
        else:
            input_var = (Variable(input[0]), Variable(input[1]), input[2], input[3])
        # normalize target
        if task == "regression":
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()
        if not disable_cuda:
            target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            target_var = Variable(target_normed)

        # compute output
        output = model(*input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        if task == "regression":
            mae_error = mae(normalizer.denorm(output.data.cpu()), target)
            losses.update(loss.data.cpu(), target.size(0))
            mae_errors.update(mae_error, target.size(0))  # type: ignore
        else:
            accuracy, precision, recall, fscore, auc_score = class_eval(
                output.data.cpu(), target
            )
            losses.update(loss.data.cpu().item(), target.size(0))
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            if task == "regression":
                logger.info(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})".format(
                        epoch,
                        i,
                        len(train_loader),
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=losses,
                        mae_errors=mae_errors,
                    )
                )
            else:
                logger.info(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accu {accu.val:.3f} ({accu.avg:.3f})\t"
                    "Precision {prec.val:.3f} ({prec.avg:.3f})\t"
                    "Recall {recall.val:.3f} ({recall.avg:.3f})\t"
                    "F1 {f1.val:.3f} ({f1.avg:.3f})\t"
                    "AUC {auc.val:.3f} ({auc.avg:.3f})".format(
                        epoch,
                        i,
                        len(train_loader),
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=losses,
                        accu=accuracies,
                        prec=precisions,
                        recall=recalls,
                        f1=fscores,
                        auc=auc_scores,
                    )
                )


def validate(
    val_loader: Union[DataLoader[Any], Any],
    model: CrystalGraphConvNet,
    criterion: Union[nn.MSELoss, nn.NLLLoss],
    normalizer: Normalizer,
    disable_cuda: bool,
    task: str,
    print_freq: int,
    test: bool = False,
) -> float:
    """Validation step for cgcnn models.

    Args:
        val_loader: Dataloader for the validation set.
        model: CGCNN model.
        criterion: loss function.
        normalizer: Normalize.
        disable_cuda: Disable CUDA.
        task: Training task.
        print_freq: Print frequency.
        test: test or only validate using the given dataset.

    Returns:
       average auc or mae depending on the training task.
    """

    batch_time = AverageMeter()
    losses = AverageMeter()
    mae_errors = AverageMeter()
    accuracies = AverageMeter()
    precisions = AverageMeter()
    recalls = AverageMeter()
    fscores = AverageMeter()
    auc_scores = AverageMeter()
    test_targets = []
    test_preds = []
    test_cif_ids = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target, batch_cif_ids) in enumerate(val_loader):
        if not disable_cuda:
            with torch.no_grad():
                input_var = (
                    Variable(input[0].cuda(non_blocking=True)),
                    Variable(input[1].cuda(non_blocking=True)),
                    input[2].cuda(non_blocking=True),
                    [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]],
                )
        else:
            with torch.no_grad():
                input_var = (Variable(input[0]), Variable(input[1]), input[2], input[3])
        if task == "regression":
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()
        if not disable_cuda:
            with torch.no_grad():
                target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            with torch.no_grad():
                target_var = Variable(target_normed)

        # compute output
        output = model(*input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        if task == "regression":
            mae_error = mae(normalizer.denorm(output.data.cpu()), target)
            losses.update(loss.data.cpu().item(), target.size(0))
            mae_errors.update(mae_error, target.size(0))  # type: ignore
            if test:
                test_pred = normalizer.denorm(output.data.cpu())
                test_target = target
                test_preds += test_pred.view(-1).tolist()
                test_targets += test_target.view(-1).tolist()
                test_cif_ids += batch_cif_ids
        else:
            accuracy, precision, recall, fscore, auc_score = class_eval(
                output.data.cpu(), target
            )
            losses.update(loss.data.cpu().item(), target.size(0))
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))
            if test:
                test_pred = torch.exp(output.data.cpu())
                test_target = target
                assert test_pred.shape[1] == 2
                test_preds += test_pred[:, 1].tolist()
                test_targets += test_target.view(-1).tolist()
                test_cif_ids += batch_cif_ids

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            if task == "regression":
                logger.info(
                    "Test: [{0}/{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})".format(
                        i,
                        len(val_loader),
                        batch_time=batch_time,
                        loss=losses,
                        mae_errors=mae_errors,
                    )
                )
            else:
                logger.info(
                    "Test: [{0}/{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accu {accu.val:.3f} ({accu.avg:.3f})\t"
                    "Precision {prec.val:.3f} ({prec.avg:.3f})\t"
                    "Recall {recall.val:.3f} ({recall.avg:.3f})\t"
                    "F1 {f1.val:.3f} ({f1.avg:.3f})\t"
                    "AUC {auc.val:.3f} ({auc.avg:.3f})".format(
                        i,
                        len(val_loader),
                        batch_time=batch_time,
                        loss=losses,
                        accu=accuracies,
                        prec=precisions,
                        recall=recalls,
                        f1=fscores,
                        auc=auc_scores,
                    )
                )

    if task == "regression":
        logger.info("MAE {mae_errors.avg:.3f}".format(mae_errors=mae_errors))
        return mae_errors.avg
    else:
        logger.info("AUC {auc.avg:.3f}".format(auc=auc_scores))
        return auc_scores.avg


def mae(prediction: Tensor, target: Tensor) -> Tensor:
    """Computes the mean absolute error between prediction and target.

    Args:
        prediction: torch.Tensor (N, 1)
        target: torch.Tensor (N, 1)

    Returns:
        the computed mean absolute error.
    """
    return torch.mean(torch.abs(target - prediction))


def class_eval(
    prediction: Tensor, target: Tensor
) -> Tuple[float, float, float, float, float]:
    """Class evaluation.

    Args:
        prediction: Predictions.
        target: Groundtruth.

    Returns:
        Computed accuracy, precision, recall, fscore, and auc_score.

    """

    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if not target_label.shape:
        target_label = np.asarray([target_label])
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average="binary"
        )
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        """Initialize an AverageMeter object."""

        self.reset()

    def reset(self) -> None:
        """Reset values to 0."""

        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """Update values of the AverageMeter.

        Args:
            val: value to be added.
            n: count.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(
    state: object, is_best: bool, path: str = ".", filename: str = "checkpoint.pth.tar"
) -> None:
    """Save CGCNN checkpoint.

    Args:
        state: checkpoint's object.
        is_best: whether the given checkpoint has the best performance or not.
        path: path to save the checkpoint.
        filename: checkpoint's filename.

    """

    torch.save(state, os.path.join(path, filename))
    if is_best:
        shutil.copyfile(filename, os.path.join(path, "model_best.pth.tar"))


@dataclass
class CGCNNDataArguments(TrainingPipelineArguments):
    """Data arguments related to diffusion trainer."""

    __name__ = "dataset_args"

    datapath: str = field(
        metadata={
            "help": "Path to the dataset."
            "The dataset should follow the directory structure as described in https://github.com/txie-93/cgcnn"
        },
    )
    train_size: Optional[int] = field(
        default=None, metadata={"help": "Number of training data to be loaded."}
    )
    val_size: Optional[int] = field(
        default=None, metadata={"help": "Number of validation data to be loaded."}
    )
    test_size: Optional[int] = field(
        default=None, metadata={"help": "Number of testing data to be loaded."}
    )


@dataclass
class CGCNNModelArguments(TrainingPipelineArguments):
    """Model arguments related to CGCNN trainer."""

    __name__ = "model_args"

    atom_fea_len: int = field(
        default=64, metadata={"help": "Number of hidden atom features in conv layers."}
    )
    h_fea_len: int = field(
        default=128, metadata={"help": "Number of hidden features after pooling."}
    )
    n_conv: int = field(default=3, metadata={"help": "Number of conv layers."})
    n_h: int = field(
        default=1, metadata={"help": "Number of hidden layers after pooling."}
    )


@dataclass
class CGCNNTrainingArguments(TrainingPipelineArguments):
    """Training arguments related to CGCNN trainer."""

    __name__ = "training_args"

    task: str = field(
        default="regression",
        metadata={"help": "Select the type of the task."},
    )
    output_path: str = field(
        default=".",
        metadata={"help": "Path to the store the checkpoints."},
    )
    disable_cuda: bool = field(default=False, metadata={"help": "Disable CUDA."})
    workers: int = field(
        default=0, metadata={"help": "Number of data loading workers."}
    )
    epochs: int = field(default=30, metadata={"help": "Number of total epochs to run."})
    start_epoch: int = field(
        default=0, metadata={"help": "Manual epoch number (useful on restarts)."}
    )
    batch_size: int = field(default=256, metadata={"help": "Mini-batch size."})
    lr: float = field(default=0.01, metadata={"help": "Initial learning rate."})
    lr_milestone: float = field(
        default=100, metadata={"help": "Milestone for scheduler."}
    )
    momentum: float = field(default=0.9, metadata={"help": "Momentum."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay."})
    print_freq: int = field(default=10, metadata={"help": "Print frequency."})
    resume: str = field(default="", metadata={"help": "Path to latest checkpoint."})
    optim: str = field(default="SGD", metadata={"help": "Optimizer."})


@dataclass
class CGCNNSavingArguments(TrainingPipelineArguments):
    """Saving arguments related to CGCNN trainer."""

    __name__ = "saving_args"
