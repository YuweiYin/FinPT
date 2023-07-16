#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__author__ = "@YuweiYin"
"""

import os
import sys
import logging
import argparse
import gc

import numpy as np

import torch
from datasets import load_dataset

from model import *
from criterion import *
from utils.seed import set_seed


def run_baselines_nn(cur_ds_name, cur_model_name):
    logger.info(f"\n\n\n *** *** cur_ds_name: {cur_ds_name}; cur_model_name: {cur_model_name}")
    data = load_dataset("yuweiyin/FinBench", cur_ds_name, cache_dir=cache_ds)
    train_set = data["train"] if "train" in data else []
    val_set = data["validation"] if "train" in data else []
    test_set = data["test"] if "train" in data else []

    args.device = device
    args.num_classes = train_set[0]["num_classes"]  # int (The total number of classes)
    args.num_features = train_set[0]["num_features"]  # int (The total number of features)
    args.num_idx = train_set[0]["num_idx"]  # List[int] (The indices of the numerical datatype columns)
    args.cat_idx = train_set[0]["cat_idx"]  # List[int] (The indices of the categorical datatype columns)
    args.cat_dim = train_set[0]["cat_dim"]  # List[int] (The dimension of each categorical column)
    args.cat_str = train_set[0]["cat_str"]  # List[List[str]] (The category names of categorical columns)
    args.col_name = train_set[0]["col_name"]  # List[str] (The name of each column)

    x_key = "X_ml"
    # x_key = "X_ml_unscale"
    train_X_ml, train_y = np.asarray(train_set[x_key], dtype=np.float32), np.asarray(train_set["y"], dtype=np.int64)
    val_X_ml, val_y = np.asarray(val_set[x_key], dtype=np.float32), np.asarray(val_set["y"], dtype=np.int64)
    test_X_ml, test_y = np.asarray(test_set[x_key], dtype=np.float32), np.asarray(test_set["y"], dtype=np.int64)

    assert isinstance(criterion_name, str) and criterion_name in CRITERION_DICT
    criterion = CRITERION_DICT[criterion_name]()
    logger.info(criterion)

    model = MODEL_DICT[cur_model_name](args=args)
    # model = model.to(device=device)

    eval_results_train[cur_ds_name] = []
    eval_results_val[cur_ds_name] = []
    eval_results_test[cur_ds_name] = []

    model.fit(X=train_X_ml, y=train_y, X_val=val_X_ml, y_val=val_y, optimizer=None, criterion=criterion)

    train_eval_res = model.evaluate(X=train_X_ml, y=train_y)
    val_eval_res = model.evaluate(X=val_X_ml, y=val_y)
    test_eval_res = model.evaluate(X=test_X_ml, y=test_y)

    logger.info(f">>> Dataset = {cur_ds_name}; Model = {cur_model_name}")
    logger.info(f">>> [{cur_ds_name}---{cur_model_name}] Evaluation (Training set): {train_eval_res}")
    logger.info(f">>> [{cur_ds_name}---{cur_model_name}] Evaluation (Validation set): {val_eval_res}")
    logger.info(f">>> [{cur_ds_name}---{cur_model_name}] Evaluation (Test set): {test_eval_res}")

    del data
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s - %(levelname)s - %(name)s] -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Baseline-NN args")

    parser.add_argument("--cuda", type=str, default="cpu", help="Specify which device to use")
    parser.add_argument("--seed", type=int, default=0, help="Seed of random modules")
    parser.add_argument("--task", type=str, default="binary", help="binary/classification OR regression")
    parser.add_argument("--ds_name", type=str, default="cd1", help="Specify which dataset to use.",
                        choices=["cd1", "cd2", "ld1", "ld2", "ld3", "cf1", "cf2", "cc1", "cc2", "cc3"])
    parser.add_argument("--model_name", type=str, default="DeepFM", help="Specify which model to use.",
                        choices=["DeepFM", "STG", "VIME", "TabNet"])
    parser.add_argument("--model_name", type=str, default="MLP", help="Specify which model to use")
    parser.add_argument("--optimizer_name", type=str, default="AdamW", help="Specify which optimizer to use")
    parser.add_argument("--criterion_name", type=str, default="CrossEntropyLoss", help="Specify which criterion to use")
    parser.add_argument("--lr", type=float, default=float(3e-5), help="Learning rate")
    parser.add_argument("--bsz", type=int, default=128, help="TrainingArguments: per_device_train/eval_batch_size")
    parser.add_argument("--epoch", type=int, default=100, help="TrainingArguments: num_train_epochs")
    parser.add_argument("--early_stopping_rounds", type=int, default=100, help="Early-stopping rounds")
    parser.add_argument("--logging_period", type=int, default=100, help="logging_period")
    parser.add_argument("--objective", type=str, default="classification", help="The type of the current task",
                        choices=["classification", "binary", "regression"])

    args = parser.parse_args()
    logger.info(args)

    cuda = str(args.cuda)
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda
    has_cuda = torch.cuda.is_available()
    cnt_cuda = torch.cuda.device_count()
    device = torch.device("cpu" if not has_cuda else f"cuda")
    logger.info(f"has_cuda: {has_cuda}; cnt_cuda: {cnt_cuda}; device: {device}")

    seed = int(args.seed)
    task = str(args.task)
    ds_name = str(args.ds_name)
    model_name = str(args.model_name)
    optimizer_name = str(args.optimizer_name)
    criterion_name = str(args.criterion_name)
    lr = float(args.lr)
    bsz = int(args.bsz)
    epoch = int(args.epoch)
    early_stopping_rounds = int(args.early_stopping_rounds)
    logging_period = int(args.logging_period)

    set_seed(seed)

    cache_dir = "~/.cache/huggingface/"
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    cache_model = os.path.join(cache_dir, "models")
    cache_ds = os.path.join(cache_dir, "datasets")

    eval_results_train = dict({})  # dict{ds_name: List[Tuple(epoch_idx, acc, f1, auc, p, r, avg_p)]}
    eval_results_val = dict({})
    eval_results_test = dict({})

    run_baselines_nn(cur_ds_name=ds_name, cur_model_name=model_name)

    logger.info(f">>> END: eval_results_train: {eval_results_train}")
    logger.info(f">>> END: eval_results_val: {eval_results_val}")
    logger.info(f">>> END: eval_results_test: {eval_results_test}")

    sys.exit(0)
