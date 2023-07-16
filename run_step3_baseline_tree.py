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

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score
)

from datasets import load_dataset

from model import *
from utils.seed import set_seed


def run_baselines(cur_ds_name, cur_model_name):
    logger.info(f"\n\n\n *** *** cur_ds_name: {cur_ds_name}; cur_model_name: {cur_model_name}")
    data = load_dataset("yuweiyin/FinBench", cur_ds_name, cache_dir=cache_ds)
    train_set = data["train"] if "train" in data else []
    val_set = data["validation"] if "train" in data else []
    test_set = data["test"] if "train" in data else []

    x_key = "X_ml"
    # x_key = "X_ml_unscale"
    train_X_ml, train_y = np.asarray(train_set[x_key], dtype=np.float32), np.asarray(train_set["y"], dtype=np.int64)
    val_X_ml, val_y = np.asarray(val_set[x_key], dtype=np.float32), np.asarray(val_set["y"], dtype=np.int64)
    test_X_ml, test_y = np.asarray(test_set[x_key], dtype=np.float32), np.asarray(test_set["y"], dtype=np.int64)

    # get pos ratio of the training set for loss computing
    total_y = float(len(train_y))
    pos_y = float(sum(train_y))
    assert total_y >= pos_y > 0.0
    neg_to_pos = float((total_y - pos_y) / pos_y)
    pos_ratio = float(pos_y / total_y)
    logger.info(f">>> pos_ratio = {pos_ratio}; neg_to_pos = {neg_to_pos}")
    class_weight = {0: 1.0, 1: neg_to_pos}
    args.class_weight = class_weight

    model = MODEL_DICT[cur_model_name](args)
    if grid_search and hasattr(model, "param_grid") and isinstance(model.param_grid, dict):
        clf = GridSearchCV(model, model.param_grid, cv=5, scoring="f1")
        clf.fit(train_X_ml, train_y)
        logger.info(f"clf.best_params_: {clf.best_params_}")
        best_model = clf.best_estimator_
    else:
        model.fit(train_X_ml, train_y)
        best_model = model

    y_pred_train = best_model.predict(train_X_ml)
    acc_train, f1_train, auc_train, p_train, r_train, avg_p_train = \
        accuracy_score(train_y, y_pred_train), f1_score(train_y, y_pred_train), roc_auc_score(train_y, y_pred_train), \
        precision_score(train_y, y_pred_train), recall_score(train_y, y_pred_train), \
        average_precision_score(train_y, y_pred_train)

    y_pred_val = best_model.predict(val_X_ml)
    acc_val, f1_val, auc_val, p_val, r_val, avg_p_val = \
        accuracy_score(val_y, y_pred_val), f1_score(val_y, y_pred_val), roc_auc_score(val_y, y_pred_val), \
        precision_score(val_y, y_pred_val), recall_score(val_y, y_pred_val), \
        average_precision_score(val_y, y_pred_val)

    y_pred_test = best_model.predict(test_X_ml)
    acc_test, f1_test, auc_test, p_test, r_test, avg_p_test = \
        accuracy_score(test_y, y_pred_test), f1_score(test_y, y_pred_test), roc_auc_score(test_y, y_pred_test), \
        precision_score(test_y, y_pred_test), recall_score(test_y, y_pred_test), \
        average_precision_score(test_y, y_pred_test)

    logger.info(f">>> Dataset = {cur_ds_name}; Model = {cur_model_name}")
    logger.info(f">>> [{cur_ds_name}---{cur_model_name}] Evaluation (Training set): "
                "Accuracy = %.4f; F1 = %.4f; AUC = %.4f; Precision = %.4f; Recall = %.4f; Avg Precision = %.4f" % (
                    acc_train, f1_train, auc_train, p_train, r_train, avg_p_train))
    logger.info(f">>> [{cur_ds_name}---{cur_model_name}] Evaluation (Validation set): "
                "Accuracy = %.4f; F1 = %.4f; AUC = %.4f; Precision = %.4f; Recall = %.4f; Avg Precision = %.4f" % (
                    acc_val, f1_val, auc_val, p_val, r_val, avg_p_val))
    logger.info(f">>> [{cur_ds_name}---{cur_model_name}] Evaluation (Test set): "
                "Accuracy = %.4f; F1 = %.4f; AUC = %.4f; Precision = %.4f; Recall = %.4f; Avg Precision = %.4f" % (
                    acc_test, f1_test, auc_test, p_test, r_test, avg_p_test))

    eval_results_train[f"{cur_ds_name}---{cur_model_name}---train"] = (
        acc_train, f1_train, auc_train, p_train, r_train, avg_p_train)
    eval_results_val[f"{cur_ds_name}---{cur_model_name}---val"] = (
        acc_val, f1_val, auc_val, p_val, r_val, avg_p_val)
    eval_results_test[f"{cur_ds_name}---{cur_model_name}---test"] = (
        acc_test, f1_test, auc_test, p_test, r_test, avg_p_test)

    del data
    del model
    gc.collect()


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s - %(levelname)s - %(name)s] -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Baseline args")

    parser.add_argument("--seed", type=int, default=0, help="Seed of random modules")
    parser.add_argument("--ds_name", type=str, default="cd1", help="Specify which dataset to use.",
                        choices=["cd1", "cd2", "ld1", "ld2", "ld3", "cf1", "cf2", "cc1", "cc2", "cc3"])
    parser.add_argument("--model_name", type=str, default="LogisticRegression", help="Specify which model to use.",
                        choices=["RandomForestClassifier", "XGBClassifier", "CatBoostClassifier", "LGBMClassifier"])
    parser.add_argument("--bsz", type=int, default=128, help="TrainingArguments: per_device_train/eval_batch_size")
    parser.add_argument("--epoch", type=int, default=100, help="TrainingArguments: num_train_epochs")
    parser.add_argument("--objective", type=str, default="classification",
                        choices=["classification", "binary", "regression"],
                        help="The type of the current task")
    parser.add_argument("--grid_search", action="store_true", help="GridSearch")

    args = parser.parse_args()
    logger.info(args)

    seed = int(args.seed)
    ds_name = str(args.ds_name)
    model_name = str(args.model_name)
    bsz = int(args.bsz)
    epoch = int(args.epoch)
    grid_search = bool(args.grid_search)

    set_seed(seed)

    cache_dir = "~/.cache/huggingface/"
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    cache_model = os.path.join(cache_dir, "models")
    cache_ds = os.path.join(cache_dir, "datasets")

    eval_results_train = dict({})  # dict{ds_name: Tuple(acc, f1, auc, p, r, avg_p)}
    eval_results_val = dict({})
    eval_results_test = dict({})

    run_baselines(cur_ds_name=ds_name, cur_model_name=model_name)

    logger.info(f"\n\n>>> END: eval_results_train: {eval_results_train}")
    logger.info(f">>> END: eval_results_val: {eval_results_val}")
    logger.info(f">>> END: eval_results_test: {eval_results_test}")

    sys.exit(0)
