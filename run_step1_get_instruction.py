#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__author__ = "@YuweiYin"
"""

import os
import sys
import json
import logging
import argparse

from datasets import load_dataset


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s - %(levelname)s - %(name)s] -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Step1 Get_Instruction Args")
    args = parser.parse_args()

    logger.info(args)

    cache_dir = "~/.cache/huggingface/"
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    cache_model = os.path.join(cache_dir, "models")
    cache_ds = os.path.join(cache_dir, "datasets")

    profile_root_dir = os.path.join("./data/profile")
    ds_name_list = ["cd1", "cd2", "ld1", "ld2", "ld3", "cf1", "cf2", "cc1", "cc2", "cc3"]
    for ds_name in ds_name_list:
        logger.info(f"\n\n>>> ds_name: {ds_name}")
        profile_dir = os.path.join(profile_root_dir, ds_name)
        os.makedirs(profile_dir, exist_ok=True)
        try:
            data = load_dataset("yuweiyin/FinBench", ds_name, cache_dir=cache_ds)

            if "train" in data:
                logger.info(f">>> len(data['train']) = {len(data['train'])}")
            if "validation" in data:
                logger.info(f">>> len(data['validation']) = {len(data['validation'])}")
            if "test" in data:
                logger.info(f">>> len(data['test']) = {len(data['test'])}")

            for data_split in ["train", "validation", "test"]:
                if data_split in data:
                    cur_data = data[data_split]
                else:
                    logger.info(f">>> >>> {data_split} NOT in data")
                    continue  # should NOT enter here

                instruction_path = os.path.join(profile_dir, f"instruction_for_profile_X_{data_split}.jsonl")
                with open(instruction_path, mode="w", encoding="utf-8") as fp_out:
                    for instance in cur_data:
                        # X_ml = instance["X_ml"]  # List[float] (The tabular data array of the current instance)
                        X_ml_unscale = instance["X_ml_unscale"]  # List[float] (Scaled tabular data array)
                        # y = instance["y"]  # int (The label / ground-truth)
                        # num_classes = instance["num_classes"]  # int (The total number of classes)
                        # num_features = instance["num_features"]  # int (The total number of features)
                        num_idx = instance["num_idx"]  # List[int] (The indices of the numerical datatype columns)
                        cat_idx = instance["cat_idx"]  # List[int] (The indices of the categorical datatype columns)
                        # cat_dim = instance["cat_dim"]  # List[int] (The dimension of each categorical column)
                        cat_str = instance["cat_str"]  # List[List[str]] (The category names of each column)
                        col_name = instance["col_name"]  # List[str] (The name of each column)
                        assert len(X_ml_unscale) == len(num_idx) + len(cat_idx) == len(col_name)
                        num_idx_set = set(num_idx)
                        cat_idx_set = set(cat_idx)
                        col_idx_2_cat_idx = dict({
                            col_idx: cat_idx for cat_idx, col_idx in enumerate(cat_idx)
                        })
                        # Construct the customer profiles
                        instruction = "Construct a concise customer profile description " \
                                      "including all the following information:\n"
                        for col_idx, x in enumerate(X_ml_unscale):
                            cur_col_name = col_name[col_idx]
                            if ds_name == "cf3" and cur_col_name[:2] == "x_":
                                continue  # skip features without certain meaning
                            if col_idx in num_idx_set:
                                instruction += f"{cur_col_name}: {x};\n"
                            elif col_idx in cat_idx_set:
                                x = int(x)
                                cat_idx = col_idx_2_cat_idx[col_idx]
                                cat_string = cat_str[cat_idx][x]
                                instruction += f"{cur_col_name}: {cat_string};\n"
                            else:
                                continue  # should NOT enter here
                        instruction = instruction.replace("_", " ")
                        ins_json = json.dumps(instruction.strip())
                        fp_out.write(ins_json + "\n")

        except Exception as e:
            logger.info(f"Exception: {e}")
            continue  # should NOT enter here

    sys.exit(0)
