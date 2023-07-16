#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__author__ = "@YuweiYin"
"""

import os
import sys
import time
import json
import logging
import argparse
import openai


def run_openai(ds_name: str, ds_split: str, start_idx: int = 0, end_idx: int = -1) -> int:
    # print_cnt = int(1e3)
    print_cnt = 100
    # for ds_name in ds_name_list:
    profile_dir = os.path.join(profile_root_dir, ds_name)
    os.makedirs(profile_dir, exist_ok=True)

    logger.info(f"\n\n>>> ds_name: {ds_name}; ds_split: {ds_split}")
    instruction_path = os.path.join(profile_dir, f"instruction_for_profile_X_{ds_split}.jsonl")
    if end_idx > 0:
        profile_path = os.path.join(profile_dir, f"profile_X_{ds_split}_{end_idx}.jsonl")
    else:
        profile_path = os.path.join(profile_dir, f"profile_X_{ds_split}_all.jsonl")
    logger.info(f">>> profile_path: {profile_path}")

    read_cnt = 0
    write_cnt = 0
    # write_cnt = start_idx
    while True:
        try:
            logger.info(f"\n\n>>> >>> start_idx: {start_idx}")
            read_cnt = 0
            with open(instruction_path, mode="r", encoding="utf-8") as fp_in:
                with open(profile_path, mode="a+", encoding="utf-8") as fp_out:
                    for line_idx, line in enumerate(fp_in):
                        read_cnt += 1
                        if line_idx < start_idx:
                            continue
                        if line_idx >= end_idx > 0:
                            logger.info(f">>> >>> [{ds_name} - {ds_split}] line_idx >= end_idx > 0; "
                                        f"read_cnt = {read_cnt}; write_cnt = {write_cnt}")
                            break

                        instruction = str(json.loads(line.strip()))

                        # OpenAI request
                        response = openai.ChatCompletion.create(
                            model=openai_model,
                            messages=[
                                {"role": "system", "content":
                                    "You are a helpful financial assistant."},
                                {"role": "user", "content": f"{instruction}"},
                            ],
                            temperature=0,
                        )

                        res_content = response["choices"][0]["message"]["content"]
                        res_json = json.dumps(res_content.strip())
                        fp_out.write(res_json + "\n")
                        write_cnt += 1
                        if read_cnt % print_cnt == 0:
                            logger.info(f">>> >>> [{ds_name} - {ds_split}] "
                                        f"read_cnt = {read_cnt}; write_cnt = {write_cnt}")
                        time.sleep(0.2)
            break

        except Exception as e:
            start_idx = read_cnt - 1
            # logger.info(f">>> *** >>> Exception: {e}")
            logger.info(f">>> *** >>> [{ds_name} - {ds_split}] "
                        f"read_cnt = {read_cnt}; write_cnt = {write_cnt} Next start_idx: {start_idx}\n")
            # time.sleep(1.0)

    logger.info(f"\n>>> DONE: [{ds_name} - {ds_split}] read_cnt = {read_cnt}; write_cnt = {write_cnt}\n\n")
    return 0


if __name__ == "__main__":
    """
    python3 run_step2_gpt_profile.py --ds_name cd1 --ds_split train --start_idx 0 --end_idx -1
    python3 run_step2_gpt_profile.py --ds_name cd1 --ds_split validation --start_idx 0 --end_idx -1
    python3 run_step2_gpt_profile.py --ds_name cd1 --ds_split test --start_idx 0 --end_idx -1
    """

    logging.basicConfig(
        format="[%(asctime)s - %(levelname)s - %(name)s] -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Step2 Get_Profile Args")
    parser.add_argument("--ds_name", type=str, default="cd1", help="Specify which dataset to use")
    parser.add_argument("--ds_split", type=str, default="train", help="train OR validation OR test")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index for continue generating")
    parser.add_argument("--end_idx", type=int, default=-1, help="Ending index for continue generating")
    args = parser.parse_args()

    logger.info(args)

    ds_name = str(args.ds_name)
    ds_split = str(args.ds_split)
    start_idx = int(args.start_idx)
    end_idx = int(args.end_idx)

    cache_dir = "~/.cache/huggingface/"
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    cache_model = os.path.join(cache_dir, "models")
    cache_ds = os.path.join(cache_dir, "datasets")

    # OpenAI settings
    openai.organization = "YOUR_ORG_ID"
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai_model = "gpt-3.5-turbo"

    profile_root_dir = os.path.join("./data/profile")
    os.makedirs(profile_root_dir, exist_ok=True)

    run_openai(ds_name=ds_name, ds_split=ds_split, start_idx=start_idx, end_idx=end_idx)

    sys.exit(0)
