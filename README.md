# FinPT: Financial Risk Prediction with Profile Tuning on Pretrained Foundation Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2308.00065-b31b1b.svg)](https://arxiv.org/abs/2308.00065)

![picture](https://yuweiyin.com/files/img/2023-07-22-FinPT.png)

* **Abstract**:

```text
Financial risk prediction plays a crucial role in the financial sector. 
Machine learning methods have been widely applied for automatically 
detecting potential risks and thus saving the cost of labor.
However, the development in this field is lagging behind in recent years 
by the following two facts: 1) the algorithms used are somewhat outdated, 
especially in the context of the fast advance of generative AI and 
large language models (LLMs); 2) the lack of a unified and open-sourced 
financial benchmark has impeded the related research for years.
To tackle these issues, we propose FinPT and FinBench: the former is a 
novel approach for financial risk prediction that conduct Profile Tuning 
on large pretrained foundation models, and the latter is a set of 
high-quality datasets on financial risks such as default, fraud, and churn.
In FinPT, we fill the financial tabular data into the pre-defined instruction 
template, obtain natural-language customer profiles by prompting LLMs, and 
fine-tune large foundation models with the profile text to make predictions.
We demonstrate the effectiveness of the proposed FinPT by experimenting with 
a range of representative strong baselines on FinBench. The analytical studies 
further deepen the understanding of LLMs for financial risk prediction.
```

## Environment

```bash
conda create -n finpt python=3.9
conda activate finpt
pip install -r requirements.txt
```

## Data

- **FinBench** on Hugging Face Datasets: https://huggingface.co/datasets/yuweiyin/FinBench

```python
from datasets import load_dataset

# ds_name_list = ["cd1", "cd2", "ld1", "ld2", "ld3", "cf1", "cf2", "cc1", "cc2", "cc3"]
ds_name = "cd1"  # change the dataset name here
dataset = load_dataset("yuweiyin/FinBench", ds_name)
```

## Experiments

The instructions obtained in Step 1 and customer profiles generated in Step 2
are provided as `X_instruction_for_profile` and `X_profile` in FinBench.

### Run Tree-based Baselines

```bash
SAVE_DIR="./log/baseline_tree/"
mkdir -p "${SAVE_DIR}"

DATASETS=("cd1" "cd2" "cd3" "ld1" "ld2" "cf1" "cc1" "cc2" "cc3")
MODELS=("RandomForestClassifier" "XGBClassifier" "CatBoostClassifier" "LGBMClassifier")
SEEDS=(0 1 42 1234)

for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      echo -e "\n\n\n>>> run_step3_baseline_tree.py: dataset: ${dataset}; model: ${model} seed: ${seed}"
      python run_step3_baseline_tree.py --ds_name "${dataset}" --model_name ${model} --seed ${cur_seed} --grid_search \
        > "${SAVE_DIR}/${dataset}-${model}-${seed}.log"
    done
  done
done
```

### Run Neural Network Baselines

```bash
SAVE_DIR="./log/baseline_nn/"
mkdir -p "${SAVE_DIR}"

DATASETS=("cd1" "cd2" "cd3" "ld1" "ld2" "cf1" "cc1" "cc2" "cc3")
MODELS=("DeepFM" "STG" "VIME" "TabNet")
SEEDS=(0 1 42 1234)

for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      echo -e "\n\n\n>>> run_step3_baseline_nn.py: dataset: ${dataset}; model: ${model} seed: ${seed}"
      python run_step3_baseline_nn.py --cuda "0" --ds_name "${dataset}" --model_name ${model} --seed ${cur_seed} \
        > "${SAVE_DIR}/${dataset}-${model}-${seed}.log"
    done
  done
done
```

### Run FinPT

```bash
SAVE_DIR="./log/finpt/"
mkdir -p "${SAVE_DIR}"

DATASETS=("cd1" "cd2" "cd3" "ld1" "ld2" "cf1" "cc1" "cc2" "cc3")
MODELS=("bert" "finbert" "gpt2" "t5-base" "flan-t5-base" "t5-xxl" "flan-t5-xxl" "llama-7b" "llama-13b")

for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
      echo -e "\n\n\n>>> run_step3_finpt.py: dataset: ${dataset}; model: ${model} seed: ${seed}"
      python run_step3_finpt.py --cuda "0,1" --ds_name "${dataset}" --model_name ${model} --use_pos_weight \
        > "${SAVE_DIR}/${dataset}-${model}-${seed}.log"
  done
done
```


## License

Please refer to the [LICENSE](./LICENSE) file for more details.


## Citation

```bibtex
@article{yin2023finpt,
  title   = {FinPT: Financial Risk Prediction with Profile Tuning on Pretrained Foundation Models},
  author  = {Yin, Yuwei and Yang, Yazheng and Yang, Jian and Liu, Qi},
  journal = {arXiv preprint arXiv:2308.00065},
  year    = {2023},
  url     = {https://arxiv.org/abs/2308.00065},
}
```
