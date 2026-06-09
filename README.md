<div align="center">

# LiteSpecFormer <img width="15%" align="right" src="https://github.com/wwhenxuan/S2Generator/blob/main/docs/source/_static/S2Generator_logo.png?raw=true">

<!-- [![preprint](https://img.shields.io/static/v1?label=Chronos-2-Report&message=2510.15821&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/2510.15821) -->
[![PyPI version](https://badge.fury.io/py/litespecformer.svg)](https://pypi.org/project/litespecformer/)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20HF-Datasets-FFD21E)](https://huggingface.co/datasets/FlowVortex/Large-Spectrum-Prediction-Dataset)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20HF-Models-FFD21E)](https://huggingface.co/FlowVortex/LiteSpecFormer-1.0-36M)
[![License: MIT](https://img.shields.io/badge/License-Apache--2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

## ✨Overview

[LiteSpecFormer](https://huggingface.co/FlowVortex/LiteSpecFormer-1.0-36M) is the first lightweight wireless foundation model for zero-shot confidence spectrum prediction. Built on a channel-independent Transformer with a sliding autoregressive paradigm and a novel linear correlation loss to mitigate error accumulation, it achieves state-of-the-art performance across arbitrary frequency bands and sequence lengths without downstream fine-tuning.

<div style="text-align: center;">
    <img src="https://raw.githubusercontent.com/FlowVortex/LiteSpecFormer/main/notebooks/figures/architecture.png" alt="model" style="zoom:60%;" />
</div>

We present [Large-Spectrum-Prediction-Dataset (LSPD)](https://huggingface.co/datasets/FlowVortex/Large-Spectrum-Prediction-Dataset), the first large-scale dataset specifically designed for pre-training spectrum prediction foundation models. It comprises 18 billion timestamps and integrates two learnable data generation mechanisms to generate high-quality, diverse spectrum samples.

<div style="text-align: center;">
    <img src="https://raw.githubusercontent.com/FlowVortex/LiteSpecFormer/main/notebooks/figures/dataset_info.jpg" alt="dataset" style="zoom:90%;" />
</div>

## 🧭 Quickstart

### Installation

First create a Python virtual environment with 3.11+, then install the required dependencies through [PyPI](https://pypi.org/project/litespecformer/):

```
pip install litespecformer
```

Then, you can refer to our [demo](notebooks/demo.ipynb) file to create our pipeline and then use the `prediction` method for zero-shot prediction:

```python
from s2generator.utils import generate_nonstationary_sine
from litespecformer import LiteSpecFormerPipeline


# Load our model using our pipeline.
pipeline = LiteSpecFormerPipeline.from_pretrained("FlowVortex/LiteSpecFormer")

# Set the context length and prediction length
context_length = 256
# Generate a non-stationary sine wave time series with the specified lengths
time_series = np.vstack(
    [
        generate_nonstationary_sine(
            seq_length=context_length + prediction_length, freq=2 + i
        )
        for i in range(5)
    ]
)

# Prediction the inputs through our pipeline
outputs = pipeline.predict(
    time_series[:, :context_length], prediction_length=prediction_length
)
```

### Data Preparation

Download [Large-Spectrum-Prediction-Dataset (LSPD)](https://huggingface.co/datasets/FlowVortex/Large-Spectrum-Prediction-Dataset) for pre-training or downstream evaluation:

```bash
hf download FlowVortex/Large-Spectrum-Prediction-Dataset \
  --local-dir ./data \
  --type dataset
```

To fetch only the test split (useful for quick benchmarking):

```bash
hf download FlowVortex/Large-Spectrum-Prediction-Dataset \
  --local-dir ./data \
  --type dataset \
  --include "test/*"
```

The simulation and augmentation pipelines used to build LSPD are available in [S2Generator](https://github.com/wwhenxuan/S2Generator): see the [simulator](https://github.com/wwhenxuan/S2Generator/tree/main/s2generator/simulator) and [augmentation](https://github.com/wwhenxuan/S2Generator/tree/main/s2generator/augmentation) modules.

### Model Pre-Train and Fine-Tune

We provide scripts for pre-training, fine-tuning, and zero-shot out-of-distribution evaluation on downstream tasks:

```bash
# Pre-train LiteSpecFormer
bash scripts/pre-training.sh

# Fine-tune LiteSpecFormer
bash scripts/fine_tuning.sh
```

### Zero-Shot Evaluation

After downloading the test split, run the evaluation scripts for zero-shot out-of-distribution prediction. At inference time, LiteSpecFormer performs channel-independent forecasting:

```bash
# Example: Madrid dataset
bash scripts/zero_shot_prediction/Madrid.sh
```

Other benchmark datasets are available under `scripts/zero_shot_prediction/` (e.g., `Alcorcon1`, `IADAM`, `Nudelsalat`, `Oreland`, `PiSDR1`).

## 📊 Results

### Benchmark Results

After large-scale pre-training, LiteSpecFormer achieves stronger out-of-distribution generalization and scalability than existing supervised baselines:

<div style="text-align: center;">
    <img src="https://raw.githubusercontent.com/FlowVortex/LiteSpecFormer/main/notebooks/figures/benchmark.jpg" alt="benchmark" style="zoom:60%;" />
</div>

### Forecasting Visualization

Predictions on the Madrid dataset. Compared with supervised in-distribution models and other out-of-distribution zero-shot baselines, LiteSpecFormer produces more accurate forecasts, especially in fine-grained spectral detail:

<div style="text-align: center;">
    <img src="https://raw.githubusercontent.com/FlowVortex/LiteSpecFormer/main/notebooks/figures/visualization.jpg" alt="visualization" style="zoom:60%;" />
</div>

<!-- ## 🎓 Citation <a id="Citation"></a>

If you find this code useful, please cite our paper.

```
@article{...}
```
-->
## 🎖️ Acknowledgement

We appreciate the following GitHub repos a lot for their valuable code and efforts.

- chronos-forecasting (https://github.com/amazon-science/chronos-forecasting);
- PySDKit (https://github.com/wwhenxuan/PySDKit);
- S2Generator (https://github.com/wwhenxuan/S2Generator);
- Gift-Eval (https://huggingface.co/spaces/Salesforce/GIFT-Eval);


## 🤗 Contact

If you have any questions or are interested in our view on the complex dynamics of time series, feel free to contact:

- [Whenxuan Wang](https://wwhenxuan.github.io/) (whenxuanwang@stu.xidian.edu.cn)
- [Dan Wang](https://web.xidian.edu.cn/danwang/) (danwang@xidian.edu.cn)