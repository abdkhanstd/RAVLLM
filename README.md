# RAVLLM

Codes for our paper *Driving Like Humans: Leveraging Vision Large Language Models for Road Anomaly Detection*.

## Overview

This repository contains the codebase for road anomaly detection using Vision Large Language Models (VLLMs). It includes dataset loaders, model training, and evaluation scripts, targeting the detection of road anomalies in a human-like driving context.

## Project Structure

- **DataSetConverters/**: Tools for dataset conversion (to Florence-2 and PalliGemma format).
- **dataloaders/**: Dataset loading scripts.
- **samples/**: Sample inputs and outputs.
- **scripts/**: Helping scripts.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/abdkhanstd/RAVLLM.git
   cd RAVLLM
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start training:
   ```bash
   python TrainFlorence2OD.OK.py
   ```

## Data and Weights

The model weights and related datasets can be accessed from:

- **[Weights](#)**
- **[Datasets](#)**

## Citation

Please cite this work if it aids your research:

```
@inproceedings{Shafiq2024,
  title={Driving Like Humans: Leveraging Vision Large Language Models for Road Anomaly Detection},
  author={Sidra Shafiq and Hassan Moatasam Awan and Abdullah Aman Khan and Waqas Amin},
  year={2024},
  booktitle={2024 3rd International Conference on Emerging Trends in Electrical, Control, and Telecommunication Engineering (ETECTE)}
}
```

## License

This project is licensed under the MIT License.
```

You can adjust the links for weights and datasets accordingly.
