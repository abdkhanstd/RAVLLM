# RAVLLM: Driving Like Humans - Vision Large Language Models for Road Anomaly Detection

[![Paper](https://img.shields.io/badge/Paper-IEEE%20ETECTE%202024-blue.svg)](https://ieeexplore.ieee.org/document/)
[![Framework](https://img.shields.io/badge/Framework-Vision%20LLM-green.svg)](https://github.com/abdkhanstd/RAVLLM)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-red.svg)](https://python.org)

## Overview

This repository presents the implementation of RAVLLM, a novel framework that leverages Vision Large Language Models (VLLMs) for road anomaly detection in autonomous driving contexts. Our approach demonstrates how modern vision-language models can be effectively adapted to identify and reason about road anomalies through human-like perception and reasoning patterns.

The framework addresses the critical challenge of detecting road anomalies that may not conform to predefined categories, utilizing the emergent reasoning capabilities of VLLMs to achieve robust performance across diverse driving scenarios. Through careful prompt engineering and model adaptation, RAVLLM provides both accurate anomaly detection and interpretable reasoning processes that mirror human decision-making patterns.

## Repository Structure

```
RAVLLM/
├── DataSetConverters/          # Dataset format conversion utilities
│   ├── florence2_converter.py  # Florence-2 format conversion
│   └── palligemma_converter.py # PaliGemma format conversion
├── dataloaders/               # Dataset loading and preprocessing
│   ├── anomaly_loader.py      # Road anomaly dataset loader
│   └── preprocessing.py       # Data preprocessing utilities
├── samples/                   # Example inputs and outputs
│   ├── input_examples/        # Sample road images
│   └── output_examples/       # Model prediction samples
├── scripts/                   # Utility and helper scripts
│   ├── evaluation.py          # Model evaluation utilities
│   └── visualization.py       # Result visualization tools
├── TrainFlorence2OD.OK.py     # Main training script for Florence-2
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## Installation and Setup

### Prerequisites

Ensure you have Python 3.8 or higher installed on your system.

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/abdkhanstd/RAVLLM.git
   cd RAVLLM
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python -c "import torch; print('PyTorch version:', torch.__version__)"
   ```

## Usage Guide

### Training the Model

To begin training the RAVLLM framework:

```bash
python TrainFlorence2OD.OK.py
```

The training script supports various configuration options that can be modified within the script or passed as command-line arguments.

### Dataset Preparation

The framework supports multiple dataset formats through the provided conversion utilities:

**For Florence-2 format**:
```bash
python DataSetConverters/florence2_converter.py --input_dir /path/to/raw/data --output_dir /path/to/converted/data
```

**For PaliGemma format**:
```bash
python DataSetConverters/palligemma_converter.py --input_dir /path/to/raw/data --output_dir /path/to/converted/data
```

### Evaluation and Testing

Evaluate trained models using the provided evaluation scripts:

```bash
python scripts/evaluation.py --model_path /path/to/trained/model --test_data /path/to/test/data
```

## Data and Model Weights

The model weights and related datasets can be accessed from:

- **[Weights](#)**: Pre-trained RAVLLM models for road anomaly detection
- **[Datasets](#)**: Curated road anomaly datasets with annotations

### Dataset Format

The framework expects input datasets to follow the standardized format for road anomaly detection tasks. Sample data structures and formatting guidelines are provided in the `samples/` directory.

## Technical Implementation

### Supported Vision Language Models

The current implementation provides support for:
- **Florence-2**: Microsoft's vision-language model optimized for object detection tasks
- **PaliGemma**: Google's efficient vision-language model for multimodal understanding

### Model Architecture

The RAVLLM framework implements a multi-modal approach that combines visual feature extraction with language-based reasoning. The architecture leverages pre-trained vision-language models while incorporating domain-specific adaptations for road anomaly detection scenarios.

## Experimental Results

Our experimental validation demonstrates that RAVLLM achieves competitive performance on standard road anomaly detection benchmarks while providing interpretable reasoning outputs that facilitate understanding of model decisions. Detailed results and analysis will be included in the full paper publication.

## Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@inproceedings{shafiq2024driving,
  title={Driving Like Humans: Leveraging Vision Large Language Models for Road Anomaly Detection},
  author={Shafiq, Sidra and Awan, Hassan Moatasam and Khan, Abdullah Aman and Amin, Waqas},
  booktitle={2024 3rd International Conference on Emerging Trends in Electrical, Control, and Telecommunication Engineering (ETECTE)},
  pages={1--6},
  year={2024},
  organization={IEEE}
}
```

## Contributing

We welcome contributions to improve RAVLLM. Please feel free to submit issues and enhancement requests through the GitHub repository.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions and collaborations, please reach out to the corresponding authors through the institutional channels or create an issue in this repository.

---

**Note**: This repository represents ongoing research in vision-language models for autonomous driving applications. We encourage researchers to build upon this work and contribute to advancing the field of interpretable road anomaly detection.
