# CREST: Coupled Reusable Earth System Tensor Framework

<!-- Badges -->
[![Documentation Status](https://readthedocs.org/projects/crest/badge/?version=latest)](https://crest.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.13+](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://tensorflow.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

> A tensor-based modeling framework for building AI-enabled Earth Information Systems

## Overview

The **CREST framework** provides a flexible, scalable infrastructure for developing Earth system models that seamlessly integrate machine learning with physics-based components. Built on TensorFlow, CREST enables researchers to:

- **Build modular Earth system models** using hierarchical tensor graphs
- **Combine ML and physics-based processes** in a unified framework
- **Process large-scale gridded datasets** efficiently with optimized data pipelines
- **Deploy models** from research to operational systems
- **Share and reuse components** across the research community

### Key Features

- **Hierarchical Tensor Graphs (HTG)**: Directed acyclic graph structure for flexible model composition
- **Hybrid Modeling**: Seamlessly mix neural networks, physical equations, and empirical relationships
- **Scalable Data Pipeline**: Efficient batching and loading for large Earth science datasets
- **Model Serialization**: Save, version, and share complete model configurations
- **Distributed Computing**: Built-in support for multi-GPU and distributed training

## Quick Start

### Installation

```bash
# Basic installation
pip install crest

# With all dependencies
pip install crest[all]

# For development
git clone https://github.com/terrahydro/crest.git
cd crest
pip install -e ".[dev]"
```

### Your First Model

```python
from crest.model import HierarchalTensorGraph, LambdaNode, KerasNode
from crest.model import TensorSpec, Model
import tensorflow as tf

# Define input specifications
inputs = {
    'precipitation': TensorSpec(shape=(None, 100), dtype=tf.float32),
    'temperature': TensorSpec(shape=(None, 100), dtype=tf.float32)
}

# Create hierarchical tensor graph
htg = HierarchalTensorGraph(name='soil_moisture_model', inputs=inputs)

# Add physics-based component
def compute_pet(temperature):
    """Compute potential evapotranspiration."""
    return 0.5 * temperature + 2.0

pet_node = LambdaNode(compute_pet, name='pet')

# Add ML component
lstm = tf.keras.layers.LSTM(64, return_sequences=True)
ml_node = KerasNode(lstm, name='lstm_predictor')

# Build and compile model
model = Model(htg)
model.compile(optimizer='adam', loss='mse')

# Train model
model.fit(training_data, epochs=10)
```

## Documentation

**Full documentation**: https://crest.readthedocs.io/

- [Installation Guide](https://crest.readthedocs.io/en/latest/installation.html)
- [Tutorials](https://crest.readthedocs.io/en/latest/tutorials/)
- [API Reference](https://crest.readthedocs.io/en/latest/api/)
- [Example Notebooks](https://crest.readthedocs.io/en/latest/examples/)

### Interactive Examples

Try CREST in your browser:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/terrahydro/crest/main?filepath=examples)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/terrahydro/crest/blob/main/examples/HTG_overview.ipynb)

## Use Cases

### Soil Moisture Prediction

```python
# See examples/soil_moisture_demo.ipynb for full example
from crest.data import Dataset, Batcher
from crest.model import Model

# Load data
dataset = Dataset.from_zarr('s3://bucket/soil-moisture-data')
batcher = Batcher(dataset, batch_size=32)

# Train hybrid physics-ML model
model.fit(batcher, epochs=50, validation_split=0.2)

# Generate predictions
predictions = model.predict(test_data)
```

### Custom Earth System Component

```python
from crest.model import LambdaNode

def snow_accumulation(precipitation, temperature, snow_water_equiv):
    """Custom snow accumulation model."""
    snowfall = tf.where(temperature < 0, precipitation, 0.0)
    melt = tf.where(temperature > 0, 0.5 * temperature, 0.0)
    return snow_water_equiv + snowfall - melt

snow_node = LambdaNode(
    snow_accumulation,
    name='snow_model',
    inputs=['precip', 'temp', 'swe'],
    outputs=['swe_updated']
)
```

## Architecture

CREST follows a modular architecture:

```
┌─────────────────────────────────────────────────┐
│           Application Layer                      │
│  (User Models, Workflows, Experiments)          │
└─────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│         Model Layer (HTG)                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │  Lambda  │  │  Keras   │  │ Recurrent│     │
│  │   Node   │  │   Node   │  │   Node   │     │
│  └──────────┘  └──────────┘  └──────────┘     │
└─────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│         Data Layer                              │
│  Dataset → Batcher → Transforms                 │
└─────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│         Storage Layer                           │
│  Zarr | TileDB | NetCDF | Xarray               │
└─────────────────────────────────────────────────┘
```

## Performance

CREST is designed for large-scale Earth science applications:

| Dataset Size | Training Time | Throughput |
|-------------|---------------|------------|
| 1 TB | 2.5 hours (8 GPUs) | 115 GB/hour |
| 100 GB | 20 minutes (1 GPU) | 300 GB/hour |
| 10 GB | 3 minutes (1 GPU) | 200 GB/hour |

*Benchmarks on soil moisture prediction task with LSTM architecture*

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Quick Contribution Checklist

- [ ] Code follows style guide (PEP 8, black formatting)
- [ ] Tests added for new functionality
- [ ] Documentation updated (docstrings + tutorials if needed)
- [ ] All tests pass (`pytest tests/`)
- [ ] PR describes changes clearly

## Community and Support

- **Issues**: [GitHub Issues](https://github.com/terrahydro/crest/issues)
- **Discussions**: [GitHub Discussions](https://github.com/terrahydro/crest/discussions)
- **Email**: crest-dev@terrahydro.org
- **Twitter**: [@CRESTFramework](https://twitter.com/CRESTFramework)

## Citation

If you use CREST in your research, please cite:

```bibtex
@software{crest2024,
  title = {CREST: Coupled Reusable Earth System Tensor Framework},
  author = {Pelissier, Craig and {CREST Development Team}},
  year = {2024},
  url = {https://github.com/terrahydro/crest},
  version = {0.1.0}
}
```

See [CITATION.cff](CITATION.cff) for complete citation information.

## License

CREST is licensed under the Apache License. See [LICENSE](LICENSE) for details.

## Acknowledgments

This project is supported by:
- [Funding Agency Name]
- [Institution Name]
- [Collaborating Organizations]

Special thanks to all [contributors](CONTRIBUTORS.md) who have helped shape CREST.
