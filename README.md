# DeepChem Easy Model Loading

A HuggingFace-style interface for loading and sharing pretrained models in DeepChem, with special support for Morgan fingerprints.

## Overview

This module simplifies the process of saving, sharing, and loading DeepChem models by providing a standardized metadata format and easy-to-use functions. It's designed to work seamlessly with models that use Morgan fingerprints, which are common in cheminformatics.

Key features:

- Save models with comprehensive metadata
- Load pretrained models without knowledge of original training parameters
- Special support for Morgan/Circular fingerprints
- Model hub functionality for discovering and sharing models
- Transfer learning capabilities

## Installation

```bash
pip install deepchem  # Required dependency
# Then clone or download this repository
```

## Basic Usage

### Saving a Model

```python
import deepchem as dc
from deepchem.feat import CircularFingerprint
from deepchem.models import MultitaskClassifier
from deepchem_easy_loading import save_model

# Create a featurizer
featurizer = CircularFingerprint(size=2048, radius=2)

# Create a model
model = MultitaskClassifier(
    n_tasks=2,
    n_features=2048,
    layer_sizes=[1024, 512],
    dropouts=0.1
)

# Save the model with metadata
save_model(
    model=model,
    save_dir="./my_model",
    featurizer=featurizer,
    metadata={"task_type": "toxicity_prediction", "author": "Your Name"}
)
```

### Loading a Pretrained Model

```python
from deepchem_easy_loading import load_pretrained

# Load a model and its featurizer
model, featurizer = load_pretrained("./my_model")

# Use the model and featurizer
smiles = ["CC(=O)OC1=CC=CC=C1C(=O)O", "CCO"]
mol_features = featurizer.featurize(smiles)
predictions = model.predict(mol_features)
```

### Using the Model Hub

```python
from deepchem_easy_loading import EasyModelHub

# Initialize the hub
hub = EasyModelHub()

# List local models
available_models = hub.list_local_models()
print(available_models)

# Download a model (if there's a web service with models)
hub.download_model("toxicity_model", "https://example.com/models/toxicity_model.zip")

# Load a model from the hub
model, featurizer = hub.load_model("toxicity_model")
```

## Advanced Features

### Creating Morgan Fingerprints

```python
from deepchem_easy_loading import create_morgan_featurizer

# Create a Morgan fingerprint featurizer with custom parameters
morgan_featurizer = create_morgan_featurizer(
    radius=3,
    size=1024,
    use_chirality=True,
    use_features=True
)
```

### Transfer Learning

```python
from deepchem_easy_loading import create_model_from_pretrained

# Create a new model based on a pretrained one, with different parameters
new_model = create_model_from_pretrained(
    "./pretrained_model",
    n_tasks=3,  # Different number of tasks
    dropouts=0.2  # Different dropout rate
)
```

### Inspecting Model Metadata

```python
from deepchem_easy_loading import get_model_metadata

# Get metadata without loading the model
metadata = get_model_metadata("./my_model")
print(metadata["model_class"])
print(metadata["user_metadata"])
```

## Supported Model Types

The module supports all DeepChem model types, including:

- MultitaskClassifier
- MultitaskRegressor
- GraphConvModel
- ScScoreModel
- AttentiveFPModel
- Keras and PyTorch-based models

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

