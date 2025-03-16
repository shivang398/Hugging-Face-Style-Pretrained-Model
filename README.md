# DeepChem Easy Model Loading

This repository provides utilities for easy saving and loading of DeepChem models, similar to the Hugging Face Transformers library. It simplifies the process of working with pretrained models, especially those using Morgan fingerprints.

## Features

-   **Easy Model Saving:** Save DeepChem models with comprehensive metadata, including initialization parameters, featurizer information, and custom objects.
-   **Easy Model Loading:** Load pretrained DeepChem models using only the model directory path.
-   **Morgan Fingerprint Support:** Special handling for models using Morgan (Circular) fingerprints, including saving and loading featurizer parameters.
-   **Model Hub Interface:** A simple `EasyModelHub` class for managing local pretrained models, including listing, downloading, and loading.
-   **Transfer Learning Support:** Easily create new model instances from pretrained models for transfer learning or fine-tuning.
-   **Metadata Handling:** Functions to extract, process, and retrieve model metadata.
-   **Automatic Model and Featurizer Registration:** Automatically registers all available DeepChem models and featurizers.

## Installation

```bash
pip install deepchem  # Ensure DeepChem is installed
# Clone this repository or copy the deepchem_easy_model_loading.py file into your project.

## Usage
import deepchem as dc
from deepchem_easy_model_loading import save_model, create_morgan_featurizer

# Create a sample model and featurizer
tasks, datasets, transformers = dc.molnet.load_tox21(featurizer='Raw')
train_dataset, valid_dataset, test_dataset = datasets
featurizer = create_morgan_featurizer()
model = dc.models.GraphConvModel(n_tasks=len(tasks), mode='classification')

# Train the model
model.fit(train_dataset, nb_epoch=1)

# Save the model
save_dir = "my_saved_model"
save_model(model, save_dir, featurizer=featurizer)

print(f"Model saved to: {save_dir}")
