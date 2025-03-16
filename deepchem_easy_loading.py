"""
DeepChem Easy Model Loading with Morgan Fingerprint Support

This module implements HuggingFace-style easy loading for pretrained models in DeepChem.
It provides a standardized metadata format and functions to save/load models without
requiring knowledge of the original training parameters, with special support for
models using Morgan fingerprints.
"""

import os
import json
import yaml
import tempfile
import warnings
import inspect
import importlib
from typing import Dict, Any, Optional, Union, List, Type, Tuple
from pathlib import Path

import numpy as np
import deepchem as dc
from deepchem.models.models import Model
from deepchem.feat import MolecularFeaturizer
from deepchem.feat.molecule_featurizers import CircularFingerprint

# Dictionary mapping model class names to their actual classes
MODEL_REGISTRY = {}

# Dictionary mapping featurizer class names to their actual classes
FEATURIZER_REGISTRY = {}

def register_models_and_featurizers():
    """
    Automatically register all available model and featurizer classes in DeepChem.
    This populates the MODEL_REGISTRY and FEATURIZER_REGISTRY dictionaries.
    """
    # Register models
    for module_name in dir(dc.models):
        if module_name.startswith('_'):
            continue

        try:
            module = getattr(dc.models, module_name)
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and
                    issubclass(attr, Model) and
                    attr != Model):
                    MODEL_REGISTRY[attr.__name__] = attr
        except (ImportError, AttributeError):
            continue

    # Register featurizers
    for module_name in dir(dc.feat):
        if module_name.startswith('_'):
            continue

        try:
            module = getattr(dc.feat, module_name)
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and
                    issubclass(attr, MolecularFeaturizer) and
                    attr != MolecularFeaturizer):
                    FEATURIZER_REGISTRY[attr.__name__] = attr
        except (ImportError, AttributeError):
            continue


def save_model(
    model: Model,
    save_dir: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
    featurizer: Optional[MolecularFeaturizer] = None,
    save_format: str = "default"
) -> str:
    """
    Save a DeepChem model with metadata for easy reloading.

    Parameters
    ----------
    model: Model
        The DeepChem model to save
    save_dir: str or Path
        Directory where the model will be saved
    metadata: dict, optional
        Additional metadata to save with the model
    featurizer: MolecularFeaturizer, optional
        Featurizer used with this model, if any
    save_format: str, default 'default'
        Format to save the model ('default', 'keras', 'pytorch', or 'pickle')

    Returns
    -------
    str
        Path to the saved model directory
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Extract model initialization parameters
    init_params = extract_init_params(model)

    # Create comprehensive metadata
    full_metadata = {
        "model_class": model.__class__.__name__,
        "model_module": model.__class__.__module__,
        "dc_version": dc.__version__,
        "init_params": init_params,
        "model_specific": model.get_model_specific_metadata(),
        "custom_objects": get_custom_objects(model)
    }

    # Add featurizer metadata if provided
    if featurizer is not None:
        featurizer_params = extract_init_params(featurizer)
        full_metadata["featurizer"] = {
            "class": featurizer.__class__.__name__,
            "module": featurizer.__class__.__module__,
            "params": featurizer_params
        }

        # Special handling for Morgan/Circular fingerprints
        if isinstance(featurizer, CircularFingerprint):
            full_metadata["featurizer"]["type"] = "morgan"
            full_metadata["featurizer"]["radius"] = featurizer.radius
            full_metadata["featurizer"]["size"] = featurizer.size
            full_metadata["featurizer"]["chiral"] = getattr(featurizer, "use_chirality", False)
            full_metadata["featurizer"]["bonds"] = getattr(featurizer, "use_bond_types", True)
            full_metadata["featurizer"]["features"] = getattr(featurizer, "use_features", False)

    # Add user-provided metadata
    if metadata:
        full_metadata["user_metadata"] = metadata

    # Save the metadata
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(full_metadata, f, indent=2, default=json_serializer)

    # Save the model using the appropriate method
    if save_format == "default":
        # Let the model decide how to save itself
        model.save_to_path(str(save_dir / "model"))
    elif save_format == "keras":
        assert hasattr(model, "model"), "Model doesn't have a Keras model attribute"
        model.model.save(str(save_dir / "keras_model"))
    elif save_format == "pytorch":
        import torch
        assert hasattr(model, "model"), "Model doesn't have a PyTorch model attribute"
        torch.save(model.model.state_dict(), str(save_dir / "pytorch_model.pt"))
    elif save_format == "pickle":
        import pickle
        with open(save_dir / "pickled_model.pkl", "wb") as f:
            pickle.dump(model, f)
    else:
        raise ValueError(f"Unsupported save format: {save_format}")

    # Save the featurizer if provided
    if featurizer is not None:
        try:
            import pickle
            with open(save_dir / "featurizer.pkl", "wb") as f:
                pickle.dump(featurizer, f)
        except Exception as e:
            warnings.warn(f"Failed to save featurizer: {e}")

    return str(save_dir)


def extract_init_params(obj: Any) -> Dict[str, Any]:
    """
    Extract initialization parameters from an object instance.

    Parameters
    ----------
    obj: Any
        The object to extract parameters from

    Returns
    -------
    dict
        Dictionary of parameter names and values
    """
    # Get the signature of the object's __init__ method
    signature = inspect.signature(obj.__class__.__init__)
    parameters = signature.parameters

    # Exclude 'self' parameter
    param_names = [name for name in parameters.keys() if name != 'self']

    # Get object attributes that match parameter names
    init_params = {}
    for name in param_names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            # Handle special cases for non-serializable types
            init_params[name] = value

    return init_params


def get_custom_objects(model: Model) -> Dict[str, str]:
    """
    Extract information about any custom objects used by the model.

    Parameters
    ----------
    model: Model
        The model to extract custom objects from

    Returns
    -------
    dict
        Dictionary mapping custom object names to their import paths
    """
    custom_objects = {}

    # Get attributes that might be custom classes
    for attr_name in dir(model):
        if attr_name.startswith('_'):
            continue

        try:
            attr = getattr(model, attr_name)
            # Check if it's a class instance that's not a built-in type
            if (attr is not None and
                hasattr(attr, '__class__') and
                not attr.__class__.__module__.startswith('builtins')):

                class_name = attr.__class__.__name__
                module_name = attr.__class__.__module__
                custom_objects[class_name] = f"{module_name}.{class_name}"
        except (AttributeError, TypeError):
            continue

    return custom_objects


def json_serializer(obj):
    """Custom JSON serializer for handling non-standard objects."""
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    else:
        return str(obj)


def load_pretrained(
    model_path: Union[str, Path],
    load_featurizer: bool = True
) -> Union[Model, Tuple[Model, MolecularFeaturizer]]:
    """
    Load a pretrained DeepChem model using only the path.

    Parameters
    ----------
    model_path: str or Path
        Path to the saved model directory
    load_featurizer: bool, default True
        Whether to load the featurizer if one was saved

    Returns
    -------
    Model or (Model, MolecularFeaturizer)
        The loaded DeepChem model and optionally the featurizer
    """
    # Ensure registries are populated
    if not MODEL_REGISTRY:
        register_models_and_featurizers()

    model_path = Path(model_path)

    # Load metadata
    try:
        with open(model_path / "metadata.json", "r") as f:
            metadata = json.load(f)
    except FileNotFoundError:
        raise ValueError(f"No metadata file found in {model_path}")

    # Get model class
    model_class_name = metadata["model_class"]

    # Try to get the model class from registry
    if model_class_name in MODEL_REGISTRY:
        model_class = MODEL_REGISTRY[model_class_name]
    else:
        # Try to import the module and get the class
        try:
            module_path = metadata["model_module"]
            module = importlib.import_module(module_path)
            model_class = getattr(module, model_class_name)
        except (ImportError, AttributeError):
            raise ValueError(f"Could not find model class {model_class_name}")

    # Extract initialization parameters
    init_params = metadata["init_params"]

    # Handle special parameter types (convert strings back to enums, etc.)
    processed_params = process_init_params(init_params, model_class)

    # Instantiate the model
    model = model_class(**processed_params)

    # Load model-specific data
    if hasattr(model, "load_from_path"):
        model.load_from_path(str(model_path / "model"))
    elif (model_path / "keras_model").exists():
        try:
            from tensorflow import keras
            model.model = keras.models.load_model(str(model_path / "keras_model"))
        except ImportError:
            raise ImportError("TensorFlow is required to load this model.")
    elif (model_path / "pytorch_model.pt").exists():
        try:
            import torch
            model.model.load_state_dict(torch.load(str(model_path / "pytorch_model.pt")))
        except ImportError:
            raise ImportError("PyTorch is required to load this model.")
    elif (model_path / "pickled_model.pkl").exists():
        import pickle
        with open(model_path / "pickled_model.pkl", "rb") as f:
            loaded_model = pickle.load(f)
            # Transfer state to the new model instance
            transfer_state(loaded_model, model)
    else:
        warnings.warn(f"Could not find model weights in {model_path}. "
                     "Only the model structure has been loaded.")

    # Load featurizer if requested
    featurizer = None
    if load_featurizer:
        featurizer = load_featurizer_from_metadata(model_path, metadata)

    if featurizer is not None:
        return model, featurizer
    else:
        return model


def load_featurizer_from_metadata(
    model_path: Path,
    metadata: Dict[str, Any]
) -> Optional[MolecularFeaturizer]:
    """
    Load featurizer from metadata or saved file.

    Parameters
    ----------
    model_path: Path
        Path to the model directory
    metadata: dict
        Model metadata

    Returns
    -------
    MolecularFeaturizer or None
        The loaded featurizer if available
    """
    # First try to load directly from saved file
    featurizer_path = model_path / "featurizer.pkl"
    if featurizer_path.exists():
        try:
            import pickle
            with open(featurizer_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            warnings.warn(f"Failed to load saved featurizer: {e}")

    # If that fails, try to recreate from metadata
    if "featurizer" in metadata:
        featurizer_info = metadata["featurizer"]
        featurizer_class_name = featurizer_info["class"]

        # Special case for Morgan/Circular fingerprints
        if featurizer_info.get("type") == "morgan":
            # Create a CircularFingerprint
            if "CircularFingerprint" in FEATURIZER_REGISTRY:
                morgan_class = FEATURIZER_REGISTRY["CircularFingerprint"]
            else:
                # Try to import directly
                try:
                    from deepchem.feat.molecule_featurizers import CircularFingerprint
                    morgan_class = CircularFingerprint
                except ImportError:
                    return None

            # Create fingerprint featurizer with saved params
            return morgan_class(
                radius=featurizer_info.get("radius", 2),
                size=featurizer_info.get("size", 2048),
                use_chirality=featurizer_info.get("chiral", False),
                use_bond_types=featurizer_info.get("bonds", True),
                use_features=featurizer_info.get("features", False)
            )

        # For other featurizers, try to recreate from registry
        elif featurizer_class_name in FEATURIZER_REGISTRY:
            featurizer_class = FEATURIZER_REGISTRY[featurizer_class_name]
            featurizer_params = featurizer_info.get("params", {})
            try:
                return featurizer_class(**featurizer_params)
            except Exception as e:
                warnings.warn(f"Failed to recreate featurizer from metadata: {e}")

    return None


def process_init_params(params: Dict[str, Any], target_class: Type) -> Dict[str, Any]:
    """
    Process initialization parameters for proper model instantiation.

    This handles special cases like converting string representations back to enum values,
    recreating objects from their serialized form, and other special parameter types.

    Parameters
    ----------
    params: dict
        Dictionary of parameter names and their serialized values
    target_class: Type
        The class these parameters will be used to instantiate

    Returns
    -------
    dict
        Processed parameters suitable for instantiating the target class
    """
    processed = {}

    # Get signature of the target class's __init__ method
    signature = inspect.signature(target_class.__init__)
    param_specs = signature.parameters

    for name, value in params.items():
        # Skip parameters not in the signature
        if name not in param_specs:
            continue

        # Handle special types based on the parameter's annotation
        param_spec = param_specs[name]
        if param_spec.annotation != inspect.Parameter.empty:
            # Handle numpy arrays
            if 'numpy' in str(param_spec.annotation) and isinstance(value, list):
                processed[name] = np.array(value)
                continue

            # Handle enum types
            if hasattr(param_spec.annotation, '__members__') and isinstance(value, str):
                if value in param_spec.annotation.__members__:
                    processed[name] = param_spec.annotation[value]
                    continue

        # Default case - use the value as is
        processed[name] = value

    return processed


def transfer_state(source_model: Model, target_model: Model) -> None:
    """
    Transfer the state from a source model to a target model.

    This is used when loading a pickled model to transfer its state to
    a freshly instantiated model object.

    Parameters
    ----------
    source_model: Model
        Source model with the state to transfer
    target_model: Model
        Target model to receive the state
    """
    # Get all attributes of the source model
    for attr_name in dir(source_model):
        # Skip private attributes and methods
        if attr_name.startswith('_') or callable(getattr(source_model, attr_name)):
            continue

        # Transfer the attribute if it exists in both models
        if hasattr(target_model, attr_name):
            setattr(target_model, attr_name, getattr(source_model, attr_name))


def create_model_from_pretrained(
    model_dir: Union[str, Path],
    model_type: str = None,
    **model_kwargs
) -> Model:
    """
    Create a new model instance using a pretrained model as a starting point.

    This allows for transfer learning or fine-tuning by creating a new model
    with potentially different hyperparameters while keeping the trained weights.

    Parameters
    ----------
    model_dir: str or Path
        Path to the pretrained model directory
    model_type: str, optional
        If specified, create a model of this type instead of the original model type
    **model_kwargs
        Additional keyword arguments to override the original model parameters

    Returns
    -------
    Model
        New model instance with weights from the pretrained model
    """
    model_dir = Path(model_dir)

    # Load metadata
    with open(model_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    # Determine model class
    if model_type is not None:
        # Use specified model type
        if model_type in MODEL_REGISTRY:
            model_class = MODEL_REGISTRY[model_type]
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    else:
        # Use original model type
        model_class_name = metadata["model_class"]
        if model_class_name in MODEL_REGISTRY:
            model_class = MODEL_REGISTRY[model_class_name]
        else:
            module_path = metadata["model_module"]
            try:
                module = importlib.import_module(module_path)
                model_class = getattr(module, model_class_name)
            except (ImportError, AttributeError):
                raise ValueError(f"Could not find model class {model_class_name}")

    # Get original parameters and update with provided kwargs
    init_params = metadata["init_params"].copy()
    init_params.update(model_kwargs)

    # Process parameters
    processed_params = process_init_params(init_params, model_class)

    # Create new model
    model = model_class(**processed_params)

    # Load weights if available
    if hasattr(model, "load_from_path"):
        model.load_from_path(str(model_dir / "model"))
    elif (model_dir / "keras_model").exists():
        try:
            from tensorflow import keras
            model.model = keras.models.load_model(str(model_dir / "keras_model"))
        except ImportError:
            raise ImportError("TensorFlow is required to load this model.")
    elif (model_dir / "pytorch_model.pt").exists():
        try:
            import torch
            model.model.load_state_dict(torch.load(str(model_dir / "pytorch_model.pt")))
        except ImportError:
            raise ImportError("PyTorch is required to load this model.")

    return model


def get_model_metadata(model_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Retrieve the metadata for a saved model without loading the model.

    Parameters
    ----------
    model_path: str or Path
        Path to the saved model directory

    Returns
    -------
    dict
        Model metadata
    """
    model_path = Path(model_path)

    try:
        with open(model_path / "metadata.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise ValueError(f"No metadata file found in {model_path}")


def create_morgan_featurizer(
    radius: int = 2,
    size: int = 2048,
    use_chirality: bool = False,
    use_bond_types: bool = True,
    use_features: bool = False,
    **kwargs
) -> CircularFingerprint:
    """
    Helper function to create a Morgan fingerprint featurizer with common parameters.

    Parameters
    ----------
    radius: int, default 2
        Morgan fingerprint radius
    size: int, default 2048
        Length of generated bit/count vectors
    use_chirality: bool, default False
        Whether to consider chirality in fingerprints
    use_bond_types: bool, default True
        Whether to consider bond types in fingerprints
    use_features: bool, default False
        Whether to use feature information instead of atom information
    **kwargs
        Additional arguments to pass to CircularFingerprint constructor

    Returns
    -------
    CircularFingerprint
        Configured Morgan fingerprint featurizer
    """
    # Ensure registries are populated
    if not FEATURIZER_REGISTRY:
        register_models_and_featurizers()

    # Get the CircularFingerprint class
    if "CircularFingerprint" in FEATURIZER_REGISTRY:
        morgan_class = FEATURIZER_REGISTRY["CircularFingerprint"]
    else:
        # Try to import directly
        try:
            from deepchem.feat.molecule_featurizers import CircularFingerprint
            morgan_class = CircularFingerprint
        except ImportError:
            raise ImportError("Could not import CircularFingerprint from DeepChem")

    # Create and return the featurizer
    return morgan_class(
        radius=radius,
        size=size,
        use_chirality=use_chirality,
        use_bond_types=use_bond_types,
        use_features=use_features,
        **kwargs
    )


class EasyModelHub:
    """
    A simple hub interface for managing pretrained DeepChem models.

    This class provides functionality to list, download, and load pretrained models,
    similar to the Hugging Face model hub interface.
    """

    def __init__(self, models_dir: Union[str, Path] = None):
        """
        Initialize the model hub.

        Parameters
        ----------
        models_dir: str or Path, optional
            Directory to store downloaded models. If None, uses a default location.
        """
        if models_dir is None:
            self.models_dir = Path.home() / ".deepchem" / "models"
        else:
            self.models_dir = Path(models_dir)

        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Ensure registries are populated
        if not MODEL_REGISTRY or not FEATURIZER_REGISTRY:
            register_models_and_featurizers()

    def list_local_models(self) -> List[str]:
        """
        List locally available pretrained models.

        Returns
        -------
        list of str
            Names of locally available models
        """
        return [d.name for d in self.models_dir.iterdir()
                if d.is_dir() and (d / "metadata.json").exists()]

    def download_model(self, model_id: str, source_url: str) -> str:
        """
        Download a pretrained model from a URL.

        Parameters
        ----------
        model_id: str
            Identifier for the model
        source_url: str
            URL to download the model from

        Returns
        -------
        str
            Path to the downloaded model directory
        """
        import requests
        import zipfile
        import shutil

        model_dir = self.models_dir / model_id

        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            zip_path = temp_path / "model.zip"

            # Download model
            print(f"Downloading model from {source_url}...")
            response = requests.get(source_url, stream=True)
            response.raise_for_status()

            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Extract model
            print(f"Extracting model to {model_dir}...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(temp_path)

            # Move to final location
            if model_dir.exists():
                shutil.rmtree(model_dir)

            # Find the actual model directory in the extracted content
            extracted_dirs = [d for d in temp_path.iterdir() if d.is_dir()]
            if extracted_dirs:
                source_dir = extracted_dirs[0]
            else:
                source_dir = temp_path

            shutil.copytree(source_dir, model_dir)

        return str(model_dir)

    def load_model(self, model_id: str, **kwargs) -> Union[Model, Tuple[Model, MolecularFeaturizer]]:
        """
        Load a pretrained model by ID.

        Parameters
        ----------
        model_id: str
            Identifier for the model
        **kwargs
            Additional arguments to pass to load_pretrained

        Returns
        -------
        Model or (Model, MolecularFeaturizer)
            The loaded model and optionally its featurizer
        """
        model_dir = self.models_dir / model_id

        if not model_dir.exists() or not (model_dir / "metadata.json").exists():
            raise ValueError(f"Model {model_id} not found in local directory. "
                           f"Please download it first using download_model.")

        return load_pretrained(model_dir, **kwargs)
