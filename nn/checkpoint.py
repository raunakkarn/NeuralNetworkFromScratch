import json
import numpy as np


def save_model(model, filename, metadata=None):
    """
    Save all trainable parameters along with optional metadata.
    """

    params = {}

    layer_idx = 0

    for layer in model.layers:
        if hasattr(layer, "W"):
            params[f"W{layer_idx}"] = layer.W
            params[f"b{layer_idx}"] = layer.b
            layer_idx += 1

    np.savez(filename, **params)

    if metadata is not None:
        meta_file = filename.replace(".npz", ".json")
        with open(meta_file, "w") as f:
            json.dump(metadata, f, indent=4)


def load_model(model, filename):
    """
    Load saved weights into an existing model.
    """

    params = np.load(filename)

    layer_idx = 0

    for layer in model.layers:
        if hasattr(layer, "W"):
            layer.W = params[f"W{layer_idx}"]
            layer.b = params[f"b{layer_idx}"]
            layer_idx += 1
