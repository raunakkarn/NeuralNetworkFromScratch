import numpy as np

def save_model(model, filename):
    params = {}

    layer_idx = 0

    for layer in model.layers:
        if hasattr(layer, "W"):
            params[f"W{layer_idx}"] = layer.W
            params[f"b{layer_idx}"] = layer.b
            layer_idx += 1

    np.savez(filename, **params)

def load_model(model, filename):
    params = np.load(filename)

    layer_idx = 0

    for layer in model.layers:
        if hasattr(layer, "W"):
            layer.W = params[f"W{layer_idx}"]
            layer.b = params[f"b{layer_idx}"]
            layer_idx += 1
