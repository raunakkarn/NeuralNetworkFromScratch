import numpy as np

def xavier(in_features, out_features):
    limit = np.sqrt(6 / (in_features + out_features))
    return np.random.uniform(
        -limit,
        limit,
        (in_features, out_features)
    )

def he(in_features, out_features):
    std = np.sqrt(2 / in_features)
    return np.random.randn(
        in_features,
        out_features
    ) * std
