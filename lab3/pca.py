from sklearn.decomposition import PCA
import numpy as np


def pca_reduce(signal, n_components, block_size=1024):
    samples = len(signal)
    hanging = block_size - np.mod(samples, block_size)
    padded = np.lib.pad(signal, (0, hanging), 'constant', constant_values=0)
    reshaped = padded.reshape((len(padded) // block_size, block_size))
    pca = PCA(n_components=n_components)
    pca.fit(reshaped)
    transformed = pca.transform(reshaped)
    reconstructed = pca.inverse_transform(transformed).reshape((len(padded)))
    return pca, transformed, reconstructed