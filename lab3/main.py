from scipy.io import wavfile
from sklearn.decomposition import PCA
import numpy as np

samplerate, audio_file = wavfile.read('comare-icontinue.wav')  # cчитывание музыки в формаnt wav
signal_wav = audio_file[:, 0] #заполнения считанного сигнала нулями


def pca_reduce(signal, n_components, block_size=1024):
    # дополнение сигнал нулями, для того,чтобы он делился на block_size
    samples = len(signal)
    hanging = block_size - np.mod(samples, block_size) #остаток от деления
    padded = np.lib.pad(signal, (0, hanging), 'constant', constant_values=0) #заполения нулями
    # изменение формы сигнала на размерность 1024
    reshaped = padded.reshape((len(padded) // block_size, block_size))
    # Метод главных компонент
    pca = PCA(n_components=n_components)
    pca.fit(reshaped)
    transformed = pca.transform(reshaped)
    reconstructed = pca.inverse_transform(transformed).reshape((len(padded)))
    return pca, transformed, reconstructed


reconstructed = pca_reduce(signal_wav, 250, 1024)
wavfile.write("comare-icontinue_After_PCA.wav", samplerate,
              reconstructed[2].astype(np.int16))  # запись полученного результата
