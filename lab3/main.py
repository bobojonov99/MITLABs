from IPython.display import Audio
from scipy.io import wavfile


samplerate, audio_file = wavfile.read('comare-icontinue.wav')
start = samplerate * 14
end = start + samplerate * 10
Audio(data=audio_file[start:end, 0], rate=samplerate)

