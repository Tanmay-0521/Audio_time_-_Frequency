import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from pydub import AudioSegment

# Load the WAV file and get its properties
sample_rate, audio_signal =  wavfile.read('hello.wav')


# Calculate the length of the audio signal
signal_length = len(audio_signal)

# Calculate time values
time = np.arange(0, signal_length) / sample_rate

# Perform DFT (Discrete Fourier Transform) on the audio signal
dft_result = np.fft.fft(audio_signal)
frequencies = np.fft.fftfreq(signal_length, 1 / sample_rate)

# Calculate the magnitude spectrum (absolute values of the complex DFT result)
magnitude_spectrum = np.abs(dft_result)

# Plot the audio signal in the time domain
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.title('Audio Signal in Time Domain')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.plot(time, audio_signal, color='b')
plt.grid()

# Plot the magnitude spectrum in the frequency domain
plt.subplot(2, 1, 2)
plt.title('Magnitude Spectrum of Audio Signal')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.xlim(0, sample_rate / 2)  # Show only positive frequencies
plt.grid()
plt.plot(frequencies[:signal_length // 2], magnitude_spectrum[:signal_length // 2], color='r')
plt.tight_layout()
plt.show()
