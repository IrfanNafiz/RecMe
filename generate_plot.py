import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Define file paths
input_wav_file = os.path.join("data", "custom", "temp_record.wav")
output_image_file = "waveform_fft_output.png"
# output_image_file = "empty_plot.png"

# Read the audio file
sample_rate, audio_data = wavfile.read(input_wav_file)

# Generate the waveform and FFT plots
plt.figure(figsize=(5, 5))

# Waveform plot
plt.plot(np.arange(len(audio_data)) / sample_rate, audio_data)
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("Waveform of Recorded Audio")

# # FFT plot
# plt.subplot(2, 1, 2)
# fft_freqs = np.fft.fftfreq(len(audio_data), d=1/sample_rate)
# fft_values = np.abs(np.fft.fft(audio_data))
# plt.plot(fft_freqs[:len(fft_freqs)//2], fft_values[:len(fft_values)//2])
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Magnitude")
# plt.title("FFT of Recorded Audio")

plt.tight_layout()

# Save the combined plot
plt.savefig(output_image_file, bbox_inches="tight")
plt.close()

print(f"Temporary record's waveform and FFT plots saved as {output_image_file} successfully!")
