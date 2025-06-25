import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment

def analyze_audio(file_path: str, title: str) -> None:
    """
    Visualize the waveform and spectrogram of an audio file.
    """
    sample_rate, data = wavfile.read(file_path)

    # If stereo, convert to mono
    if data.ndim > 1:  
        data = np.mean(data, axis=1)
    
    # Plot waveform
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(data)
    plt.title(f'Waveform of {title}')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')

    # Plot spectrogram
    plt.subplot(2, 1, 2)
    plt.specgram(data, Fs=sample_rate, NFFT=1024, noverlap=512, cmap='viridis')
    plt.title(f'Spectrogram of {title}')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(label='Intensity (dB)')
    
    plt.tight_layout()
    plt.savefig(f'{title}_waveform.png')
    plt.show()

input_file = "input.wav"
output_file = "output.mp3"

audio = AudioSegment.from_wav(input_file)
audio.export(output_file, format="mp3")
print(f"Converted {input_file} to {output_file}")

original_size = os.path.getsize(input_file)
compressed_size = os.path.getsize(output_file)

print(f"Original size: {original_size / 1024:.2f} KB")
print(f"Compressed size: {compressed_size / 1024:.2f} KB")
print(f"Compression ratio: {original_size / compressed_size:.2f}")

analyze_audio(input_file, "Input WAV File")

temp_wav = AudioSegment.from_mp3(output_file)
temp_wav.export("temp.wav", format="wav")
analyze_audio("temp.wav", "Output MP3 File")
os.remove("temp.wav")
