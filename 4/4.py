import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import sounddevice as sd

SAMPLE_RATE = 48000  
DURATION = 3       
FREQUENCY = 440    
SEED_M = 13     
SEED_D = 32

RANDOM_SEED = SEED_M * 100 + SEED_D

WATERMARK_MESSAGE = np.array([1, 0, 1, 0, 1])
print(f"Watermark binary array: {WATERMARK_MESSAGE}")

WEIGHTS = [0.1, 0.01] 

def generate_sine_wave(frequency, duration, sample_rate):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    amplitude = np.sin(2 * np.pi * frequency * t)
    return amplitude / np.max(np.abs(amplitude))

def embed_watermark(audio_signal, watermark, weight, random_seed):
    np.random.seed(random_seed)
    
    # Generate PN sequence for the full audio length
    full_pn_sequence = np.random.randn(len(audio_signal))

    # Convert binary watermark (0s and 1s) to bipolar (-1s and 1s)
    bipolar_watermark = np.array([1 if bit == 1 else -1 for bit in watermark])    
    repeated_watermark_bipolar = np.tile(bipolar_watermark, int(np.ceil(len(audio_signal) / len(bipolar_watermark))))[:len(audio_signal)]

    watermark_component_embedded = weight * repeated_watermark_bipolar * full_pn_sequence

    # Spread spectrum embedding: s_w[n] = s[n] + alpha * m_bipolar[n] * p[n]
    watermarked_signal = audio_signal + watermark_component_embedded

    watermarked_signal = np.clip(watermarked_signal, -1.0, 1.0)
    
    return watermarked_signal, watermark_component_embedded 

def global_dot_product_detection(original, suspect, expected_watermark_component):
    return np.dot(suspect - original, expected_watermark_component) / len(expected_watermark_component)

def plot_signal(signal, title, filename=None):
    plt.figure(figsize=(12, 4))
    plt.plot(signal)
    plt.title(title)
    plt.xlabel('Sampel')
    plt.ylabel('Amplitudo')
    plt.grid(True)
    if filename:
        plt.savefig(filename)
    plt.show()

def save_and_play_audio(signal, filename, sample_rate):
    audio_int16 = np.int16(signal * 32767) 
    write(filename, sample_rate, audio_int16)
    print(f"Menyimpan '{filename}'")
    
    print(f"Memainkan '{filename}'...")
    sd.play(audio_int16, sample_rate)
    sd.wait()
    print("Selesai memainkan.")

print("--- Mulai Simulasi Watermarking ---")

# 1. Generasi Sinyal Asli
original_signal = generate_sine_wave(FREQUENCY, DURATION, SAMPLE_RATE)
print(f"Sinyal asli {FREQUENCY} Hz dibuat dengan durasi {DURATION} detik.")

# Simpan dan mainkan sinyal asli
save_and_play_audio(original_signal, 'original_sine_440hz.wav', SAMPLE_RATE)
plot_signal(original_signal[:SAMPLE_RATE // 50], 'Sinyal Asli (440 Hz)', 'original_signal_plot.png') # Plot bagian awal saja

watermarked_signals = []
detected_watermark_results_per_bit = []
global_detection_results = {} 

# 2. Iterasi untuk Bobot yang Berbeda
for i, weight in enumerate(WEIGHTS):
    print(f"\n--- Memproses dengan Bobot (Weight) = {weight} ---")
    
    watermarked_signal, embedded_watermark_component = embed_watermark(original_signal, WATERMARK_MESSAGE, weight, RANDOM_SEED)
    watermarked_signals.append(watermarked_signal)
    
    # Simpan dan mainkan sinyal yang sudah di-watermark
    filename_wm = f'watermarked_sine_weight_{weight}.wav'
    save_and_play_audio(watermarked_signal, filename_wm, SAMPLE_RATE)
    
    # Tampilkan grafik sinyal yang sudah di-watermark
    plot_signal(watermarked_signal[:SAMPLE_RATE // 50], f'Sinyal Setelah Watermarking (Bobot: {weight})', f'watermarked_signal_plot_weight_{weight}.png')
    
    # 4. Deteksi Global Menggunakan Dot Product 
    global_detection_value = global_dot_product_detection(original_signal, watermarked_signal, embedded_watermark_component)
    global_detection_results[weight] = global_detection_value
    print(f"Nilai Deteksi Global (alpha={weight}): {global_detection_value}")


print("\n--- Simulasi Watermarking Selesai ---")

print("\n--- Deteksi Watermark ---")
for weight, value in global_detection_results.items():
    print(f"Deteksi alpha={weight}: {value}")
print("")