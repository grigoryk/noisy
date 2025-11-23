#!/usr/bin/env python3

import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PolyCollection
from scipy.signal import butter, sosfilt


class AudioVisualizer:
    def __init__(self, sample_rate, num_bins=50, highpass_freq=80, smoothing=0.5):
        self.sample_rate = sample_rate
        self.num_bins = num_bins
        self.audio_buffer = None
        self.highpass_freq = highpass_freq
        self.smoothing = smoothing
        self.prev_magnitude = None
        self.waveform_history = []
        self.max_history = 100
        
        self.sos = butter(4, highpass_freq, btype='high', fs=sample_rate, output='sos')
        
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(14, 8))
        self.ax_wave = plt.subplot(2, 2, 1)
        self.ax_fft = plt.subplot(2, 2, 2)
        self.ax_bars = plt.subplot(2, 1, 2)
        
        self.line_wave, = self.ax_wave.plot([], [], lw=1, color='cyan')
        self.ax_wave.set_ylim(-1, 1)
        self.ax_wave.set_xlabel('Sample')
        self.ax_wave.set_ylabel('Amplitude')
        self.ax_wave.set_title('Waveform')
        
        self.ax_fft.set_xlim(0, 5000)
        self.ax_fft.set_xlabel('Frequency (Hz)')
        self.ax_fft.set_ylabel('Magnitude (dB)')
        self.ax_fft.set_title('FFT Spectrum')
        self.fft_fill = None
        
        self.bars = None
        self.ax_bars.set_xlim(0, 10000)
        self.ax_bars.set_xlabel('Frequency (Hz)')
        self.ax_bars.set_ylabel('Magnitude (dB)')
        self.ax_bars.set_title('Frequency Bands')
        
        plt.tight_layout()
    
    def update(self, frame):
        if self.audio_buffer is None:
            return []
        
        self.line_wave.set_data(np.arange(len(self.audio_buffer)), self.audio_buffer)
        self.ax_wave.set_xlim(0, len(self.audio_buffer))
        
        windowed = self.audio_buffer * np.hanning(len(self.audio_buffer))
        fft = np.fft.rfft(windowed)
        magnitude = np.abs(fft)
        
        magnitude_db = 20 * np.log10(magnitude + 1e-10)
        
        if self.prev_magnitude is not None:
            magnitude_db = self.smoothing * self.prev_magnitude + (1 - self.smoothing) * magnitude_db
        
        self.prev_magnitude = magnitude_db
        
        freqs = np.fft.rfftfreq(len(self.audio_buffer), 1/self.sample_rate)
        
        visible_mask = freqs <= 10000
        visible_freqs = freqs[visible_mask]
        visible_magnitude = magnitude_db[visible_mask]
        
        if self.fft_fill:
            self.fft_fill.remove()
        self.fft_fill = self.ax_fft.fill_between(visible_freqs, visible_magnitude, 
                                                   alpha=0.7, color='magenta', 
                                                   edgecolor='cyan', linewidth=2)
        
        y_max = np.max(visible_magnitude)
        threshold = y_max - 60
        y_min = max(np.min(visible_magnitude), threshold)
        y_range = y_max - y_min
        self.ax_fft.set_ylim(y_min - y_range * 0.3, y_max + y_range * 0.3)
        
        bin_edges = np.linspace(0, 10000, self.num_bins + 1)
        bin_magnitudes = []
        for i in range(self.num_bins):
            mask = (visible_freqs >= bin_edges[i]) & (visible_freqs < bin_edges[i+1])
            if np.any(mask):
                bin_magnitudes.append(np.max(visible_magnitude[mask]) + 80)
            else:
                bin_magnitudes.append(0)
        
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_widths = bin_edges[1] - bin_edges[0]
        
        if self.bars:
            for bar, height in zip(self.bars, bin_magnitudes):
                bar.set_height(height)
        else:
            normalized = np.clip(np.array(bin_magnitudes) / 80, 0, 1)
            colors = plt.cm.plasma(normalized)
            self.bars = self.ax_bars.bar(bin_centers, bin_magnitudes, width=bin_widths * 0.8, color=colors)
        
        normalized = np.clip(np.array(bin_magnitudes) / 80, 0, 1)
        colors = plt.cm.plasma(normalized)
        for bar, color in zip(self.bars, colors):
            bar.set_color(color)
        
        self.ax_bars.set_ylim(16, 100)
        
        return [self.line_wave, self.fft_fill] + list(self.bars)
    
    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Status: {status}")
        
        filtered = sosfilt(self.sos, indata[:, 0])
        self.audio_buffer = filtered.copy()
    
    def run(self, device_id):
        with sd.InputStream(device=device_id, callback=self.audio_callback, channels=1, samplerate=self.sample_rate):
            ani = FuncAnimation(self.fig, self.update, interval=50, blit=True)
            plt.show()


def main():
    try:
        print("Available audio devices:")
        devices = sd.query_devices()
        print(devices)
        print("\n" + "="*60)
        
        default_input = sd.default.device[0]
        if default_input == -1:
            print("\nNo default input device found.")
            print("Available input devices:")
            for i, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                    print(f"  {i}: {dev['name']}")
            
            if not any(dev['max_input_channels'] > 0 for dev in devices):
                print("\nNo input devices available. Please connect a microphone.")
                return
            
            device_id = int(input("\nEnter device ID to use: "))
            device_info = sd.query_devices(device_id)
        else:
            device_id = default_input
            device_info = sd.query_devices(device_id, kind='input')
        
        sample_rate = int(device_info['default_samplerate'])

        print(f"\nStarting audio capture from: {device_info['name']}")
        print(f"Sample rate: {sample_rate} Hz")
        print("="*60 + "\n")
        
        visualizer = AudioVisualizer(sample_rate)
        visualizer.run(device_id)
    except KeyboardInterrupt:
        print("\nStopped audio capture")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you have a working audio input device.")
        print("You may need to grant microphone permissions to your terminal.")


if __name__ == "__main__":
    main()
