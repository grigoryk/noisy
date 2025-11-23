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
        self.fig = plt.figure(figsize=(16, 9))
        
        gs = self.fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.5], hspace=0.3, wspace=0.3)
        
        self.ax_wave = self.fig.add_subplot(gs[0, 0:2])
        self.ax_spectrogram = self.fig.add_subplot(gs[0, 2:4])
        self.ax_phase = self.fig.add_subplot(gs[1, 0:2])
        self.ax_mel = self.fig.add_subplot(gs[1, 2:4])
        self.ax_bars = self.fig.add_subplot(gs[2, :])
        
        self.line_wave, = self.ax_wave.plot([], [], lw=1, color='cyan')
        self.ax_wave.set_ylim(-1, 1)
        self.ax_wave.set_title('Waveform', fontsize=10)
        
        self.spectrogram_data = np.zeros((100, 50))
        self.spectrogram_img = self.ax_spectrogram.imshow(self.spectrogram_data, 
                                                           aspect='auto', origin='lower',
                                                           cmap='inferno', extent=[0, 5000, 0, 100])
        self.ax_spectrogram.set_title('Spectrogram', fontsize=10)
        self.ax_spectrogram.set_ylabel('Time')
        
        self.line_phase, = self.ax_phase.plot([], [], lw=1, color='lime')
        self.ax_phase.set_xlim(0, 2000)
        self.ax_phase.set_ylim(-np.pi, np.pi)
        self.ax_phase.set_title('Phase', fontsize=10)
        
        self.mel_img = self.ax_mel.imshow(np.zeros((100, 20)), aspect='auto', 
                                          origin='lower', cmap='viridis')
        self.ax_mel.set_title('Mel Spectrogram', fontsize=10)
        self.ax_mel.set_ylabel('Time')
        
        self.bars = None
        self.ax_bars.set_xlim(0, 10000)
        self.ax_bars.set_ylabel('Magnitude (dB)')
        self.ax_bars.set_title('Frequency Bands', fontsize=10)
        self.ax_bars.set_ylim(16, 100)
    
    def update(self, frame):
        if self.audio_buffer is None:
            return []
        
        self.line_wave.set_data(np.arange(len(self.audio_buffer)), self.audio_buffer)
        self.ax_wave.set_xlim(0, len(self.audio_buffer))
        
        windowed = self.audio_buffer * np.hanning(len(self.audio_buffer))
        fft = np.fft.rfft(windowed)
        magnitude = np.abs(fft)
        phase = np.angle(fft)
        
        magnitude_db = 20 * np.log10(magnitude + 1e-10)
        
        if self.prev_magnitude is not None:
            magnitude_db = self.smoothing * self.prev_magnitude + (1 - self.smoothing) * magnitude_db
        
        self.prev_magnitude = magnitude_db
        
        freqs = np.fft.rfftfreq(len(self.audio_buffer), 1/self.sample_rate)
        
        visible_mask = freqs <= 10000
        visible_freqs = freqs[visible_mask]
        visible_magnitude = magnitude_db[visible_mask]
        
        spec_mask = freqs <= 5000
        spec_freqs = freqs[spec_mask]
        spec_mag = magnitude_db[spec_mask]
        
        if len(spec_mag) > 0:
            spec_mag_normalized = (spec_mag - np.min(spec_mag)) / (np.max(spec_mag) - np.min(spec_mag) + 1e-10)
            spec_mag_normalized = np.clip(spec_mag_normalized, 0, 1)
            
            self.spectrogram_data = np.roll(self.spectrogram_data, 1, axis=0)
            self.spectrogram_data[0, :] = np.interp(np.linspace(0, len(spec_mag_normalized)-1, 50), 
                                                       np.arange(len(spec_mag_normalized)), spec_mag_normalized)
            self.spectrogram_img.set_data(self.spectrogram_data)
            self.spectrogram_img.set_clim(0, 1)
        
        phase_mask = freqs <= 2000
        phase_freqs = freqs[phase_mask]
        phase_vals = phase[phase_mask]
        phase_mags = magnitude[phase_mask]
        
        mag_threshold = np.max(magnitude) * 0.1
        significant_mask = phase_mags > mag_threshold
        
        if np.any(significant_mask):
            self.line_phase.set_data(phase_freqs[significant_mask], phase_vals[significant_mask])
        else:
            self.line_phase.set_data([], [])
        
        mel_banks = 20
        mel_low = 2595 * np.log10(1 + 80 / 700)
        mel_high = 2595 * np.log10(1 + 5000 / 700)
        mel_points = np.linspace(mel_low, mel_high, mel_banks + 2)
        hz_points = 700 * (10**(mel_points / 2595) - 1)
        
        mel_energies = []
        for i in range(mel_banks):
            mask = (freqs >= hz_points[i]) & (freqs < hz_points[i+2])
            if np.any(mask):
                mel_energies.append(np.mean(magnitude[mask]))
            else:
                mel_energies.append(1e-10)
        
        mel_data = np.roll(self.mel_img.get_array(), 1, axis=0)
        mel_array = np.array(mel_energies)
        mel_normalized = (mel_array - np.min(mel_array)) / (np.max(mel_array) - np.min(mel_array) + 1e-10)
        mel_normalized = np.clip(mel_normalized, 0, 1)
        mel_data[0, :] = mel_normalized
        self.mel_img.set_data(mel_data)
        self.mel_img.set_clim(0, 1)
        
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
        
        return []
    
    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Status: {status}")
        
        filtered = sosfilt(self.sos, indata[:, 0])
        self.audio_buffer = filtered.copy()
    
    def run(self, device_id):
        with sd.InputStream(device=device_id, callback=self.audio_callback, channels=1, samplerate=self.sample_rate):
            ani = FuncAnimation(self.fig, self.update, interval=50, blit=False)
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
