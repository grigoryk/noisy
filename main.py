#!/usr/bin/env python3

import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import butter, sosfilt


class AudioVisualizer:
    def __init__(self, sample_rate, num_bins=100, highpass_freq=80, smoothing=0.5):
        # Waveform tuning
        self.waveform_zoom = 2.1  # how much padding around signal (higher = more padding, less jumpy)
        self.ylim_history_size = 8  # how quickly zoom adjusts to loudness changes (higher = slower)
        self.ylim_smoothing = 0.8  # how smooth zoom transitions are (higher = smoother)
        
        # Frequency bands tuning
        self.num_bins = num_bins  # how many bars to display (higher = more detail)
        self.bands_max_freq = 8000  # highest frequency to analyze and display (Hz)
        self.bands_baseline_percentile = 0  # cuts off bottom N% as noise floor (higher = more cutoff)
        self.bands_magnitude_offset = 80  # shifts magnitude values up (dB offset for visibility)
        self.bands_magnitude_scale = 80  # normalization factor for colors
        self.bands_smoothing = 0.7  # how smooth bar transitions are (higher = smoother)
        
        # Spectrogram tuning
        self.spectrogram_max_freq = 5000  # max frequency to display (Hz)
        self.spectrogram_height = 50  # vertical resolution (time bins)
        self.spectrogram_noise_floor = 30  # percentile for noise removal (higher = more aggressive)
        self.spectrogram_power = 1.5  # contrast enhancement (higher = more contrast)
        
        # Mel spectrogram tuning
        self.mel_banks = 40  # number of frequency bands
        self.mel_freq_low = 80  # lowest frequency (Hz) - captures voice fundamentals
        self.mel_freq_high = 8000  # highest frequency (Hz) - captures consonants
        self.mel_noise_floor = 30  # percentile for noise removal (higher = cleaner)
        
        # Audio processing
        self.highpass_freq = highpass_freq  # filters out frequencies below this (Hz)
        self.smoothing = smoothing  # how much to smooth FFT data between frames (higher = smoother)
        
        # Visual styling - vibrant purple palette
        self.background_color = '#0a1628'  # deep navy
        self.palette = ['#E354CA', '#9354E3', '#C254E3', '#513EE6', '#E60E3F', '#D074EB']
        self.waveform_color = '#E354CA'  # bright magenta
        self.border_color = '#D074EB'  # light purple for borders
        self.text_color = '#C254E3'  # medium purple for text
        self.spectrogram_colormap = 'inferno'
        self.mel_colormap = 'viridis'
        # Create custom colormap from palette
        self.bands_colormap = LinearSegmentedColormap.from_list('custom_purple', self.palette)
        
        self.sample_rate = sample_rate
        self.audio_buffer = None
        self.prev_magnitude = None
        self.prev_ylim = None
        self.prev_bin_magnitudes = None
        self.ylim_history = []
        
        self.sos = butter(4, highpass_freq, btype='high', fs=sample_rate, output='sos')

        plt.rcParams['toolbar'] = 'None'
        plt.rcParams['text.color'] = self.text_color
        plt.rcParams['axes.labelcolor'] = self.text_color
        plt.rcParams['xtick.color'] = self.border_color
        plt.rcParams['ytick.color'] = self.border_color
        self.fig = plt.figure(figsize=(16, 9), facecolor=self.background_color)
        self.fig.subplots_adjust(left=0.06, right=0.97, top=0.94, bottom=0.06)
        
        gs = self.fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.5], hspace=0.2, wspace=0.2)
        
        self.ax_mel = self.fig.add_subplot(gs[0, 0:2], facecolor=self.background_color)
        self.ax_spectrogram = self.fig.add_subplot(gs[0, 2:4], facecolor=self.background_color)
        self.ax_wave = self.fig.add_subplot(gs[1, :], facecolor=self.background_color)
        self.ax_bars = self.fig.add_subplot(gs[2, :], facecolor=self.background_color)
        
        self.setup_waveform()
        self.setup_spectrogram()
        self.setup_mel()
        self.setup_frequency_bands()
    
    def setup_waveform(self):
        self.line_wave, = self.ax_wave.plot([], [], lw=1.5, color=self.waveform_color, alpha=0.9)
        self.ax_wave.set_ylim(-1, 1)
        self.ax_wave.set_title('Waveform', fontsize=10, color=self.text_color, pad=12)
        for spine in self.ax_wave.spines.values():
            spine.set_color(self.border_color)
            spine.set_linewidth(0.5)
            spine.set_alpha(0.3)
    
    def setup_spectrogram(self):
        self.spectrogram_data = np.zeros((100, self.spectrogram_height))
        self.spectrogram_img = self.ax_spectrogram.imshow(self.spectrogram_data, 
                                                           aspect='auto', origin='lower',
                                                           cmap=self.spectrogram_colormap, extent=[0, self.spectrogram_max_freq, 0, 100])
        self.ax_spectrogram.set_title('Spectrogram', fontsize=10, color=self.text_color, pad=12)
        self.ax_spectrogram.set_yticks([])
        for spine in self.ax_spectrogram.spines.values():
            spine.set_color(self.border_color)
            spine.set_linewidth(0.5)
            spine.set_alpha(0.3)
    
    def setup_mel(self):
        self.mel_img = self.ax_mel.imshow(np.zeros((100, self.mel_banks)), aspect='auto', 
                                          origin='lower', cmap=self.mel_colormap,
                                          extent=[self.mel_freq_low, self.mel_freq_high, 0, 100])
        self.ax_mel.set_title('Mel Spectrogram', fontsize=10, color=self.text_color, pad=12)
        self.ax_mel.set_yticks([])
        for spine in self.ax_mel.spines.values():
            spine.set_color(self.border_color)
            spine.set_linewidth(0.5)
            spine.set_alpha(0.3)
    
    def setup_frequency_bands(self):
        self.bars = None
        self.ax_bars.set_xlim(0, self.bands_max_freq)
        self.ax_bars.set_ylabel('Magnitude (dB)', color=self.text_color)
        self.ax_bars.set_title('Frequency Bands', fontsize=10, color=self.text_color, pad=12)
        self.ax_bars.set_ylim(16, 100)
        for spine in self.ax_bars.spines.values():
            spine.set_color(self.border_color)
            spine.set_linewidth(0.5)
            spine.set_alpha(0.3)
    
    def update_waveform(self):
        self.line_wave.set_data(np.arange(len(self.audio_buffer)), self.audio_buffer)
        self.ax_wave.set_xlim(0, len(self.audio_buffer))
        
        max_amp = np.max(np.abs(self.audio_buffer))
        if max_amp > 0:
            target_ylim = max(max_amp * self.waveform_zoom, 0.01)
            
            self.ylim_history.append(target_ylim)
            if len(self.ylim_history) > self.ylim_history_size:
                self.ylim_history.pop(0)
            
            avg_ylim = np.mean(self.ylim_history)
            
            if self.prev_ylim is not None:
                ylim = self.ylim_smoothing * self.prev_ylim + (1 - self.ylim_smoothing) * avg_ylim
            else:
                ylim = avg_ylim
            self.prev_ylim = ylim
            self.ax_wave.set_ylim(-ylim, ylim)
    
    def _normalize_magnitude(self, data, noise_floor_percentile):
        """Remove noise floor and normalize magnitude data to 0-1 range."""
        noise_floor = np.percentile(data, noise_floor_percentile)
        data_clean = np.maximum(data - noise_floor, 0)
        data_normalized = (data_clean - np.min(data_clean)) / (np.max(data_clean) - np.min(data_clean) + 1e-10)
        return np.clip(data_normalized, 0, 1)
    
    def update_spectrogram(self, freqs, magnitude_db):
        # Filter to display range and normalize with noise reduction
        spectrogram_mask = freqs <= self.spectrogram_max_freq
        spectrogram_magnitude = magnitude_db[spectrogram_mask]
        
        if len(spectrogram_magnitude) > 0:
            spectrogram_magnitude_normalized = self._normalize_magnitude(spectrogram_magnitude, self.spectrogram_noise_floor)
            spectrogram_magnitude_normalized = np.power(spectrogram_magnitude_normalized, self.spectrogram_power)
            
            self.spectrogram_data = np.roll(self.spectrogram_data, 1, axis=0)
            self.spectrogram_data[0, :] = np.interp(np.linspace(0, len(spectrogram_magnitude_normalized)-1, self.spectrogram_height), 
                                                       np.arange(len(spectrogram_magnitude_normalized)), spectrogram_magnitude_normalized)
            self.spectrogram_img.set_data(self.spectrogram_data)
            self.spectrogram_img.set_clim(0, 1)
    
    def update_mel(self, freqs, magnitude):
        # Convert frequency range to mel scale and create filter bank points
        mel_low = 2595 * np.log10(1 + self.mel_freq_low / 700)
        mel_high = 2595 * np.log10(1 + self.mel_freq_high / 700)
        mel_points = np.linspace(mel_low, mel_high, self.mel_banks + 2)
        hz_points = 700 * (10**(mel_points / 2595) - 1)
        
        # Calculate energy in each mel-spaced frequency band
        mel_energies = []
        for i in range(self.mel_banks):
            mask = (freqs >= hz_points[i]) & (freqs < hz_points[i+2])
            if np.any(mask):
                mel_energies.append(np.mean(magnitude[mask]))
            else:
                mel_energies.append(1e-10)
        
        mel_spectrogram_data = np.roll(self.mel_img.get_array(), 1, axis=0)
        mel_magnitude = np.array(mel_energies)
        
        mel_normalized = self._normalize_magnitude(mel_magnitude, self.mel_noise_floor)
        mel_spectrogram_data[0, :] = mel_normalized
        self.mel_img.set_data(mel_spectrogram_data)
        self.mel_img.set_clim(0, 1)
    
    def update_frequency_bands(self, visible_freqs, visible_magnitude):
        # Divide frequency range into bins and find peak magnitude in each
        bin_edges = np.linspace(0, self.bands_max_freq, self.num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Interpolate magnitude values at bin centers for smooth visualization
        bin_magnitudes = np.interp(bin_centers, visible_freqs, visible_magnitude) + self.bands_magnitude_offset
        
        # Remove baseline noise and apply smoothing
        if self.bands_baseline_percentile > 0:
            baseline = np.percentile(bin_magnitudes, self.bands_baseline_percentile)
            bin_magnitudes = np.maximum(bin_magnitudes - baseline, 0)
        
        bin_magnitudes = bin_magnitudes.tolist()
        
        if self.prev_bin_magnitudes is not None:
            bin_magnitudes = [
                self.bands_smoothing * prev + (1 - self.bands_smoothing) * curr
                for prev, curr in zip(self.prev_bin_magnitudes, bin_magnitudes)
            ]
        self.prev_bin_magnitudes = bin_magnitudes
        
        bin_widths = bin_edges[1] - bin_edges[0]
        
        normalized = np.clip(np.array(bin_magnitudes) / self.bands_magnitude_scale, 0, 1)
        colormap = plt.cm.get_cmap(self.bands_colormap)
        colors = colormap(normalized)
        
        if self.bars:
            for bar, height in zip(self.bars, bin_magnitudes):
                bar.set_height(height)
            for bar, color in zip(self.bars, colors):
                bar.set_color(color)
        else:
            self.bars = self.ax_bars.bar(bin_centers, bin_magnitudes, width=bin_widths * 0.8, color=colors)
    
    def update(self, frame):
        if self.audio_buffer is None:
            return []
        
        windowed = self.audio_buffer * np.hanning(len(self.audio_buffer))
        fft = np.fft.rfft(windowed)
        magnitude = np.abs(fft)
        
        magnitude_db = 20 * np.log10(magnitude + 1e-10)
        
        if self.prev_magnitude is not None:
            magnitude_db = self.smoothing * self.prev_magnitude + (1 - self.smoothing) * magnitude_db
        
        self.prev_magnitude = magnitude_db
        
        freqs = np.fft.rfftfreq(len(self.audio_buffer), 1/self.sample_rate)
        
        visible_mask = freqs <= self.bands_max_freq
        visible_freqs = freqs[visible_mask]
        visible_magnitude = magnitude_db[visible_mask]
        
        self.update_waveform()
        self.update_spectrogram(freqs, magnitude_db)
        self.update_mel(freqs, magnitude)
        self.update_frequency_bands(visible_freqs, visible_magnitude)
        
        return []
    
    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Status: {status}")
        
        filtered = sosfilt(self.sos, indata[:, 0])
        self.audio_buffer = filtered.copy()
    
    def run(self, device_id):
        with sd.InputStream(device=device_id, callback=self.audio_callback, channels=1, samplerate=self.sample_rate):
            manager = plt.get_current_fig_manager()
            manager.full_screen_toggle()
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
