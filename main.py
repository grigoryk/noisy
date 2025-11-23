#!/usr/bin/env python3

import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import butter, sosfilt
import time
import line_profiler


class AudioVisualizer:
    @line_profiler.profile
    def __init__(self, sample_rate, num_bins=100, highpass_freq=80, smoothing=0.5):
        # Performance tuning
        self.update_interval_ms = 20  # milliseconds between frames (lower = higher FPS, more CPU)
        
        # FPS tracking
        self.last_frame_time = None
        self.frame_intervals = []
        self.fps_update_interval = 10  # update FPS display every N frames
        self.frame_counter = 0
        
        # Waveform tuning
        self.waveform_zoom = 10 # how much padding around signal (higher = more padding, less jumpy)
        self.ylim_history_size = 8  # how quickly zoom adjusts to loudness changes (higher = slower)
        self.ylim_smoothing = 0.8  # how smooth zoom transitions are (higher = smoother)
        
        # Frequency bands tuning
        self.num_bins = num_bins  # how many bars to display (higher = more detail)
        self.bands_max_freq = 8000  # highest frequency to analyze and display (Hz)
        self.bands_baseline_percentile = 0  # cuts off bottom N% as noise floor (higher = more cutoff)
        self.bands_magnitude_offset = 80  # shifts magnitude values up (dB offset for visibility)
        self.bands_magnitude_scale = 80  # normalization factor for colors
        self.bands_smoothing = 0.7  # how smooth bar transitions are (higher = smoother)
        
        # Voice frequency bands tuning
        self.voice_num_bins = 30  # bars for voice range
        self.voice_max_freq = 1000  # focus on fundamental voice frequencies (Hz)
        self.voice_noise_threshold = 15  # dB above noise floor to show a point
        self.voice_noise_history_size = 50  # frames to track for noise floor estimation
        
        # Spectrogram tuning
        self.spectrogram_max_freq = 8000  # max frequency to display (Hz)
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
        self.palette = ['#0a1628', '#513EE6', '#9354E3', '#C254E3', '#E354CA', '#E60E3F']
        self.waveform_color = '#E354CA'  # bright magenta
        self.border_color = '#D074EB'  # light purple for borders
        self.text_color = '#C254E3'  # medium purple for text
        self.label_alpha = 0.35  # opacity for all axis labels and ticks (lower = more faint)
        self.label_fontsize = 8  # font size for tick labels
        self.waveform_linewidth = 1.5  # thickness of waveform line
        self.waveform_alpha = 0.9  # opacity of waveform line
        # Create custom colormaps from palette for all visualizations
        self.spectrogram_colormap = LinearSegmentedColormap.from_list('purple_spec', self.palette)
        self.bands_colormap = LinearSegmentedColormap.from_list('purple_bands', self.palette)
        
        self.sample_rate = sample_rate
        self.audio_buffer = None
        self.prev_magnitude = None
        self.prev_ylim = None
        self.prev_bin_magnitudes = None
        self.prev_voice_bin_magnitudes = None
        self.ylim_history = []
        self.voice_noise_floor_history = []
        
        # Pre-compute mel filter bank to avoid recalculating every frame
        self._setup_mel_filterbank()
        
        self.sos = butter(4, highpass_freq, btype='high', fs=sample_rate, output='sos')

        plt.rcParams['toolbar'] = 'None'
        plt.rcParams['text.color'] = self.text_color
        plt.rcParams['axes.labelcolor'] = self.text_color
        plt.rcParams['xtick.color'] = self.border_color
        plt.rcParams['ytick.color'] = self.border_color
        self.fig = plt.figure(figsize=(16, 9), facecolor=self.background_color)
        self.fig.subplots_adjust(left=0.06, right=0.97, top=0.94, bottom=0.06)
        
        gs = self.fig.add_gridspec(3, 1, height_ratios=[1.5, 1, 1.5], hspace=0.2)
        
        self.ax_spectrogram = self.fig.add_subplot(gs[0], facecolor=self.background_color)
        self.ax_wave = self.fig.add_subplot(gs[1], facecolor=self.background_color)
        
        # Split bottom row into voice (1/3) and full range (2/3)
        gs_bottom = gs[2].subgridspec(1, 2, width_ratios=[1, 2], wspace=0.15)
        self.ax_voice_bars = self.fig.add_subplot(gs_bottom[0], projection='polar', facecolor=self.background_color)
        self.ax_bars = self.fig.add_subplot(gs_bottom[1], facecolor=self.background_color)
        
        self.setup_waveform()
        self.setup_spectrogram()
        self.setup_voice_frequency_bands()
        self.setup_frequency_bands()
        self.setup_fps_display()
    
    def setup_waveform(self):
        self.line_wave, = self.ax_wave.plot([], [], lw=self.waveform_linewidth, color=self.waveform_color, alpha=self.waveform_alpha)
        self.ax_wave.set_ylim(-1, 1)
        self.ax_wave.set_xticks([])
        
        # Add scale text indicator
        self.scale_text = self.ax_wave.text(0.02, 0.95, '', transform=self.ax_wave.transAxes,
                                           verticalalignment='top', fontsize=self.label_fontsize,
                                           color=self.text_color, alpha=0.6,
                                           bbox=dict(boxstyle='round,pad=0.3', facecolor=self.background_color, 
                                                    edgecolor=self.border_color, alpha=0.3))
        
        for spine in self.ax_wave.spines.values():
            spine.set_visible(False)
        self.ax_wave.tick_params(axis='y', colors=self.border_color, labelsize=self.label_fontsize, which='both')
        for label in self.ax_wave.get_yticklabels():
            label.set_alpha(self.label_alpha)
    
    def setup_spectrogram(self):
        # Use mel-scale frequency bins for perceptually meaningful spacing
        self.spectrogram_data = np.zeros((100, self.mel_banks))
        self.spectrogram_img = self.ax_spectrogram.imshow(self.spectrogram_data, 
                                                           aspect='auto', origin='lower',
                                                           cmap=self.spectrogram_colormap, extent=[0, self.spectrogram_max_freq, 0, 100])
        self.ax_spectrogram.set_yticks([])
        # Show logarithmically-spaced Hz labels
        self.ax_spectrogram.set_xticks([100, 200, 500, 1000, 2000, 4000, 8000])
        self.ax_spectrogram.xaxis.tick_top()
        self.ax_spectrogram.xaxis.set_label_position('top')
        for spine in self.ax_spectrogram.spines.values():
            spine.set_visible(False)
        self.ax_spectrogram.tick_params(axis='x', colors=self.border_color, labelsize=self.label_fontsize, which='both')
        for label in self.ax_spectrogram.get_xticklabels():
            label.set_alpha(self.label_alpha)
    
    def setup_voice_frequency_bands(self):
        self.voice_bars = None
        self.ax_voice_bars.set_ylim(0, 60)
        self.ax_voice_bars.set_theta_zero_location('N')
        self.ax_voice_bars.set_theta_direction(-1)
        self.ax_voice_bars.set_facecolor(self.background_color)
        self.ax_voice_bars.grid(True, alpha=0.2, color=self.border_color)
        self.ax_voice_bars.spines['polar'].set_visible(False)
        # Reduce number of angle labels
        self.ax_voice_bars.set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
        self.ax_voice_bars.set_xticklabels(['250', '500', '750', '1000'])
        # Reduce radial labels
        self.ax_voice_bars.set_yticks([20, 40, 60])
        self.ax_voice_bars.tick_params(colors=self.border_color, labelsize=self.label_fontsize)
        for label in self.ax_voice_bars.get_xticklabels() + self.ax_voice_bars.get_yticklabels():
            label.set_alpha(self.label_alpha)
    
    def setup_frequency_bands(self):
        self.bars = None
        self.ax_bars.set_xlim(0, self.bands_max_freq)
        ylabel = self.ax_bars.set_ylabel('Magnitude (dB)', color=self.text_color)
        ylabel.set_alpha(self.label_alpha)
        self.ax_bars.set_ylim(16, 100)
        for spine in self.ax_bars.spines.values():
            spine.set_visible(False)
        self.ax_bars.tick_params(axis='both', colors=self.border_color, labelsize=self.label_fontsize, which='both')
        for label in self.ax_bars.get_xticklabels() + self.ax_bars.get_yticklabels():
            label.set_alpha(self.label_alpha)
    
    def setup_fps_display(self):
        self.fps_text = self.fig.text(0.98, 0.02, '', ha='right', va='bottom',
                                     fontsize=10, color=self.text_color, alpha=0.6,
                                     bbox=dict(boxstyle='round,pad=0.3', facecolor=self.background_color,
                                              edgecolor=self.border_color, alpha=0.3))
    
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
            
            # Update scale indicator text
            self.scale_text.set_text(f'±{ylim:.3f}')
    
    def _normalize_magnitude(self, data, noise_floor_percentile):
        """Remove noise floor and normalize magnitude data to 0-1 range."""
        noise_floor = np.percentile(data, noise_floor_percentile)
        data_clean = np.maximum(data - noise_floor, 0)
        data_normalized = (data_clean - np.min(data_clean)) / (np.max(data_clean) - np.min(data_clean) + 1e-10)
        return np.clip(data_normalized, 0, 1)
    
    def _setup_mel_filterbank(self):
        mel_low = 2595 * np.log10(1 + self.mel_freq_low / 700)
        mel_high = 2595 * np.log10(1 + self.mel_freq_high / 700)
        mel_points = np.linspace(mel_low, mel_high, self.mel_banks + 2)
        self.mel_hz_points = 700 * (10**(mel_points / 2595) - 1)
    
    def update_spectrogram(self, freqs, magnitude_db):
        hz_points = self.mel_hz_points
        
        # Calculate energy in each mel-spaced frequency band
        mel_energies = []
        for i in range(self.mel_banks):
            mask = (freqs >= hz_points[i]) & (freqs < hz_points[i+2])
            if np.any(mask):
                mel_energies.append(np.mean(magnitude_db[mask]))
            else:
                mel_energies.append(-100)  # Very low dB value for empty bins
        
        mel_magnitude = np.array(mel_energies)
        mel_normalized = self._normalize_magnitude(mel_magnitude, self.spectrogram_noise_floor)
        mel_normalized = np.power(mel_normalized, self.spectrogram_power)
        
        self.spectrogram_data = np.roll(self.spectrogram_data, 1, axis=0)
        self.spectrogram_data[0, :] = mel_normalized
        self.spectrogram_img.set_data(self.spectrogram_data)
        self.spectrogram_img.set_clim(0, 1)
    
    def update_voice_frequency_bands(self, visible_freqs, visible_magnitude):
        # Get voice frequency range
        voice_mask = visible_freqs <= self.voice_max_freq
        voice_freqs = visible_freqs[voice_mask]
        voice_magnitude = visible_magnitude[voice_mask]
        
        # Track noise floor over time
        current_noise_floor = np.percentile(voice_magnitude, 20)
        self.voice_noise_floor_history.append(current_noise_floor)
        if len(self.voice_noise_floor_history) > self.voice_noise_history_size:
            self.voice_noise_floor_history.pop(0)
        
        # Calculate adaptive noise floor (average over history)
        adaptive_noise_floor = np.mean(self.voice_noise_floor_history)
        
        # Create bins for polar chart
        bin_edges = np.linspace(0, self.voice_max_freq, self.voice_num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Interpolate magnitude at bin centers
        bin_magnitudes = np.interp(bin_centers, voice_freqs, voice_magnitude) - adaptive_noise_floor
        
        # Filter out bins below threshold
        threshold = self.voice_noise_threshold
        bin_magnitudes = np.maximum(bin_magnitudes - threshold, 0)
        
        # Smooth the data
        if self.prev_voice_bin_magnitudes is not None:
            bin_magnitudes = self.bands_smoothing * self.prev_voice_bin_magnitudes + (1 - self.bands_smoothing) * bin_magnitudes
        self.prev_voice_bin_magnitudes = bin_magnitudes
        
        # Convert frequency bins to angles (0 to 2π)
        theta = np.linspace(0, 2 * np.pi, self.voice_num_bins, endpoint=False)
        width = 2 * np.pi / self.voice_num_bins
        
        # Normalize for colors
        normalized = np.clip(bin_magnitudes / 60, 0, 1)
        colors = self.bands_colormap(normalized)
        
        # Update polar bar chart
        if self.voice_bars:
            # Batch update heights and colors
            for bar, height, color in zip(self.voice_bars, bin_magnitudes, colors):
                bar.set_height(height)
                bar.set_color(color)
        else:
            self.voice_bars = self.ax_voice_bars.bar(theta, bin_magnitudes, width=width, color=colors, alpha=0.8)
    
    @line_profiler.profile
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
        
        if self.prev_bin_magnitudes is not None:
            bin_magnitudes = self.bands_smoothing * self.prev_bin_magnitudes + (1 - self.bands_smoothing) * bin_magnitudes
        self.prev_bin_magnitudes = bin_magnitudes
        
        bin_widths = bin_edges[1] - bin_edges[0]
        
        normalized = np.clip(bin_magnitudes / self.bands_magnitude_scale, 0, 1)
        colors = self.bands_colormap(normalized)
        
        if self.bars:
            # Batch update heights and colors (more efficient than loop)
            for bar, height, color in zip(self.bars, bin_magnitudes, colors):
                bar.set_height(height)
                bar.set_color(color)
        else:
            self.bars = self.ax_bars.bar(bin_centers, bin_magnitudes, width=bin_widths * 0.8, color=colors)
    
    @line_profiler.profile
    def update(self, _frame):
        frame_start = time.perf_counter()
        
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
        self.update_voice_frequency_bands(visible_freqs, visible_magnitude)
        self.update_frequency_bands(visible_freqs, visible_magnitude)
        
        # Update FPS display - measure actual interval between frames
        current_time = time.perf_counter()
        if self.last_frame_time is not None:
            frame_interval = current_time - self.last_frame_time
            self.frame_intervals.append(frame_interval)
            if len(self.frame_intervals) > 30:
                self.frame_intervals.pop(0)
        self.last_frame_time = current_time
        
        self.frame_counter += 1
        if self.frame_counter >= self.fps_update_interval and len(self.frame_intervals) > 0:
            avg_interval = np.mean(self.frame_intervals)
            fps = 1.0 / avg_interval if avg_interval > 0 else 0
            update_time = current_time - frame_start
            self.fps_text.set_text(f'{fps:.1f} FPS ({update_time*1000:.1f}ms)')
            self.frame_counter = 0
        
        return []
    
    def audio_callback(self, indata, _frames, _time, status):
        if status:
            print(f"Status: {status}")
        
        filtered = sosfilt(self.sos, indata[:, 0])
        self.audio_buffer = filtered.copy()
    
    @line_profiler.profile
    def run(self, device_id):
        with sd.InputStream(device=device_id, callback=self.audio_callback, channels=1, samplerate=self.sample_rate):
            manager = plt.get_current_fig_manager()
            manager.full_screen_toggle()
            ani = FuncAnimation(self.fig, self.update, interval=self.update_interval_ms, blit=False, cache_frame_data=False)
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
