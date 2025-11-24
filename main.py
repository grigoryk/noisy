#!/usr/bin/env python3

import argparse
import time

import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import PolyCollection
from matplotlib.patches import Circle, Ellipse
from matplotlib.widgets import Slider, Button, RangeSlider, CheckButtons
from matplotlib.transforms import Bbox
from scipy.signal import butter, sosfilt
import line_profiler


class AudioVisualizer:
    @line_profiler.profile
    def __init__(self, sample_rate, num_bins=50, highpass_freq=100, smoothing=0.5):
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
        self.waveform_fade_decay = 0.0  # 0 = instant, closer to 1 = slower fade
        self.waveform_trail_alpha = 0.5
        self.waveform_trail_min_alpha = 0.015
        
        # Frequency bands tuning
        self.num_bins = num_bins  # how many bars to display (higher = more detail)
        self.bands_max_freq = 8000  # highest frequency to analyze and display (Hz)
        self.bands_baseline_percentile = 35  # cuts off bottom N% as noise floor (higher = more cutoff)
        self.bands_noise_history_size = 4
        self.bands_magnitude_offset = 60  # shifts magnitude values up (dB offset for visibility)
        self.bands_magnitude_scale = 60  # normalization factor for colors
        self.bands_smoothing = 0.7  # how smooth bar transitions are (higher = smoother)
        
        # Voice frequency bands tuning
        self.voice_num_bins = 30  # bars for voice range
        self.voice_max_freq = 2000  # focus on fundamental voice frequencies (Hz)
        self.voice_noise_threshold = 8  # dB above noise floor to show (lower = more detail)
        self.voice_noise_history_size = 45  # frames to track for noise floor estimation
        self.voice_amplification = 1.6  # amplify voice magnitudes for visibility
        self.voice_radius_base = 15
        self.voice_radius_max = 60
        self.noise_radius_base = 10
        self.noise_radius_max = 45
        
        # Spectrogram tuning
        self.spectrogram_max_freq = 8000  # max frequency to display (Hz)
        self.spectrogram_noise_floor = 30  # percentile for noise removal (higher = more aggressive)
        self.spectrogram_power = 1.5  # contrast enhancement (higher = more contrast)
        self.spectrogram_view_min = 0
        self.spectrogram_view_max = self.spectrogram_max_freq
        self.spectrogram_min_view_bins = 40
        self.spectrogram_noise_history_size = self.voice_noise_history_size
        self.spectrogram_subtract_enabled = True
        self.spectrogram_subtract_aggressiveness = 1.0
        self.voice_subtract_enabled = True
        self.voice_subtract_aggressiveness = 1.0
        self.show_noise_spectrogram = True
        self.show_noise_polar = True
        
        # Mel spectrogram tuning
        self.mel_banks = 40  # number of frequency bands
        self.mel_freq_low = 80  # lowest frequency (Hz) - captures voice fundamentals
        self.mel_freq_high = 8000  # highest frequency (Hz) - captures consonants
        
        # Audio processing
        self.highpass_freq = highpass_freq  # filters out frequencies below this (Hz)
        self.smoothing = smoothing  # how much to smooth FFT data between frames (higher = smoother)
        
        # Visual styling - vibrant purple palette
        self.background_color = '#0a1628'  # deep navy
        self.palette = ['#0a1628', '#513EE6', '#9354E3', '#C254E3', '#E354CA', '#E60E3F']
        self.waveform_color = '#E354CA'  # bright magenta
        self.border_color = '#D074EB'  # light purple for borders
        self.text_color = '#C254E3'  # medium purple for text
        self.toggle_panel_color = '#101c30'
        self.toggle_inactive_color = '#0d1627'
        self.toggle_active_color = '#513EE6'
        self.toggle_circle_active = '#E354CA'
        self.toggle_circle_inactive = '#1a2337'
        self.label_alpha = 0.35  # opacity for all axis labels and ticks (lower = more faint)
        self.label_fontsize = 8  # font size for tick labels
        self.waveform_linewidth = 1.5  # thickness of waveform line
        self.waveform_alpha = 0.9  # opacity of waveform line
        self.waveform_x = None
        self.waveform_len = 0
        self.waveform_ylim_epsilon = 1e-3
        self.waveform_last_data = None
        self.waveform_trail_lines = []
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
        self.voice_noise_history_frames = []
        self.spectrogram_noise_history_frames = []
        self.bands_noise_floor_history = []
        self.voice_bar_colors = None
        self.voice_bar_heights = None
        self.noise_bar_colors = None
        self.noise_bar_heights = None
        voice_angle_edges = np.linspace(0, 2 * np.pi, self.voice_num_bins + 1)
        self.voice_theta_left = voice_angle_edges[:-1]
        self.voice_theta_right = voice_angle_edges[1:]
        self._configure_voice_bins()
        self.voice_vertices = None
        self.voice_collection = None
        self.voice_paths = None
        self.noise_vertices = None
        self.noise_collection = None
        self.noise_paths = None
        self.prev_noise_levels = None
        self.bars_colors = None
        self.bars_heights = None
        self.color_change_epsilon = 1e-3
        self.height_change_epsilon = 0.5
        self.color_update_interval = 10
        self.color_frame_counter = 0
        self.should_update_colors = True
        self.window = None
        self.window_size = None
        self.mel_bin_starts = None
        self.mel_bin_ends = None
        self.mel_freq_len = None
        self.bars_vertices = None
        self.bars_collection = None
        self.bars_paths = None
        self.noise_spectrogram_img = None
        
        # Pre-compute mel filter bank to avoid recalculating every frame
        self._setup_mel_filterbank()
        
        self.sos = butter(4, highpass_freq, btype='high', fs=sample_rate, output='sos')

        plt.rcParams['toolbar'] = 'None'
        plt.rcParams['text.color'] = self.text_color
        plt.rcParams['axes.labelcolor'] = self.text_color
        plt.rcParams['xtick.color'] = self.border_color
        plt.rcParams['ytick.color'] = self.border_color
        self.fig = plt.figure(figsize=(16, 9), facecolor=self.background_color)
        canvas = getattr(self.fig, 'canvas', None)
        manager = getattr(canvas, 'manager', None)
        if manager and hasattr(manager, 'set_window_title'):
            manager.set_window_title('Noisy – Live Audio Visualizer')
        self.fig.subplots_adjust(left=0.06, right=0.78, top=0.94, bottom=0.06)
        
        gs = self.fig.add_gridspec(3, 1, height_ratios=[1.5, 1, 1.5], hspace=0.2)
        
        gs_top = gs[0].subgridspec(1, 2, width_ratios=[1, 1], wspace=0.055)
        self.ax_spectrogram = self.fig.add_subplot(gs_top[0], facecolor=self.background_color)
        self.ax_noise_spectrogram = self.fig.add_subplot(gs_top[1], facecolor=self.background_color)
        self.ax_wave = self.fig.add_subplot(gs[1], facecolor=self.background_color)
        
        # Split bottom row into three columns: voice polar, noise polar, and expanded frequency bars
        gs_bottom = gs[2].subgridspec(1, 3, width_ratios=[1, 1, 2.2], wspace=0.18)
        self.ax_voice_bars = self.fig.add_subplot(gs_bottom[0], projection='polar', facecolor=self.background_color)
        self.ax_noise_bars = self.fig.add_subplot(gs_bottom[1], projection='polar', facecolor=self.background_color)
        self.ax_bars = self.fig.add_subplot(gs_bottom[2], facecolor=self.background_color)
        self._spectrogram_pos = self.ax_spectrogram.get_position().frozen()
        self._noise_spectrogram_pos = self.ax_noise_spectrogram.get_position().frozen()
        self._spectrogram_full_pos = Bbox.union([self._spectrogram_pos, self._noise_spectrogram_pos])
        self._bars_pos = self.ax_bars.get_position().frozen()
        self._noise_bars_pos = self.ax_noise_bars.get_position().frozen()
        self._bars_expanded_pos = Bbox.union([self._bars_pos, self._noise_bars_pos])
        
        self._configure_frequency_bins(self.num_bins)
        
        self.setup_waveform()
        self.setup_spectrogram()
        self.setup_voice_frequency_bands()
        self.setup_noise_frequency_bands()
        self.setup_frequency_bands()
        self.setup_fps_display()
        self.setup_frequency_tuning_controls()
        self._apply_layout_toggles()
    
    def setup_waveform(self):
        self.line_wave, = self.ax_wave.plot([], [], lw=self.waveform_linewidth, color=self.waveform_color, alpha=self.waveform_alpha)
        self.ax_wave.set_ylim(-1, 1)
        self.ax_wave.set_xticks([])
        self.waveform_trail_lines = []
        self.waveform_last_data = None
        
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

    def _decay_waveform_trails(self):
        if not self.waveform_trail_lines:
            return
        if self.waveform_fade_decay <= 0:
            for line in self.waveform_trail_lines:
                line.remove()
            self.waveform_trail_lines.clear()
            return
        kept = []
        for line in self.waveform_trail_lines:
            new_alpha = line.get_alpha() * self.waveform_fade_decay
            if new_alpha <= self.waveform_trail_min_alpha:
                line.remove()
                continue
            line.set_alpha(new_alpha)
            kept.append(line)
        self.waveform_trail_lines = kept

    def _add_waveform_trail(self, data):
        if data is None or self.waveform_x is None or self.waveform_fade_decay <= 0:
            return
        trail_line, = self.ax_wave.plot(self.waveform_x, data, lw=self.waveform_linewidth,
                                        color=self.waveform_color, alpha=self.waveform_trail_alpha)
        trail_line.set_zorder(self.line_wave.get_zorder() - 1)
        self.waveform_trail_lines.append(trail_line)
        if len(self.waveform_trail_lines) > 60:
            oldest = self.waveform_trail_lines.pop(0)
            oldest.remove()
    
    def setup_spectrogram(self):
        # Use mel-scale frequency bins for perceptually meaningful spacing
        self.spectrogram_full_data = np.zeros((100, self.mel_banks))
        self.noise_spectrogram_full_data = np.zeros_like(self.spectrogram_full_data)
        self.spectrogram_view_indices = np.arange(self.mel_banks)
        self.spectrogram_img = self.ax_spectrogram.imshow(
            self.spectrogram_full_data[:, self.spectrogram_view_indices],
            aspect='auto',
            origin='lower',
            cmap=self.spectrogram_colormap,
            extent=[0, self.spectrogram_max_freq, 0, 100],
        )
        self.noise_spectrogram_img = self.ax_noise_spectrogram.imshow(
            self.noise_spectrogram_full_data[:, self.spectrogram_view_indices],
            aspect='auto',
            origin='lower',
            cmap=self.spectrogram_colormap,
            extent=[0, self.spectrogram_max_freq, 0, 100],
        )
        for axis, title in ((self.ax_spectrogram, 'SPECTRUM'), (self.ax_noise_spectrogram, 'NOISE')):
            axis.set_yticks([])
            axis.set_xticks([100, 200, 500, 1000, 2000, 4000, 8000])
            axis.xaxis.tick_top()
            axis.xaxis.set_label_position('top')
            for spine in axis.spines.values():
                spine.set_visible(False)
            axis.tick_params(axis='x', colors=self.border_color, labelsize=self.label_fontsize, which='both')
            for label in axis.get_xticklabels():
                label.set_alpha(self.label_alpha)
            axis.set_title(title, color=self.text_color, fontsize=self.label_fontsize, pad=6, alpha=0.6)
        self._update_spectrogram_view(self.spectrogram_view_min, self.spectrogram_view_max, refresh_ticks=True)
    
    def setup_voice_frequency_bands(self):
        self.ax_voice_bars.set_ylim(self.voice_radius_base, self.voice_radius_max)  # Start bars away from center
        self.ax_voice_bars.set_theta_zero_location('N')
        self.ax_voice_bars.set_theta_direction(-1)
        self.ax_voice_bars.set_facecolor(self.background_color)
        self.ax_voice_bars.grid(True, alpha=0.2, color=self.border_color)
        self.ax_voice_bars.spines['polar'].set_visible(False)
        self.ax_voice_bars.set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
        # Reduce radial labels
        self.ax_voice_bars.set_yticks([30, 45, 60])
        self.ax_voice_bars.tick_params(colors=self.border_color, labelsize=self.label_fontsize)
        self._update_polar_frequency_labels()
        for label in self.ax_voice_bars.get_yticklabels():
            label.set_alpha(self.label_alpha)

    def setup_noise_frequency_bands(self):
        self.ax_noise_bars.set_ylim(self.noise_radius_base, self.noise_radius_max)
        self.ax_noise_bars.set_theta_zero_location('N')
        self.ax_noise_bars.set_theta_direction(-1)
        self.ax_noise_bars.set_facecolor(self.background_color)
        self.ax_noise_bars.grid(True, alpha=0.15, color=self.border_color)
        self.ax_noise_bars.spines['polar'].set_visible(False)
        self.ax_noise_bars.set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
        self.ax_noise_bars.set_yticks([20, 35, 50])
        self.ax_noise_bars.tick_params(colors=self.border_color, labelsize=self.label_fontsize)
        self._update_polar_frequency_labels()
        for label in self.ax_noise_bars.get_yticklabels():
            label.set_alpha(self.label_alpha)
        self.ax_noise_bars.set_title('NOISE', color=self.text_color, fontsize=self.label_fontsize, pad=10, alpha=0.6)
    
    def setup_frequency_bands(self):
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
        self.fps_text = self.fig.text(0.98, 0.98, '', ha='right', va='top',
                                     fontsize=10, color=self.text_color, alpha=0.6,
                                     bbox=dict(boxstyle='round,pad=0.3', facecolor=self.background_color,
                                              edgecolor=self.border_color, alpha=0.3))
    
    def setup_frequency_tuning_controls(self):
        panel_left = 0.71
        panel_width = 0.17
        button_height = 0.04
        slider_height = 0.026
        slider_spacing = 0.006
        top = 0.9
        button_ax = self.fig.add_axes([panel_left, top, panel_width, button_height])
        button_ax.set_facecolor(self.background_color)
        self.tuning_toggle_button = Button(button_ax, 'Hide Tuning', color=self.border_color,
                                           hovercolor=self.palette[3])
        self.tuning_toggle_button.label.set_color(self.background_color)
        self.tuning_toggle_button.on_clicked(self._toggle_tuning_controls)
        current_top = top - button_height - 0.012
        slider_defs = [
            ('Baseline %', 0, 60, self.bands_baseline_percentile, 1, self._on_baseline_slider,
             'Trims this percentile of bar magnitudes. More = stricter noise cut, less = fuller bars'),
            ('Noise Frames', 1, 10, self.bands_noise_history_size, 1, self._on_noise_history_slider,
             'Rolling frames for bar noise floor. More = steadier baseline, less = quicker reaction'),
            ('Offset (dB)', 20, 100, self.bands_magnitude_offset, 1, self._on_offset_slider,
             'Adds dB offset before clipping. More = brighter low notes, less = more headroom'),
            ('Scale (dB)', 20, 120, self.bands_magnitude_scale, 1, self._on_scale_slider,
             'dB span for bar colors. More = softer gradients, less = more contrast'),
            ('Smoothing', 0.0, 0.95, self.bands_smoothing, 0.01, self._on_smoothing_slider,
             'EMA between frames. More = smoother motion, less = snappier response'),
            ('Wave Fade', 0.0, 0.99, self.waveform_fade_decay, 0.01, self._on_waveform_fade_slider,
             'Trail decay per frame. More = trails linger, less = almost no afterglow'),
            ('Voice Gain', 0.5, 3.0, self.voice_amplification, 0.05, self._on_voice_gain_slider,
             'Gain after noise removal. More = louder bars, less = quieter bars'),
            ('Voice Noise Frames', 5, 90, self.voice_noise_history_size, 1, self._on_voice_noise_history_slider,
             'Frames in voice baseline. More = steadier floor, less = reacts to new noise fast'),
            ('Voice Threshold', 0.0, 18.0, self.voice_noise_threshold, 0.5, self._on_voice_threshold_slider,
             'Extra dB before showing. More = only big speech, less = sensitive to quiet sounds'),
          ('Voice Max Hz', 500, 8000, self.voice_max_freq, 50, self._on_voice_max_freq_slider,
           'Upper limit for voice/noise polar charts. More = see consonants, less = focus on lows'),
            ('Spectro Aggro', 0.25, 3.0, self.spectrogram_subtract_aggressiveness, 0.05,
             self._on_spectrogram_subtract_aggressiveness_slider,
             'Multiplier on the noise frame before subtracting from the spectrogram'),
            ('Voice Aggro', 0.25, 3.0, self.voice_subtract_aggressiveness, 0.05,
             self._on_voice_subtract_aggressiveness_slider,
             'Multiplier on the voice noise baseline before subtracting it'),
            ('Spectro Bins', 10, 120, self.spectrogram_min_view_bins, 1, self._on_spectrogram_bins_slider,
             'Minimum bins in spectrogram. More = smoother view, less = blockier but faster'),
        ]
        self.tuning_sliders = []
        self.tuning_toggle_widgets = []
        self.tuning_controls_visible = True
        self._ensure_tuning_tooltip_support()
        for label, vmin, vmax, init, step, callback, description in slider_defs:
            ax = self.fig.add_axes([panel_left, current_top - slider_height, panel_width, slider_height])
            ax.set_facecolor(self.background_color)
            slider = Slider(ax=ax, label=label, valmin=vmin, valmax=vmax, valinit=init,
                            valstep=step)
            slider.on_changed(callback)
            slider.label.set_color(self.text_color)
            slider.valtext.set_color(self.text_color)
            self.tuning_sliders.append(slider)
            self._register_tuning_hover(slider.label, description)
            current_top -= slider_height + slider_spacing
        range_height = 0.035
        range_ax = self.fig.add_axes([panel_left, current_top - range_height, panel_width, range_height])
        range_ax.set_facecolor(self.background_color)
        self.spectrogram_range_slider = RangeSlider(ax=range_ax,
                                                    label='Spectro Hz',
                                                    valmin=0,
                                                    valmax=self.spectrogram_max_freq,
                                                    valinit=(self.spectrogram_view_min, self.spectrogram_view_max))
        self.spectrogram_range_slider.on_changed(self._on_spectrogram_range_slider)
        self.spectrogram_range_slider.label.set_color(self.text_color)
        self.spectrogram_range_slider.valtext.set_color(self.text_color)
        self.tuning_sliders.append(self.spectrogram_range_slider)
        self._register_tuning_hover(self.spectrogram_range_slider.label,
                                    'Choose the spectrogram frequency window (drag both handles)')
        current_top -= range_height + slider_spacing
        checkbox_height = 0.075
        subtraction_ax = self.fig.add_axes([panel_left, current_top - checkbox_height, panel_width, checkbox_height])
        subtraction_ax.set_facecolor(self.background_color)
        self.subtraction_toggle_labels = ['Spectro Subtract', 'Voice Subtract']
        self.subtraction_checkbuttons = CheckButtons(
            subtraction_ax,
            self.subtraction_toggle_labels,
            [self.spectrogram_subtract_enabled, self.voice_subtract_enabled],
        )
        self._style_checkbuttons(self.subtraction_checkbuttons)
        self.subtraction_checkbuttons.on_clicked(self._on_subtraction_toggle)
        self.tuning_toggle_widgets.append(self.subtraction_checkbuttons)
        toggle_descriptions = [
            'Enable or disable subtracting the tracked noise frame from the spectrogram',
            'Enable or disable subtracting the tracked noise floor from voice bars',
        ]
        for label_artist, description in zip(self.subtraction_checkbuttons.labels, toggle_descriptions):
            self._register_tuning_hover(label_artist, description)
        current_top -= checkbox_height + slider_spacing
        visibility_ax = self.fig.add_axes([panel_left, current_top - checkbox_height, panel_width, checkbox_height])
        visibility_ax.set_facecolor(self.background_color)
        self.noise_toggle_labels = ['Show Noise Spectro', 'Show Noise Polar']
        self.noise_visibility_checkbuttons = CheckButtons(
            visibility_ax,
            self.noise_toggle_labels,
            [self.show_noise_spectrogram, self.show_noise_polar],
        )
        self._style_checkbuttons(self.noise_visibility_checkbuttons)
        self.noise_visibility_checkbuttons.on_clicked(self._on_noise_visibility_toggle)
        self.tuning_toggle_widgets.append(self.noise_visibility_checkbuttons)
        visibility_descriptions = [
            'Toggle the dedicated noise spectrogram panel',
            'Toggle the noise polar chart and expand the bar graph',
        ]
        for label_artist, description in zip(self.noise_visibility_checkbuttons.labels, visibility_descriptions):
            self._register_tuning_hover(label_artist, description)
        self._set_tuning_controls_visible(True)
    
    def _set_tuning_controls_visible(self, visible):
        if not hasattr(self, 'tuning_sliders'):
            return
        for slider in self.tuning_sliders:
            slider.ax.set_visible(visible)
            slider.label.set_visible(visible)
            slider.valtext.set_visible(visible)
        if hasattr(self, 'tuning_toggle_widgets'):
            for widget in self.tuning_toggle_widgets:
                self._set_checkbuttons_visible(widget, visible)
        label = 'Hide Tuning' if visible else 'Show Tuning'
        if hasattr(self, 'tuning_toggle_button'):
            self.tuning_toggle_button.label.set_text(label)
        self.fig.canvas.draw_idle()
    
    def _toggle_tuning_controls(self, _event):
        self.tuning_controls_visible = not getattr(self, 'tuning_controls_visible', True)
        self._set_tuning_controls_visible(self.tuning_controls_visible)
    
    def _apply_layout_toggles(self):
        if not hasattr(self, '_spectrogram_pos'):
            return
        if self.show_noise_spectrogram:
            self.ax_spectrogram.set_position(self._spectrogram_pos)
            self.ax_noise_spectrogram.set_visible(True)
        else:
            self.ax_spectrogram.set_position(self._spectrogram_full_pos)
            self.ax_noise_spectrogram.set_visible(False)
        if self.show_noise_polar:
            self.ax_bars.set_position(self._bars_pos)
            self.ax_noise_bars.set_visible(True)
        else:
            self.ax_bars.set_position(self._bars_expanded_pos)
            self.ax_noise_bars.set_visible(False)
        self.fig.canvas.draw_idle()

    def _style_checkbuttons(self, widget):
        widget.ax.set_facecolor(self.toggle_panel_color)
        for spine in widget.ax.spines.values():
            spine.set_edgecolor(self.border_color)
            spine.set_linewidth(1.2)
        statuses = widget.get_status()
        for label, active in zip(widget.labels, statuses):
            label.set_color(self.text_color)
            label.set_fontsize(self.label_fontsize)
            label.set_alpha(1.0 if active else 0.55)
            label.set_fontweight('bold' if active else 'normal')
            base_x, base_y = label.get_position()
            label.set_position((0.32, base_y))
        self._hide_widget_checks(widget)
        self._ensure_toggle_circles(widget)
        self._update_toggle_circles(widget, statuses)
        self._sync_toggle_hit_targets(widget)

    def _set_checkbuttons_visible(self, widget, visible):
        widget.ax.set_visible(visible)
        for label in widget.labels:
            label.set_visible(visible)
        lines = getattr(widget, 'lines', None)
        if lines is not None:
            for entry in lines:
                for line in entry:
                    line.set_visible(False)
        frames = getattr(widget, '_frames', None)
        if frames is not None:
            frames.set_visible(visible)
        checks = getattr(widget, '_checks', None)
        if checks is not None:
            checks.set_visible(False)
        circles = getattr(widget, '_toggle_circles', None)
        if circles is not None:
            for circle in circles:
                circle.set_visible(visible)

    def _ensure_toggle_circles(self, widget):
        labels = getattr(widget, 'labels', None)
        if not labels:
            return
        centers = []
        for label in labels:
            x, y = label.get_position()
            centers.append((max(0.08, x - 0.11), y))
        circles = getattr(widget, '_toggle_circles', None)
        if circles is not None and len(circles) == len(centers):
            widget._toggle_centers = centers
            return
        self._remove_toggle_circles(widget)
        widget._toggle_centers = centers
        widget._toggle_circle_diameter = 0.051  # 15% smaller than previous 0.06
        widget._toggle_circles = []
        for center in centers:
            ellipse = Ellipse(center,
                              width=widget._toggle_circle_diameter,
                              height=widget._toggle_circle_diameter,
                              transform=widget.ax.transAxes,
                              facecolor=self.toggle_circle_inactive,
                              edgecolor=self.border_color,
                              linewidth=1.0)
            widget.ax.add_patch(ellipse)
            widget._toggle_circles.append(ellipse)

    def _hide_widget_checks(self, widget):
        frames = getattr(widget, '_frames', None)
        if frames is not None:
            offsets = frames.get_offsets()
            count = len(offsets)
            if count:
                transparent = np.zeros((count, 4))
                frames.set_visible(True)
                frames.set_facecolor(transparent)
                frames.set_edgecolor(transparent)
                frames.set_linewidths(np.zeros(count))
        checks = getattr(widget, '_checks', None)
        if checks is not None:
            checks.set_visible(False)

    def _remove_toggle_circles(self, widget):
        circles = getattr(widget, '_toggle_circles', None)
        if not circles:
            return
        for circle in circles:
            circle.remove()
        widget._toggle_circles = None

    def _update_toggle_circles(self, widget, statuses):
        circles = getattr(widget, '_toggle_circles', None)
        centers = getattr(widget, '_toggle_centers', None)
        if not circles or not centers:
            return
        bbox = widget.ax.get_window_extent()
        if bbox.height <= 0:
            aspect = 1.0
        else:
            aspect = bbox.width / bbox.height
        base_d = getattr(widget, '_toggle_circle_diameter', 0.06)
        width = base_d
        height = base_d * aspect
        for circle, (center_x, center_y), active in zip(circles, centers, statuses):
            circle.center = (center_x, center_y)
            circle.width = width
            circle.height = height
            circle.set_facecolor(self.toggle_circle_active if active else self.toggle_circle_inactive)
            circle.set_edgecolor(self.toggle_circle_active if active else self.border_color)
            circle.set_alpha(1.0 if active else 0.7)
            circle.set_linewidth(1.5 if active else 1.0)
        self._sync_toggle_hit_targets(widget, bbox)

    def _sync_toggle_hit_targets(self, widget, bbox=None):
        frames = getattr(widget, '_frames', None)
        centers = getattr(widget, '_toggle_centers', None)
        if frames is None or centers is None or len(centers) == 0:
            return
        centers_arr = np.array(centers)
        frames.set_offsets(centers_arr)
        if bbox is None:
            bbox = widget.ax.get_window_extent()
        fig = widget.ax.get_figure(root=True)
        dpi = getattr(fig, 'dpi', 72)
        diameter = getattr(widget, '_toggle_circle_diameter', 0.05)
        diameter_px = diameter * bbox.width
        diameter_pt = diameter_px * 72.0 / dpi
        size = max(35.0, (diameter_pt * 1.2) ** 2)
        frames.set_sizes(np.full(len(centers_arr), size))

    def _on_subtraction_toggle(self, label):
        if not hasattr(self, 'subtraction_checkbuttons'):
            return
        states = dict(zip(self.subtraction_toggle_labels, self.subtraction_checkbuttons.get_status()))
        if label == 'Spectro Subtract':
            self.spectrogram_subtract_enabled = states[label]
        elif label == 'Voice Subtract':
            self.voice_subtract_enabled = states[label]
        self._style_checkbuttons(self.subtraction_checkbuttons)

    def _on_noise_visibility_toggle(self, label):
        if not hasattr(self, 'noise_visibility_checkbuttons'):
            return
        states = dict(zip(self.noise_toggle_labels, self.noise_visibility_checkbuttons.get_status()))
        if label == 'Show Noise Spectro':
            self.show_noise_spectrogram = states[label]
        elif label == 'Show Noise Polar':
            self.show_noise_polar = states[label]
        self._style_checkbuttons(self.noise_visibility_checkbuttons)
        self._apply_layout_toggles()
    
    def _ensure_tuning_tooltip_support(self):
        if hasattr(self, 'tuning_hover_regions') and hasattr(self, 'tuning_tooltip_text'):
            return
        self.tuning_hover_regions = []
        self.tuning_tooltip_text = self.fig.text(
            0, 0, '', fontsize=self.label_fontsize, color=self.text_color, ha='left', va='bottom',
            alpha=0.9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=self.background_color, edgecolor=self.border_color, alpha=0.85),
            visible=False,
        )
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_tuning_hover_event)

    def _register_tuning_hover(self, artist, description):
        if not hasattr(self, 'tuning_hover_regions'):
            self._ensure_tuning_tooltip_support()
        self.tuning_hover_regions.append({'artist': artist, 'description': description})

    def _on_tuning_hover_event(self, event):
        if not hasattr(self, 'tuning_hover_regions') or self.tuning_hover_regions is None:
            return
        for target in self.tuning_hover_regions:
            artist = target['artist']
            if not artist.get_visible():
                continue
            contains, _ = artist.contains(event)
            if contains:
                self._show_tuning_tooltip(target['description'], event)
                return
        self._hide_tuning_tooltip()

    def _show_tuning_tooltip(self, text, event):
        if not hasattr(self, 'tuning_tooltip_text') or event.x is None or event.y is None:
            return
        fig_x, fig_y = self.fig.transFigure.inverted().transform((event.x, event.y))
        fig_x = min(max(fig_x + 0.005, 0.0), 0.98)
        fig_y = min(max(fig_y + 0.012, 0.0), 0.96)
        self.tuning_tooltip_text.set_position((fig_x, fig_y))
        self.tuning_tooltip_text.set_text(text)
        if not self.tuning_tooltip_text.get_visible():
            self.tuning_tooltip_text.set_visible(True)
        self.fig.canvas.draw_idle()

    def _hide_tuning_tooltip(self):
        if hasattr(self, 'tuning_tooltip_text') and self.tuning_tooltip_text.get_visible():
            self.tuning_tooltip_text.set_visible(False)
            self.fig.canvas.draw_idle()

    def _on_baseline_slider(self, value):
        self.bands_baseline_percentile = float(value)
        self.bands_noise_floor_history.clear()
        self.should_update_colors = True
    
    def _on_noise_history_slider(self, value):
        new_size = max(1, int(round(value)))
        self.bands_noise_history_size = new_size
        if len(self.bands_noise_floor_history) > new_size:
            self.bands_noise_floor_history = self.bands_noise_floor_history[-new_size:]
        self.should_update_colors = True
    
    def _on_offset_slider(self, value):
        self.bands_magnitude_offset = float(value)
        self.should_update_colors = True
    
    def _on_scale_slider(self, value):
        self.bands_magnitude_scale = max(float(value), 1e-3)
        self.should_update_colors = True
    
    def _on_smoothing_slider(self, value):
        self.bands_smoothing = float(value)
        self.should_update_colors = True
    
    def _on_waveform_fade_slider(self, value):
        new_value = np.clip(float(value), 0.0, 0.99)
        self.waveform_fade_decay = new_value
        if new_value <= 0:
            self.waveform_last_data = None
            self._decay_waveform_trails()

    def _on_voice_gain_slider(self, value):
        self.voice_amplification = max(0.1, float(value))
        self.prev_voice_bin_magnitudes = None
        self.should_update_colors = True
        self.voice_noise_history_frames.clear()

    def _on_voice_noise_history_slider(self, value):
        new_size = max(1, int(round(value)))
        if new_size == self.voice_noise_history_size:
            return
        self.voice_noise_history_size = new_size
        if len(self.voice_noise_history_frames) > new_size:
            self.voice_noise_history_frames = self.voice_noise_history_frames[-new_size:]
        self.prev_voice_bin_magnitudes = None

    def _on_voice_threshold_slider(self, value):
        self.voice_noise_threshold = max(0.0, float(value))

    def _on_voice_max_freq_slider(self, value):
        new_max = np.clip(float(value), 500.0, self.bands_max_freq)
        if np.isclose(new_max, self.voice_max_freq):
            return
        self.voice_max_freq = new_max
        self._configure_voice_bins()
        self.voice_noise_history_frames.clear()
        self.prev_voice_bin_magnitudes = None
        self.prev_noise_levels = None
        self.should_update_colors = True
        self._update_polar_frequency_labels()

    def _on_spectrogram_subtract_aggressiveness_slider(self, value):
        self.spectrogram_subtract_aggressiveness = max(0.0, float(value))

    def _on_voice_subtract_aggressiveness_slider(self, value):
        self.voice_subtract_aggressiveness = max(0.0, float(value))

    def _on_spectrogram_bins_slider(self, value):
        new_min = max(1, int(round(value)))
        if new_min == self.spectrogram_min_view_bins:
            return
        self.spectrogram_min_view_bins = new_min
        self._update_spectrogram_image()

    def _on_spectrogram_range_slider(self, values):
        min_val, max_val = values
        min_val = max(0.0, min_val)
        max_val = min(self.spectrogram_max_freq, max_val)
        if max_val - min_val < 50:
            midpoint = (min_val + max_val) / 2
            half_span = 25
            min_val = max(0.0, midpoint - half_span)
            max_val = min(self.spectrogram_max_freq, midpoint + half_span)
            self.spectrogram_range_slider.set_val((min_val, max_val))
        self._update_spectrogram_view(min_val, max_val, refresh_ticks=True)
    
    @line_profiler.profile
    def update_waveform(self):
        buffer_len = len(self.audio_buffer)
        self._decay_waveform_trails()
        if self.waveform_len != buffer_len or self.waveform_x is None:
            self.waveform_x = np.arange(buffer_len)
            self.waveform_len = buffer_len
            self.ax_wave.set_xlim(0, buffer_len)
        self.line_wave.set_data(self.waveform_x, self.audio_buffer)
        self._add_waveform_trail(self.waveform_last_data)
        self.waveform_last_data = self.audio_buffer.copy()
        
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
            needs_update = (self.prev_ylim is None) or (abs(self.prev_ylim - ylim) > self.waveform_ylim_epsilon)
            self.prev_ylim = ylim
            if needs_update:
                self.ax_wave.set_ylim(-ylim, ylim)
                self.scale_text.set_text(f'±{ylim:.3f}')
    
    def _normalize_magnitude(self, data, noise_floor_percentile):
        """Remove noise floor and normalize magnitude data to 0-1 range."""
        noise_floor = self._fast_percentile(data, noise_floor_percentile)
        data_clean = np.maximum(data - noise_floor, 0)
        data_normalized = (data_clean - np.min(data_clean)) / (np.max(data_clean) - np.min(data_clean) + 1e-10)
        return np.clip(data_normalized, 0, 1)
    
    def _setup_mel_filterbank(self):
        mel_low = 2595 * np.log10(1 + self.mel_freq_low / 700)
        mel_high = 2595 * np.log10(1 + self.mel_freq_high / 700)
        mel_points = np.linspace(mel_low, mel_high, self.mel_banks + 2)
        self.mel_hz_points = 700 * (10**(mel_points / 2595) - 1)
        self.mel_bin_centers = self.mel_hz_points[1:-1]

    def _update_spectrogram_view(self, min_val, max_val, refresh_ticks=False):
        min_val = max(0.0, min(min_val, self.spectrogram_max_freq))
        max_val = max(min_val + 1.0, min(max_val, self.spectrogram_max_freq))
        if max_val - min_val < 50:
            midpoint = (min_val + max_val) / 2
            min_val = max(0.0, midpoint - 25)
            max_val = min(self.spectrogram_max_freq, midpoint + 25)
        centers = self.mel_bin_centers
        start_idx = np.searchsorted(centers, min_val, side='left')
        end_idx = np.searchsorted(centers, max_val, side='right')
        start_idx = max(0, start_idx - 1)
        end_idx = min(len(centers), end_idx + 1)
        if start_idx >= end_idx:
            start_idx = max(0, start_idx - 1)
            end_idx = min(len(centers), start_idx + 2)
        indices = np.arange(start_idx, end_idx)
        if indices.size == 0:
            indices = np.arange(len(centers))
        self.spectrogram_view_indices = indices
        self.spectrogram_view_min = min_val
        self.spectrogram_view_max = max_val
        self._update_spectrogram_image(refresh_ticks)

    def _update_spectrogram_image(self, refresh_ticks=False):
        view_data, _ = self._get_spectrogram_view_data()
        noise_view, _ = self._get_spectrogram_view_data(self.noise_spectrogram_full_data)
        extent = [self.spectrogram_view_min, self.spectrogram_view_max, 0, 100]
        if self.spectrogram_img is not None:
            self.spectrogram_img.set_data(view_data)
            self.spectrogram_img.set_extent(extent)
            self.ax_spectrogram.set_xlim(self.spectrogram_view_min, self.spectrogram_view_max)
        if self.noise_spectrogram_img is not None:
            self.noise_spectrogram_img.set_data(noise_view)
            self.noise_spectrogram_img.set_extent(extent)
            self.ax_noise_spectrogram.set_xlim(self.spectrogram_view_min, self.spectrogram_view_max)
        if refresh_ticks:
            labeled_ticks = [100, 200, 500, 1000, 2000, 4000, 8000]
            ticks = [tick for tick in labeled_ticks if self.spectrogram_view_min <= tick <= self.spectrogram_view_max]
            if len(ticks) < 2:
                ticks = np.linspace(self.spectrogram_view_min, self.spectrogram_view_max, 4)
            for axis in (self.ax_spectrogram, self.ax_noise_spectrogram):
                axis.set_xticks(ticks)
            self.fig.canvas.draw_idle()

    def _get_spectrogram_view_data(self, data_source=None):
        data = self.spectrogram_full_data if data_source is None else data_source
        indices = self.spectrogram_view_indices
        if indices.size == 0:
            indices = np.arange(len(self.mel_bin_centers))
        view_centers = self.mel_bin_centers[indices]
        view_data = data[:, indices]
        current_bins = view_data.shape[1]
        if current_bins == 0:
            target_bins = max(2, self.spectrogram_min_view_bins)
            target_centers = np.linspace(self.spectrogram_view_min, self.spectrogram_view_max, target_bins)
            return np.zeros((data.shape[0], target_bins)), target_centers
        span = max(1.0, self.spectrogram_view_max - self.spectrogram_view_min)
        bins_from_span = int(round(span * self.mel_banks / self.spectrogram_max_freq))
        target_bins = max(self.spectrogram_min_view_bins, bins_from_span)
        target_bins = min(max(target_bins, 2), max(self.spectrogram_min_view_bins, len(self.mel_bin_centers)))
        target_centers = np.linspace(self.spectrogram_view_min, self.spectrogram_view_max, target_bins)
        if current_bins == 1:
            tiled = np.repeat(view_data, target_bins, axis=1)
            return tiled, target_centers

        base_x = view_centers
        if base_x.size < 2:
            tiled = np.repeat(view_data, target_bins, axis=1)
            return tiled, target_centers

        below_mask = target_centers <= base_x[0]
        above_mask = target_centers >= base_x[-1]
        interior_mask = ~(below_mask | above_mask)
        interpolated = np.empty((view_data.shape[0], target_bins))

        if np.any(interior_mask):
            interior_centers = target_centers[interior_mask]
            interior_indices = np.searchsorted(base_x, interior_centers, side='right') - 1
            interior_indices = np.clip(interior_indices, 0, base_x.size - 2)
            left = base_x[interior_indices]
            right = base_x[interior_indices + 1]
            denom = np.maximum(right - left, 1e-9)
            weights = (interior_centers - left) / denom
            left_vals = view_data[:, interior_indices]
            right_vals = view_data[:, interior_indices + 1]
            interpolated[:, interior_mask] = left_vals + (right_vals - left_vals) * weights

        if np.any(below_mask):
            interpolated[:, below_mask] = view_data[:, 0][:, None]
        if np.any(above_mask):
            interpolated[:, above_mask] = view_data[:, -1][:, None]

        return interpolated, target_centers

    def _configure_frequency_bins(self, bin_count=None):
        target_bins = int(round(bin_count if bin_count is not None else self.num_bins))
        target_bins = max(1, target_bins)
        edges = np.linspace(0, self.bands_max_freq, target_bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2
        widths = np.diff(edges)
        self.frequency_bin_edges = edges
        self.frequency_bin_centers = centers
        self.bar_left = centers - widths / 2
        self.bar_right = centers + widths / 2
        self.bar_widths = widths
        self.num_bins = target_bins
        self.prev_bin_magnitudes = None
        self.bars_vertices = None
        self.bars_collection = None
        self.bars_paths = None
        self.bars_colors = None
        self.bars_heights = None
        self.should_update_colors = True
        if hasattr(self, 'ax_bars'):
            self.ax_bars.set_xlim(0, self.bands_max_freq)

    def _configure_voice_bins(self):
        voice_freq_edges = np.linspace(0, self.voice_max_freq, self.voice_num_bins + 1)
        self.voice_bin_centers = (voice_freq_edges[:-1] + voice_freq_edges[1:]) / 2

    def _format_frequency_label(self, value):
        if value >= 1000:
            scaled = value / 1000.0
            text = f'{scaled:.1f}'.rstrip('0').rstrip('.')
            return f'{text}k'
        return f'{int(round(value))}'

    def _update_polar_frequency_labels(self):
        ticks = np.linspace(0, 2 * np.pi, 4, endpoint=False)
        label_freqs = np.linspace(self.voice_max_freq / 4, self.voice_max_freq, len(ticks))
        labels = [self._format_frequency_label(freq) for freq in label_freqs]
        for axis in (getattr(self, 'ax_voice_bars', None), getattr(self, 'ax_noise_bars', None)):
            if axis is None:
                continue
            axis.set_xticks(ticks)
            axis.set_xticklabels(labels)
            for label in axis.get_xticklabels():
                label.set_alpha(self.label_alpha)

    def _color_change_mask(self, cached_colors, new_colors):
        if cached_colors is None or len(cached_colors) != len(new_colors):
            return np.ones(len(new_colors), dtype=bool), new_colors.copy()
        diffs = np.max(np.abs(new_colors - cached_colors), axis=1)
        change_mask = diffs > self.color_change_epsilon
        if np.any(change_mask):
            cached_colors = cached_colors.copy()
            cached_colors[change_mask] = new_colors[change_mask]
        return change_mask, cached_colors

    def _height_change_mask(self, cached_heights, new_heights):
        if cached_heights is None or len(cached_heights) != len(new_heights):
            return np.ones(len(new_heights), dtype=bool), new_heights.copy()
        diffs = np.abs(new_heights - cached_heights)
        change_mask = diffs > self.height_change_epsilon
        if np.any(change_mask):
            cached_heights[change_mask] = new_heights[change_mask]
        return change_mask, cached_heights

    def _fast_percentile(self, data, percentile):
        if data.size == 0:
            return 0.0
        percentile = np.clip(percentile, 0, 100)
        index = int(round((percentile / 100.0) * (data.size - 1)))
        partitioned = np.partition(data, index)
        return partitioned[index]

    def _percentile_along_axis(self, data, percentile, axis=0):
        arr = np.asarray(data)
        if arr.size == 0 or arr.shape[axis] == 0:
            out_shape = arr.shape[:axis] + arr.shape[axis + 1:]
            return np.zeros(out_shape, dtype=arr.dtype if arr.size else float)
        percentile = np.clip(percentile, 0, 100)
        axis_len = arr.shape[axis]
        index = int(round((percentile / 100.0) * (axis_len - 1)))
        partitioned = np.partition(arr, index, axis=axis)
        return np.take(partitioned, index, axis=axis)

    def _compute_spectrogram_noise_frame(self, mel_normalized):
        frame = np.asarray(mel_normalized, dtype=float)
        self.spectrogram_noise_history_frames.append(frame.copy())
        if len(self.spectrogram_noise_history_frames) > self.spectrogram_noise_history_size:
            self.spectrogram_noise_history_frames.pop(0)
        if len(self.spectrogram_noise_history_frames) >= 5:
            history_matrix = np.vstack(self.spectrogram_noise_history_frames)
            noise_frame = self._percentile_along_axis(history_matrix, 25, axis=0)
        else:
            percentile_value = self._fast_percentile(frame, 10)
            noise_frame = np.full_like(frame, percentile_value)
        return np.minimum(noise_frame, frame)

    def _ensure_mel_bins(self, freqs):
        freq_len = len(freqs)
        if self.mel_freq_len == freq_len and self.mel_bin_starts is not None:
            return
        starts = np.searchsorted(freqs, self.mel_hz_points[:-2], side='left')
        ends = np.searchsorted(freqs, self.mel_hz_points[2:], side='left')
        self.mel_bin_starts = starts
        self.mel_bin_ends = ends
        self.mel_freq_len = freq_len

    def _update_bar_vertices(self, heights):
        if self.bars_vertices is None:
            count = len(heights)
            self.bars_vertices = np.zeros((count, 4, 2))
            self.bars_vertices[:, 0, 0] = self.bar_left
            self.bars_vertices[:, 1, 0] = self.bar_left
            self.bars_vertices[:, 2, 0] = self.bar_right
            self.bars_vertices[:, 3, 0] = self.bar_right
        self.bars_vertices[:, 0, 1] = 0
        self.bars_vertices[:, 3, 1] = 0
        self.bars_vertices[:, 1, 1] = heights
        self.bars_vertices[:, 2, 1] = heights
        return self.bars_vertices
    
    def _update_voice_bar_vertices(self, heights):
        if self.voice_vertices is None:
            count = len(heights)
            self.voice_vertices = np.zeros((count, 4, 2))
            self.voice_vertices[:, 0, 0] = self.voice_theta_left
            self.voice_vertices[:, 1, 0] = self.voice_theta_left
            self.voice_vertices[:, 2, 0] = self.voice_theta_right
            self.voice_vertices[:, 3, 0] = self.voice_theta_right
        self.voice_vertices[:, 0, 1] = self.voice_radius_base
        self.voice_vertices[:, 3, 1] = self.voice_radius_base
        self.voice_vertices[:, 1, 1] = heights
        self.voice_vertices[:, 2, 1] = heights
        return self.voice_vertices

    def _update_noise_bar_vertices(self, heights):
        if self.noise_vertices is None:
            count = len(heights)
            self.noise_vertices = np.zeros((count, 4, 2))
            self.noise_vertices[:, 0, 0] = self.voice_theta_left
            self.noise_vertices[:, 1, 0] = self.voice_theta_left
            self.noise_vertices[:, 2, 0] = self.voice_theta_right
            self.noise_vertices[:, 3, 0] = self.voice_theta_right
        self.noise_vertices[:, 0, 1] = self.noise_radius_base
        self.noise_vertices[:, 3, 1] = self.noise_radius_base
        self.noise_vertices[:, 1, 1] = heights
        self.noise_vertices[:, 2, 1] = heights
        return self.noise_vertices
    
    @line_profiler.profile
    def update_spectrogram(self, freqs, magnitude_db):
        self._ensure_mel_bins(freqs)
        prefix = np.concatenate(([0.0], np.cumsum(magnitude_db)))
        mel_magnitude = np.full(self.mel_banks, -100.0)
        for idx in range(self.mel_banks):
            start = self.mel_bin_starts[idx]
            end = self.mel_bin_ends[idx]
            if end > start:
                mel_sum = prefix[end] - prefix[start]
                mel_magnitude[idx] = mel_sum / (end - start)
        mel_normalized = self._normalize_magnitude(mel_magnitude, self.spectrogram_noise_floor)
        mel_normalized = np.power(mel_normalized, self.spectrogram_power)
        
        noise_frame = self._compute_spectrogram_noise_frame(mel_normalized)
        if self.spectrogram_subtract_enabled:
            adjusted_noise = noise_frame * self.spectrogram_subtract_aggressiveness
            signal_frame = np.maximum(mel_normalized - adjusted_noise, 0)
        else:
            signal_frame = mel_normalized.copy()
        self.spectrogram_full_data = np.roll(self.spectrogram_full_data, 1, axis=0)
        self.spectrogram_full_data[0, :] = signal_frame
        self.noise_spectrogram_full_data = np.roll(self.noise_spectrogram_full_data, 1, axis=0)
        self.noise_spectrogram_full_data[0, :] = noise_frame
        self._update_spectrogram_image()
        self.spectrogram_img.set_clim(0, 1)
        if self.noise_spectrogram_img is not None:
            self.noise_spectrogram_img.set_clim(0, 1)
    
    @line_profiler.profile
    def update_voice_frequency_bands(self, visible_freqs, visible_magnitude):
        # Get voice frequency range
        voice_mask = visible_freqs <= self.voice_max_freq
        voice_freqs = visible_freqs[voice_mask]
        voice_magnitude = visible_magnitude[voice_mask]
        
        # Interpolate magnitude at bin centers for noise tracking
        bin_inputs = np.interp(self.voice_bin_centers, voice_freqs, voice_magnitude)
        self.voice_noise_history_frames.append(bin_inputs)
        if len(self.voice_noise_history_frames) > self.voice_noise_history_size:
            self.voice_noise_history_frames.pop(0)

        if len(self.voice_noise_history_frames) >= 5:
            history_matrix = np.vstack(self.voice_noise_history_frames)
            per_bin_noise = self._percentile_along_axis(history_matrix, 25, axis=0)
        else:
            percentile_value = self._fast_percentile(bin_inputs, 10)
            per_bin_noise = np.full_like(bin_inputs, percentile_value)

        per_bin_noise = np.minimum(per_bin_noise, bin_inputs)
        self._update_noise_floor_display(per_bin_noise)
        aggressiveness = 1.0 / max(self.voice_subtract_aggressiveness, 1e-3)
        if self.voice_subtract_enabled:
            baseline = np.minimum(per_bin_noise * aggressiveness, bin_inputs)
            signal_above_noise = bin_inputs - baseline - self.voice_noise_threshold
        else:
            shifted = bin_inputs - np.min(bin_inputs)
            signal_above_noise = shifted
        bin_magnitudes = np.maximum(signal_above_noise, 0) * self.voice_amplification
        
        # Smooth the data
        if self.prev_voice_bin_magnitudes is not None:
            bin_magnitudes = self.bands_smoothing * self.prev_voice_bin_magnitudes + (1 - self.bands_smoothing) * bin_magnitudes
        self.prev_voice_bin_magnitudes = bin_magnitudes
        
        # Normalize for colors
        normalized = np.clip(bin_magnitudes / 60, 0, 1)
        colors = self.bands_colormap(normalized)
        display_heights = np.clip(self.voice_radius_base + bin_magnitudes,
                                  self.voice_radius_base, self.voice_radius_max)
        
        # Update polar bar chart (avoid color updates when unchanged)
        update_colors = self.should_update_colors or self.voice_bar_colors is None
        if self.voice_collection is None:
            self.voice_bar_heights = display_heights.copy()
            verts = self._update_voice_bar_vertices(self.voice_bar_heights)
            self.voice_bar_colors = colors.copy()
            self.voice_collection = PolyCollection(verts, facecolors=self.voice_bar_colors,
                                                   edgecolors='none', alpha=0.8, closed=True)
            self.ax_voice_bars.add_collection(self.voice_collection)
            self.voice_paths = self.voice_collection.get_paths()
        else:
            height_changes, self.voice_bar_heights = self._height_change_mask(self.voice_bar_heights, display_heights)
            if np.any(height_changes):
                changed = np.flatnonzero(height_changes)
                for idx in changed:
                    height = self.voice_bar_heights[idx]
                    vertices = self.voice_paths[idx].vertices
                    vertices[1, 1] = height
                    vertices[2, 1] = height
                if self.voice_vertices is not None:
                    self.voice_vertices[changed, 1, 1] = self.voice_bar_heights[changed]
                    self.voice_vertices[changed, 2, 1] = self.voice_bar_heights[changed]
            if update_colors:
                color_changes, self.voice_bar_colors = self._color_change_mask(self.voice_bar_colors, colors)
                if np.any(color_changes):
                    self.voice_collection.set_facecolor(self.voice_bar_colors)

    def _update_noise_floor_display(self, noise_levels):
        if noise_levels is None or noise_levels.size == 0 or not hasattr(self, 'ax_noise_bars'):
            return
        levels = np.asarray(noise_levels, dtype=float)
        finite_mask = np.isfinite(levels)
        if not np.any(finite_mask):
            return
        finite_values = levels[finite_mask]
        min_val = finite_values.min()
        max_val = finite_values.max()
        span = max(max_val - min_val, 1.0)
        safe_levels = np.where(finite_mask, levels, min_val)
        normalized = np.clip((safe_levels - min_val) / span, 0, 1)
        display_heights = self.noise_radius_base + normalized * (self.noise_radius_max - self.noise_radius_base)
        if self.prev_noise_levels is not None:
            display_heights = self.bands_smoothing * self.prev_noise_levels + (1 - self.bands_smoothing) * display_heights
        self.prev_noise_levels = display_heights
        update_colors = self.should_update_colors or self.noise_bar_colors is None
        colors = self.bands_colormap(normalized) if update_colors else None
        if self.noise_collection is None:
            self.noise_bar_heights = display_heights.copy()
            verts = self._update_noise_bar_vertices(self.noise_bar_heights)
            self.noise_bar_colors = colors.copy() if colors is not None else self.bands_colormap(normalized).copy()
            self.noise_collection = PolyCollection(verts, facecolors=self.noise_bar_colors,
                                                   edgecolors='none', alpha=0.65, closed=True)
            self.ax_noise_bars.add_collection(self.noise_collection)
            self.noise_paths = self.noise_collection.get_paths()
        else:
            height_changes, self.noise_bar_heights = self._height_change_mask(self.noise_bar_heights, display_heights)
            if np.any(height_changes):
                changed = np.flatnonzero(height_changes)
                for idx in changed:
                    height = self.noise_bar_heights[idx]
                    vertices = self.noise_paths[idx].vertices
                    vertices[1, 1] = height
                    vertices[2, 1] = height
                if self.noise_vertices is not None:
                    self.noise_vertices[changed, 1, 1] = self.noise_bar_heights[changed]
                    self.noise_vertices[changed, 2, 1] = self.noise_bar_heights[changed]
            if update_colors:
                color_changes, self.noise_bar_colors = self._color_change_mask(self.noise_bar_colors, colors)
                if np.any(color_changes):
                    self.noise_collection.set_facecolor(self.noise_bar_colors)
    
    @line_profiler.profile
    def update_frequency_bands(self, visible_freqs, visible_magnitude):
        # Divide frequency range into bins and find peak magnitude in each
        bin_centers = self.frequency_bin_centers
        
        # Interpolate magnitude values at bin centers for smooth visualization
        bin_magnitudes = np.interp(bin_centers, visible_freqs, visible_magnitude) + self.bands_magnitude_offset
        
        # Remove baseline noise and apply smoothing
        if self.bands_baseline_percentile > 0:
            current_floor = self._fast_percentile(bin_magnitudes, self.bands_baseline_percentile)
            self.bands_noise_floor_history.append(current_floor)
            if len(self.bands_noise_floor_history) > self.bands_noise_history_size:
                self.bands_noise_floor_history.pop(0)
            baseline = np.mean(self.bands_noise_floor_history)
            bin_magnitudes = np.maximum(bin_magnitudes - baseline, 0)
        
        if self.prev_bin_magnitudes is not None:
            bin_magnitudes = self.bands_smoothing * self.prev_bin_magnitudes + (1 - self.bands_smoothing) * bin_magnitudes
        self.prev_bin_magnitudes = bin_magnitudes
        
        normalized = np.clip(bin_magnitudes / self.bands_magnitude_scale, 0, 1)
        colors = self.bands_colormap(normalized)
        
        update_colors = self.should_update_colors or self.bars_colors is None
        if self.bars_collection is None:
            self.bars_heights = bin_magnitudes.copy()
            verts = self._update_bar_vertices(self.bars_heights)
            self.bars_colors = colors.copy()
            self.bars_collection = PolyCollection(verts, facecolors=self.bars_colors, edgecolors='none')
            self.ax_bars.add_collection(self.bars_collection)
            self.bars_paths = self.bars_collection.get_paths()
        else:
            height_changes, self.bars_heights = self._height_change_mask(self.bars_heights, bin_magnitudes)
            if np.any(height_changes):
                changed = np.flatnonzero(height_changes)
                for idx in changed:
                    height = self.bars_heights[idx]
                    vertices = self.bars_paths[idx].vertices
                    vertices[1, 1] = height
                    vertices[2, 1] = height
                if self.bars_vertices is not None:
                    self.bars_vertices[changed, 1, 1] = self.bars_heights[changed]
                    self.bars_vertices[changed, 2, 1] = self.bars_heights[changed]
            if update_colors:
                color_changes, self.bars_colors = self._color_change_mask(self.bars_colors, colors)
                if np.any(color_changes):
                    self.bars_collection.set_facecolor(self.bars_colors)
    
    @line_profiler.profile
    def update(self, _frame):
        frame_start = time.perf_counter()
        
        if self.audio_buffer is None:
            return []
        
        self.should_update_colors = (self.color_frame_counter == 0)
        self.color_frame_counter = (self.color_frame_counter + 1) % self.color_update_interval
        
        buffer_len = len(self.audio_buffer)
        if self.window_size != buffer_len:
            self.window = np.hanning(buffer_len)
            self.window_size = buffer_len
        windowed = self.audio_buffer * self.window
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
    def run(self, device_id, run_for=None):
        stop_timer = None
        duration = run_for if run_for and run_for > 0 else None
        if duration:
            stop_timer = self.fig.canvas.new_timer(interval=int(duration * 1000))
            stop_timer.single_shot = True
            stop_timer.add_callback(plt.close, self.fig)
            stop_timer.start()
        try:
            with sd.InputStream(device=device_id, callback=self.audio_callback, channels=1, samplerate=self.sample_rate):
                manager = plt.get_current_fig_manager()
                manager.full_screen_toggle()
                self.animation = FuncAnimation(self.fig, self.update, interval=self.update_interval_ms,
                                               blit=False, cache_frame_data=True)
                plt.show()
        finally:
            if stop_timer:
                stop_timer.stop()


def parse_args():
    parser = argparse.ArgumentParser(description="Audio Visualizer")
    parser.add_argument("--run-for", type=float, default=None, metavar="SECONDS",
                        help="Automatically stop after SECONDS (useful for profiling).")
    return parser.parse_args()


def main():
    args = parse_args()
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
        visualizer.run(device_id, run_for=args.run_for)
    except KeyboardInterrupt:
        print("\nStopped audio capture")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you have a working audio input device.")
        print("You may need to grant microphone permissions to your terminal.")


if __name__ == "__main__":
    main()
