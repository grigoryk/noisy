#!/usr/bin/env python3

import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import butter, sosfilt


class AudioVisualizer:
    def __init__(self, sample_rate, num_bins=100, highpass_freq=80, smoothing=0.5):
        self.sample_rate = sample_rate
        self.num_bins = num_bins
        self.audio_buffer = None
        self.highpass_freq = highpass_freq
        self.smoothing = smoothing
        self.prev_magnitude = None
        
        self.sos = butter(4, highpass_freq, btype='high', fs=sample_rate, output='sos')
        
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.line, = self.ax.plot([], [], lw=2)
        
        self.ax.set_xlim(0, 5000)
        self.ax.set_xlabel('Frequency (Hz)')
        self.ax.set_ylabel('Magnitude (dB)')
        self.ax.set_title('Real-time FFT')
    
    def update(self, frame):
        if self.audio_buffer is None:
            return self.line,
        
        windowed = self.audio_buffer * np.hanning(len(self.audio_buffer))
        fft = np.fft.rfft(windowed)
        magnitude = np.abs(fft)
        
        magnitude_db = 20 * np.log10(magnitude + 1e-10)
        
        if self.prev_magnitude is not None:
            magnitude_db = self.smoothing * self.prev_magnitude + (1 - self.smoothing) * magnitude_db
        
        self.prev_magnitude = magnitude_db
        
        freqs = np.fft.rfftfreq(len(self.audio_buffer), 1/self.sample_rate)
        
        visible_mask = freqs <= 2000
        visible_magnitude = magnitude_db[visible_mask]
        
        self.line.set_data(freqs, magnitude_db)
        
        y_min, y_max = np.min(visible_magnitude), np.max(visible_magnitude)
        y_range = y_max - y_min
        self.ax.set_ylim(y_min - y_range * 0.3, y_max + y_range * 0.3)
        
        return self.line,
    
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
