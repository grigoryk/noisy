#!/usr/bin/env python3

import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class AudioVisualizer:
    def __init__(self, sample_rate, num_bins=100):
        self.sample_rate = sample_rate
        self.num_bins = num_bins
        self.audio_buffer = None
        
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.line, = self.ax.plot([], [], lw=2)
        
        self.ax.set_xlim(0, sample_rate // 2)
        self.ax.set_ylim(0, 1)
        self.ax.set_xlabel('Frequency (Hz)')
        self.ax.set_ylabel('Magnitude')
        self.ax.set_title('Real-time FFT')
    
    def update(self, frame):
        if self.audio_buffer is None:
            return self.line,
        
        fft = np.fft.rfft(self.audio_buffer)
        magnitude = np.abs(fft)
        magnitude = magnitude / np.max(magnitude) if np.max(magnitude) > 0 else magnitude
        
        freqs = np.fft.rfftfreq(len(self.audio_buffer), 1/self.sample_rate)
        
        self.line.set_data(freqs, magnitude)
        return self.line,
    
    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Status: {status}")
        self.audio_buffer = indata[:, 0].copy()
    
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
