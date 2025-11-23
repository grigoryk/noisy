#!/usr/bin/env python3

import numpy as np
import sounddevice as sd


def analyze(audio_data, sample_rate):
    audio_bytes = audio_data.tobytes()
    print(f"Received {len(audio_bytes)} bytes | Shape: {audio_data.shape} | Sample rate: {sample_rate} Hz")
    print(f"First 32 bytes: {audio_bytes[:32].hex()}")


def audio_callback(indata, frames, time, status):
    if status:
        print(f"Status: {status}")
    
    analyze(indata.copy(), frames)


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
        print("Press Ctrl+C to stop")
        print("="*60 + "\n")
        
        with sd.InputStream(device=device_id, callback=audio_callback, channels=1, samplerate=sample_rate):
            sd.sleep(-1)
    except KeyboardInterrupt:
        print("\nStopped audio capture")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you have a working audio input device.")
        print("You may need to grant microphone permissions to your terminal.")


if __name__ == "__main__":
    main()
