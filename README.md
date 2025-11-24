# Noisy – Live Audio Visualizer

A project to play around with audio processing and agentic coding. A mix of Claude Sonnet 4.5 and GPT 5.1-codex. As such, likely both incomplete and incorrect. But looks kind of neat.

Minimalist realtime visualizer built with Python, matplotlib, numpy, scipy, and sounddevice. It listens to the default input device, filters and analyzes the stream, then renders three synchronized views with adaptive noise suppression.

<img width="1483" height="928" alt="Screenshot 2025-11-24 at 12 56 19 AM" src="https://github.com/user-attachments/assets/abb8e5c0-e8cd-441f-b031-a32fc0a01c75" />

## Views

- **Mel Spectrogram (top)** – 40 mel banks spanning 80–8000 Hz. Frames are de-noised with a 30th-percentile noise floor, raised to power 1.5 for contrast, and can be zoomed with a range slider that re-samples bins so the zoomed view always stays smooth.
- **Waveform (middle)** – Raw signal after an 80 Hz high-pass filter. The vertical scale auto-adjusts with a smoothed ±zoom indicator, and recent frames linger as translucent trails whose fade speed is slider-controlled.
- **Frequency Panels (bottom)**
	- **Voice Polar Chart (left)** – 30 bins from 0–2000 Hz arranged radially. Each frame estimates the 10th-percentile noise floor (history size 1), amplifies by a tunable voice gain (default ×1.6), subtracts an 8 dB threshold, and smooths before rendering bars between radius 15–60.
	- **Full-Range Bars (right)** – 50 rectangular bins from 0–8000 Hz. Magnitudes get a 60 dB offset, subtract the rolling 35th-percentile baseline (4-frame history), and smooth with an EMA before recoloring.

## Signal Processing

- 80 Hz Butterworth high-pass filter applied before analysis.
- Windowed FFT (Hann) with exponential smoothing on magnitudes to avoid flicker.
- Custom mel filterbank caches to reduce per-frame cost.
- Adaptive percentile-based noise suppression for both voice and bar charts.

## Tuning Controls

On the right edge, a toggleable panel exposes live sliders:

- Baseline %, Noise Frames, Offset (dB), Scale (dB), Smoothing, Wave Fade (trail decay).
- Voice Gain for polar amplification.
- Spectro Bins and a Spectro Hz RangeSlider for spectrogram density/zoom.

## Running It

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
python main.py
```

Grant microphone access when prompted. Use `LINE_PROFILE=1 python main.py --run-for 30` to profile short sessions.
