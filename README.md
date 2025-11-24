# Noisy – Live Audio Visualizer

A project to play around with audio processing and agentic coding. A mix of Claude Sonnet 4.5 and GPT 5.1-codex. As such, likely both incomplete and incorrect. But looks kind of neat.

Minimalist realtime visualizer built with Python, matplotlib, numpy, scipy, and sounddevice. It listens to the default input device, filters and analyzes the stream, then renders three synchronized views with adaptive noise suppression.

<img width="1483" height="928" alt="Screenshot 2025-11-24 at 12 56 19 AM" src="https://github.com/user-attachments/assets/abb8e5c0-e8cd-441f-b031-a32fc0a01c75" />

## Views

- **Mel Spectrogram (top)** – 40 mel banks spanning 80–8000 Hz. Frames are de-noised with a 30th-percentile noise floor, raised to power 1.5 for contrast, and can be zoomed with a range slider that re-samples bins so the zoomed view always stays smooth.
- **Waveform (middle)** – Raw signal after an 80 Hz high-pass filter. The vertical scale auto-adjusts with a smoothed ±zoom indicator, and recent frames linger as translucent trails whose fade speed is slider-controlled.
- **Frequency Panels (bottom)**
	- **Voice Polar Chart (left)** – 30 bins from 0–2000 Hz arranged radially. Magnitudes are interpolated at the bin centers, appended to a 45-frame per-bin history, and once five frames exist the 25th-percentile baseline is subtracted (earlier frames fall back to a frame-level 10th percentile). Anything more than 8 dB above that floor is multiplied by the voice gain slider (default ×1.6) and rendered between radius 15–60.
	- **Noise Polar Chart (bottom-right)** – Shares the same bins as the voice chart but displays the tracked per-bin baseline directly. Baselines are normalized within the current frame, smoothed, and drawn between radius 10–45 with the same purple palette so you can watch ambient noise patterns wrap around the circle in real time.
	- **Full-Range Bars (right)** – 50 rectangular bins from 0–8000 Hz. Magnitudes get a 60 dB offset, subtract the rolling 35th-percentile baseline (4-frame history), and smooth with an EMA before recoloring.

## Signal Processing

- 80 Hz Butterworth high-pass filter applied before analysis.
- Windowed FFT (Hann) with exponential smoothing on magnitudes to avoid flicker.
- Custom mel filterbank caches to reduce per-frame cost.
- Adaptive percentile-based noise suppression for both voice and bar charts.
- Per-bin ambient noise floor visualization mirrors the voice bins so you can see filtered noise separate from speech energy.

## Tuning Controls

On the right edge, a toggleable panel exposes live sliders:

- Baseline %, Noise Frames, Offset (dB), Scale (dB), Smoothing, Wave Fade (trail decay).
- Voice controls: Voice Gain, Voice Noise Frames (history length), Voice Threshold (dB gap before drawing bars).
- Spectro Bins and a Spectro Hz RangeSlider for spectrogram density/zoom.
- Hovering any label pops a minimalist tooltip describing what that control does, so you can tune by feel without referencing docs.

## Running It

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
python main.py
```

Grant microphone access when prompted. Use `LINE_PROFILE=1 python main.py --run-for 30` to profile short sessions.
