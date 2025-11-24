# Noisy – Live Audio Visualizer

A project to play around with audio processing and agentic coding. A mix of Claude Sonnet 4.5 and GPT 5.1-codex. As such, likely both incomplete and incorrect. But looks kind of neat.

Minimalist realtime visualizer built with Python, matplotlib, numpy, scipy, and sounddevice. It listens to the default input device, filters and analyzes the stream, then renders three synchronized views with adaptive noise suppression.

<img width="1483" height="928" alt="Screenshot 2025-11-24 at 12 56 19 AM" src="https://github.com/user-attachments/assets/abb8e5c0-e8cd-441f-b031-a32fc0a01c75" />

## Views

- **Mel Spectrograms (top row)** – The left pane shows the signal spectrogram after subtracting a tracked 30th-percentile noise floor and raising it to the 1.5 power for contrast. The right pane mirrors the tracked noise spectrogram so you can see the ambient floor in real time. Both panels honor the range slider that re-samples mel bins, and the noise view can be hidden via the **Show Noise Spectro** toggle to let the main spectrogram span the full width.
- **Waveform (middle)** – Raw signal after an 80 Hz high-pass filter. The vertical scale auto-adjusts with a smoothed ±zoom indicator, and recent frames linger as translucent trails whose fade speed is slider-controlled.
- **Frequency Panels (bottom)**
	- **Voice Polar Chart (left)** – 30 bins from 0–2000 Hz arranged radially. Magnitudes are interpolated at the bin centers, appended to a 45-frame per-bin history, and once five frames exist the 25th-percentile baseline is subtracted (earlier frames fall back to a frame-level 10th percentile). Anything more than 8 dB above that floor is multiplied by the voice gain slider (default ×1.6) and rendered between radius 15–60. A **Voice Max Hz** slider lets you raise the ceiling (up to 8 kHz) or lower it to focus only on fundamentals, and the polar tick labels follow the new range.
	- **Noise Polar Chart (middle)** – Now sits immediately to the right of the voice view and shares the exact bin geometry. It displays the tracked per-bin baseline directly, normalizes each frame, smooths it, and draws between radius 10–45 with the same palette so you can watch ambient noise patterns wrap the circle in real time. It mirrors the Voice Max Hz control so you can see exactly which ambient frequencies are feeding the subtraction curve, and the **Show Noise Polar** toggle hides it while expanding the rectangular bars when you want extra screen space.
	- **Full-Range Bars (right edge)** – 50 rectangular bins from 0–8000 Hz that now stretch all the way to the tuning panel. Magnitudes get a 60 dB offset, subtract the rolling 35th-percentile baseline (4-frame history), and smooth with an EMA before recoloring.

## Signal Processing

- 80 Hz Butterworth high-pass filter applied before analysis.
- Windowed FFT (Hann) with exponential smoothing on magnitudes to avoid flicker.
- Custom mel filterbank caches to reduce per-frame cost.
- Adaptive percentile-based noise suppression for both voice and bar charts.
- Per-bin ambient noise floor visualization mirrors the voice bins so you can see filtered noise separate from speech energy.

## Tuning Controls

On the right edge, a toggleable panel exposes live sliders and pill-style toggles:

- **Bar/wave pipeline**: Baseline %, Noise Frames, Offset (dB), Scale (dB), Smoothing, Wave Fade (trail decay).
- **Voice pipeline**: Voice Gain, Voice Noise Frames (history length), Voice Threshold, Voice Max Hz (drives both polar charts), plus **Voice Aggro** which softens or tightens the subtraction baseline.
- **Spectrogram pipeline**: **Spectro Aggro** (multiplier on the noise frame before subtraction), Spectro Bins (minimum density), and a Spectro Hz RangeSlider that enforces a ~50 Hz span while re-sampling mel bins for smooth zooming.
- **Layout/subtraction toggles**: Spectro Subtract, Voice Subtract, Show Noise Spectro, Show Noise Polar — each rendered as a compact filled-circle toggle that both controls and describes the current state.
- Hovering any slider label or toggle text pops a minimalist tooltip describing what it controls, so you can tune by feel without referencing docs.

## Running It

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
python main.py
```

Grant microphone access when prompted. Use `LINE_PROFILE=1 python main.py --run-for 30` to profile short sessions.
