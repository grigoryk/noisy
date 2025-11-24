# Audio Visualizer - Session Context

## Project Overview
Real-time audio visualization tool with artistic aesthetic, built with Python, matplotlib, numpy, scipy, and sounddevice.

## Current Architecture

### Visual Layout (3 rows)
1. **Top (1.5x height)**: Mel-scaled spectrogram (0-8000 Hz)
2. **Middle (1x height)**: Waveform with dynamic scaling
3. **Bottom (1.5x height)**: Three-column layout with voice polar on the left, the noise polar chart directly beside it, and the full-range bars stretching all the way to the tuning panel on the right
   - **Left column**: Voice frequencies - Polar bar chart (0–2000 Hz by default, live-adjustable up to 8 kHz)
   - **Middle column**: Noise polar chart that mirrors the voice bins so ambient energy is visible separately and follows the same adjustable ceiling
   - **Right column**: Full range frequency bars (0–8000 Hz) spanning the remaining width for maximum horizontal real estate

### Color Palette - Deep Ocean Purple Theme
- Background: `#0a1628` (deep navy)
- Palette: `['#0a1628', '#513EE6', '#9354E3', '#C254E3', '#E354CA', '#E60E3F']`
- Waveform: `#E354CA` (bright magenta)
- Borders: `#D074EB` (light purple)
- Text: `#C254E3` (medium purple)
- All visualizations use LinearSegmentedColormap from the palette

### Key Features

#### 1. Mel-Scaled Spectrogram
- 40 mel banks, 80-8000 Hz range
- Logarithmic frequency display: [100, 200, 500, 1000, 2000, 4000, 8000] Hz
- 30th percentile noise floor removal
- Power = 1.5 for contrast enhancement
- Range slider (in tuning panel) crops the spectrogram to any 0-8 kHz window and re-samples the data so the zoomed view always displays at least 40 bins

#### 2. Dynamic Waveform
- Zoom level: 10x (adjustable padding around signal)
- 8-frame history for smooth zoom adaptation
- 0.8 smoothing factor for transitions
- Scale indicator shows current zoom level
- Optional trail fade keeps recent frames visible with slider-controlled exponential decay

#### 3. Voice Polar Chart (NEW)
- 30 radial bins covering 0–2000 Hz by default (slider extends ceiling up to 8 kHz or down to 500 Hz)
- **Adaptive noise filtering**:
   - Interpolates each frame at 30 bin centers and stores up to 45 frames of history per bin
   - Uses the per-bin 25th-percentile baseline after at least 5 frames; before then it falls back to the frame’s 10th-percentile floor
   - Displays bins only when they clear the baseline by 8 dB and then multiplies by the voice gain slider (default 1.6)
- Radial offset keeps bars between radius 15–60
- Angle labels (4 positions) track quarter-points of the current Voice Max Hz so ticks always match the selected band

#### 4. Noise Polar Chart (NEW)
- Shares the same bin geometry as the voice chart but renders the tracked per-bin noise baseline
- Radii span 10–45 so the floor sits inside the voice energy
- Normalized each frame and smoothed with the same EMA so it mirrors motion cleanly
- Automatically follows the Voice Max Hz slider so the noise visualization always matches the speech band you’re monitoring

#### 5. Full Range Frequency Bars
- 50 rectangular bins across 0–8000 Hz with adaptive percentile noise removal
- 60 dB offset/scale keep the palette responsive
- Interactive slider panel (right edge) adjusts baseline percentile, noise history length, offset, scale, smoothing, wave fade, voice gain, Voice Max Hz, spectrogram bin count, and the spectrogram Hz span in real time; panel can be hidden via toggle button

#### 6. Tuning Controls (Right Panel)
- Hide/Show button toggles the entire panel
- Sliders: Baseline %, Noise Frames, Offset (dB), Scale (dB), Smoothing, Wave Fade, Voice Gain, Voice Noise Frames, Voice Threshold, **Voice Max Hz**, Spectro Bins, Spectro Hz Range
- Voice Max Hz simultaneously moves the ceiling for both polar charts so you can focus on fundamentals or include consonants without restarting
- Spectro Hz uses a RangeSlider that enforces at least ~50 Hz span and re-samples mel bins to keep transitions smooth
- Hover overlays explain what each label/slider adjusts when you pause over the control

### Configuration Variables (Extracted)
- `label_alpha = 0.35` - opacity for all labels/ticks
- `label_fontsize = 8` - font size for labels
- `waveform_linewidth = 1.5` - waveform line thickness
- `waveform_alpha = 0.9` - waveform line opacity
- `waveform_fade_decay = 0.0` - slider-adjustable waveform trail decay factor
- `voice_noise_threshold = 8` - dB above noise floor to display
- `voice_noise_history_size = 45` - frames stored per bin for percentile-based noise tracking
- `voice_amplification = 1.6` - multiplier to boost polar bars (tunable via UI)
- `voice_max_freq = 2000` - default upper bound for the voice/noise polar charts (slider-adjustable 500–8000 Hz)
- `bands_baseline_percentile = 35` - dynamic noise floor percentile for full-range bars
- `bands_noise_history_size = 4` - frames to average baseline
- `bands_magnitude_offset = 60`, `bands_magnitude_scale = 60`
- `spectrogram_view_min = 0`, `spectrogram_view_max = 8000` - controlled by live range slider
- `spectrogram_min_view_bins = 40` - minimum number of columns shown in any zoomed spectrogram view

### Visual Aesthetics
- **No titles** on any graphs
- **No borders** (all spines invisible)
- **Faint labels** (alpha 0.35)
- **Unified color scheme** across all visualizations
- Minimalist, art-focused design

## Technical Details

### Audio Processing
- High-pass filter: 80 Hz (removes low-frequency noise)
- FFT smoothing: 0.5
- Sample rate: Configurable (typically 44100 Hz)
- Hanning window applied before FFT

### Frequency Analysis
- Uses mel-scale for perceptually meaningful spacing
- Interpolation prevents gaps in frequency bars
- Baseline percentile removal for noise reduction
- Smoothing between frames for stable visualization

### Adaptive Noise Detection (Voice Chart)
```python
bin_inputs = np.interp(voice_bin_centers, voice_freqs, voice_magnitude)
voice_history.append(bin_inputs)
voice_history = voice_history[-45:]

if len(voice_history) >= 5:
   per_bin_baseline = np.percentile(np.vstack(voice_history), 25, axis=0)
else:
   per_bin_baseline = np.percentile(bin_inputs, 10)

signal = np.maximum(bin_inputs - per_bin_baseline - 8, 0)
bars = signal * voice_amplification
```

## Recent Changes (Session History)
1. Extracted magic values to configuration variables
2. Fixed deprecated `plt.cm.get_cmap()` → `plt.colormaps[]` (then removed lookup since using object directly)
3. Reduced waveform zoom from 2.1 to 1.5 (then user adjusted to 10)
4. Split bottom frequency graph into voice (1/3) + full range (2/3)
5. Tried multiple voice visualizations:
   - Bar chart → too much fill
   - Line plot with fill → not right feel
   - Scatter plot with dots → requested different approach
   - **Polar bar chart** (current) → circular radial bars
6. Added adaptive background noise detection for voice chart
7. Reduced polar chart labels to 4 angles + 3 radial markers
8. Added `--run-for` CLI flag so profiling runs can auto-terminate after N seconds
9. Skipped redundant bar height/color updates by caching drawn values (fewer matplotlib patch updates per frame)
10. Added `_fast_percentile` helper to replace repeated `np.percentile` calls in noise removal paths
11. Offset and clipped voice polar bars between 15-60 radius so activity remains visible even at low magnitudes
12. Introduced adaptive baseline removal for rectangular frequency bars (35th percentile, rolling 4-frame average) and retuned magnitude offsets for more note contrast
13. Added matplotlib slider UI to tweak frequency-bar tuning parameters live (with hide/show toggle)
14. Added spectrogram frequency RangeSlider to zoom into specific Hz spans without restarting the app
15. Restricted the range slider to the spectrogram and slice mel bins dynamically so zoomed views actually change what the top panel shows, always re-sampling to keep at least 40 bins visible
16. Added a Spectro Bins slider tied to `spectrogram_min_view_bins` and shifted the tuning panel ~10% left to keep the controls fully visible
17. Smoothed the Spectro Hz range slider by always sampling neighboring mel bins so the max handle no longer causes sudden jumps near 7.5 kHz
18. Added adaptive resampling that interpolates the spectrogram view to a continuously-scaled bin count, eliminating visual jumps when the slider span changes
19. Removed the unused `mel_noise_floor` setting and redundant cache resets in the setup helpers to reduce clutter without changing behavior
20. Added a Voice Gain slider that adjusts `voice_amplification` so polar bars can be boosted or tamed live
21. Added waveform trails plus a Wave Fade slider so previous frames linger and fade smoothly
22. Reworked the voice polar chart to track per-bin percentile noise floors (45-frame history) so ambient noise stays suppressed
23. Updated README and this document to describe the per-bin voice noise logic and slider-clearing behavior
24. Added tuning sliders for `voice_noise_history_size` and `voice_noise_threshold` so ambient suppression can be adjusted live
25. Added hover tooltips to every tuning control so users get inline descriptions without leaving the UI
26. Added a dedicated noise polar chart that renders the per-bin noise baseline in real time next to the full-range bars and wired it into the same smoothing/percentile pipeline as the voice view
27. Added a Voice Max Hz slider that retunes the voice/noise polar ceiling (and tick labels) live up to 8 kHz

## File Structure
```
noisy/
├── main.py              # Main application code
├── agents.md            # Agent instructions/preferences
├── requirements.txt     # Python dependencies
├── session_context.md   # This file
└── env/                 # Virtual environment
```

## Dependencies
- numpy 2.3.5
- matplotlib 3.10.7
- scipy 1.16.3
- sounddevice 0.5.3
- Python 3.13.2

## User Preferences (from agents.md)
- Minimal comments (only non-obvious code)
- No docstrings or routine explanations
- Very modular, flexible code structure
- Descriptive function names showing purpose
- Small, readable functions
- Clear logical grouping
- Agent should read entire codebase at session start
- Update agents.md when receiving new instructions

## Known Issues / Notes

### Active Issues
- Adaptive noise floor takes ~50 frames to stabilize on startup (initial frames may show incorrect filtering)

### Observations / Potential Improvements
- Polar chart currently uses all 30 bins - could be reduced for cleaner look
- Voice frequency range defaults to 0–2000 Hz, but slider now extends to 8 kHz for consonant detail if needed
- Matplotlib still warns about `cache_frame_data=True` when FuncAnimation length is inferred; user is fine ignoring it for now

## Profiling Notes (2025-11-23)
- Profiling attempts used this session: 8 / 20
- Hotspots continue to be `update_spectrogram`, `update_voice_frequency_bands`, and `update_frequency_bands`, but caching/path-mutation plus `_fast_percentile` reduced per-frame overhead considerably
- Latest gains: selective bar updates, PolyCollection path mutation, Hann/mel caching, waveform axis caching, and percentile helper got `update` down to ~0.147s for the profiled clip; next focus is percentile-heavy sections and remaining numpy work

## Profiling Notes (2025-11-24)
- Profiling attempts used this session: 12 / 20
- Command template: `LINE_PROFILE=1 env/bin/python main.py --run-for 10`
- Run 10 (`profile_output_2025-11-24T091000_voice.txt`): `AudioVisualizer.update` totaled 0.153 s across 84 frames (~1.8 ms/frame). `update_spectrogram` held 52.7% of the budget, `update_voice_frequency_bands` 32.3%, and `update_frequency_bands` 9.3%. `_update_noise_floor_display` alone was 40.9% of the voice function (≈134 µs/frame) so it became the next target after the spectrogram vectorization.
- Run 11 (`profile_output_2025-11-24T092100_noisevert.txt`): moving the noise polar chart to `PolyCollection.set_verts` backfired—Matplotlib rebuilt every polygon per frame and `_update_noise_floor_display` ballooned to 61.8% of the voice cost. Archived for reference and rolled back immediately.
- Run 12 (`profile_output_2025-11-24T092430_noisecolors.txt`): restored per-path edits but gated colormap generation behind the color-change flag and cleaned up the normalization math. `_update_noise_floor_display` dropped to 9.5 ms total (≈104 µs/frame, 32.6% of the voice function) and `AudioVisualizer.update` now averages 0.139 s across 91 frames (~1.5 ms/frame). The remaining hotspots are `_update_spectrogram_image` (≈56% of `update`), the percentile stack in `update_voice_frequency_bands`, and the PolyCollection mutations in the rectangular frequency bars.
- `python -m line_profiler -rtmz profile_output.lprof` still fails because the shimmed `python` binary isn’t present; plain `env/bin/python -m line_profiler ...` works and the text dumps above live alongside the `.lprof` for future comparison.
