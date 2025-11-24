# Audio Visualizer - Session Context

## Project Overview
Real-time audio visualization tool with artistic aesthetic, built with Python, matplotlib, numpy, scipy, and sounddevice.

## Current Architecture

### Visual Layout (3 rows)
1. **Top (1.5x height)**: Mel-scaled spectrogram (0-8000 Hz)
2. **Middle (1x height)**: Waveform with dynamic scaling
3. **Bottom (1.5x height)**: Split into two panels
   - **Left (1/3 width)**: Voice frequencies - Polar bar chart (0-2000 Hz)
   - **Right (2/3 width)**: Full range frequency bars (0-8000 Hz)

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
- **Polar bar chart** showing 0-2000 Hz fundamentals
- 30 bins arranged radially around circle
- **Adaptive noise filtering**:
   - Uses 10th percentile per-frame noise estimate (history size = 1)
   - Displays bins only when they exceed noise floor by 8 dB
   - Bars are amplified ×`voice_amplification` (default 1.6, slider-controlled) before thresholding
- Radial offset keeps bars between 15 and 60 units from center so even subtle activity remains visible
- Minimal labels: 4 frequency markers (250, 500, 750, 1000 Hz)
- 3 magnitude markers (30, 45, 60 dB)
- Starts at top (12 o'clock), goes clockwise

#### 4. Full Range Frequency Bars
- 50 bins covering 0-8000 Hz
- Interpolated magnitude values for smooth visualization
- Color-mapped by magnitude
- Bottom 35th percentile treated as adaptive noise floor (rolling window of 4 frames) before visualization
- 60 dB magnitude offset/scale keep palette responsive while peaks can extend higher
- Interactive slider panel (right edge) adjusts baseline percentile, noise history length, offset, scale, smoothing, wave fade, spectrogram bin count, voice gain, and the spectrogram Hz span in real time; panel can be hidden via toggle button

#### 5. Tuning Controls (Right Panel)
- Hide/Show button toggles the entire panel
- Sliders: Baseline %, Noise Frames, Offset (dB), Scale (dB), Smoothing, Wave Fade, Voice Gain, Spectro Bins, Spectro Hz Range
- Spectro Hz uses a RangeSlider that enforces at least ~50 Hz span and re-samples mel bins to keep transitions smooth

### Configuration Variables (Extracted)
- `label_alpha = 0.35` - opacity for all labels/ticks
- `label_fontsize = 8` - font size for labels
- `waveform_linewidth = 1.5` - waveform line thickness
- `waveform_alpha = 0.9` - waveform line opacity
- `waveform_fade_decay = 0.0` - slider-adjustable waveform trail decay factor
- `voice_noise_threshold = 8` - dB above noise floor to display
- `voice_noise_history_size = 1` - frames for noise tracking
- `voice_amplification = 1.6` - multiplier to boost polar bars (tunable via UI)
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
# Track 10th percentile as noise floor each frame
current_noise_floor = np.percentile(voice_magnitude, 10)
# Average over recent history (currently size 1)
adaptive_noise_floor = np.mean(history)
# Keep bins 8 dB above threshold after amplification
threshold = adaptive_noise_floor + 8
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
- Voice frequency range (0-1000 Hz) captures fundamentals; could extend to ~4000 Hz for consonants
- Matplotlib still warns about `cache_frame_data=True` when FuncAnimation length is inferred; user is fine ignoring it for now

## Profiling Notes (2025-11-23)
- Profiling attempts used this session: 8 / 20
- Hotspots continue to be `update_spectrogram`, `update_voice_frequency_bands`, and `update_frequency_bands`, but caching/path-mutation plus `_fast_percentile` reduced per-frame overhead considerably
- Latest gains: selective bar updates, PolyCollection path mutation, Hann/mel caching, waveform axis caching, and percentile helper got `update` down to ~0.147s for the profiled clip; next focus is percentile-heavy sections and remaining numpy work
