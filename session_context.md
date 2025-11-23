# Audio Visualizer - Session Context

## Project Overview
Real-time audio visualization tool with artistic aesthetic, built with Python, matplotlib, numpy, scipy, and sounddevice.

## Current Architecture

### Visual Layout (3 rows)
1. **Top (1.5x height)**: Mel-scaled spectrogram (0-8000 Hz)
2. **Middle (1x height)**: Waveform with dynamic scaling
3. **Bottom (1.5x height)**: Split into two panels
   - **Left (1/3 width)**: Voice frequencies - Polar bar chart (0-1000 Hz)
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

#### 2. Dynamic Waveform
- Zoom level: 10x (adjustable padding around signal)
- 8-frame history for smooth zoom adaptation
- 0.8 smoothing factor for transitions
- Scale indicator shows current zoom level

#### 3. Voice Polar Chart (NEW)
- **Polar bar chart** showing 0-1000 Hz in circular layout
- 30 bins arranged radially around circle
- **Adaptive noise filtering**: 
  - Tracks noise floor over 50 frames
  - Only shows bars 15 dB above adaptive threshold
  - Removes background noise automatically
- Minimal labels: 4 frequency markers (250, 500, 750, 1000 Hz)
- 3 magnitude markers (20, 40, 60 dB)
- Starts at top (12 o'clock), goes clockwise

#### 4. Full Range Frequency Bars
- 100 bins covering 0-8000 Hz
- Interpolated magnitude values for smooth visualization
- Color-mapped by magnitude
- 80 dB magnitude offset and scale for visibility

### Configuration Variables (Extracted)
- `label_alpha = 0.35` - opacity for all labels/ticks
- `label_fontsize = 8` - font size for labels
- `waveform_linewidth = 1.5` - waveform line thickness
- `waveform_alpha = 0.9` - waveform line opacity
- `voice_noise_threshold = 15` - dB above noise floor to display
- `voice_noise_history_size = 50` - frames for noise tracking

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
# Track 20th percentile as noise floor
current_noise_floor = np.percentile(voice_magnitude, 20)
# Average over 50 frames for adaptive threshold
adaptive_noise_floor = np.mean(history)
# Filter points 15 dB above threshold
threshold = adaptive_noise_floor + 15
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
