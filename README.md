# AX-CPT (Context-Dependent Continuous Performance Test)

A Pygame implementation of the AX-CPT task for measuring sustained attention, context maintenance, and inhibitory control.

## Requirements

- Python 3.7+
- Pygame
- OpenCV (for webcam tracking)
- MediaPipe (for face/eye tracking)
- NumPy

Install dependencies:
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install pygame opencv-python mediapipe numpy
```

## Running the Task

```bash
python main.py
```

## Task Instructions

- Letters will appear one at a time on the screen
- **Press SPACEBAR only when you see X immediately after A**
- Do not respond to any other letter combinations
- Between letters, a fixation cross (+) will appear to help you maintain focus
- The task will run for the configured number of trials

## Trial Types

- **AX**: A followed by X → **RESPOND** (target)
- **BX**: B followed by X → Do not respond
- **AY**: A followed by Y → Do not respond  
- **BY**: B followed by Y → Do not respond

## Configuration

Edit `config.json` to customize task parameters:

### Timing Parameters
- `stimulus_duration_ms`: How long each letter appears (default: 300ms)
- `response_window_ms`: Time allowed to respond after stimulus (default: 1000ms)
- `inter_stimulus_interval_ms`: Delay between stimuli (default: 700ms)

### Task Parameters
- `total_trials`: Number of trials to run (default: 800)
- `target_probability`: Frequency of AX trials (default: 0.15 = 15%)

### Display Parameters
- `fullscreen`: Run in fullscreen mode (default: true)
- `font_size`: Size of stimulus letters (default: 96)
- `background_color`: RGB color for background (default: [0, 0, 0] = black)
- `stimulus_color`: RGB color for stimuli (default: [255, 255, 255] = white)
- `random_colors`: Randomly vary stimulus colors each trial (default: false)

### Fixation & Input
- `show_fixation`: Show fixation cross during ISI (default: true)
- `fixation_symbol`: Symbol to show during ISI (default: "+")
- `response_key`: Key to press for responses (default: "space")

### Webcam Tracking (NEW!)
- `webcam_tracking.enabled`: Enable/disable webcam tracking (default: true)
- `webcam_tracking.camera_index`: Camera device index (default: 0 for default webcam)
- `webcam_tracking.save_frame_data`: Save frame-by-frame tracking data (default: true)
- `webcam_tracking.save_session_summary`: Save session-level summary (default: true)

## Data Output

Results are automatically saved to CSV files with timestamps in the `results/` directory.

### Main Trial Data
File: `results/ax_cpt_results_YYYY-MM-DDTHH-MM-SS.csv`

**Behavioral Columns:**
- `trial_index`: Trial number
- `stimulus`: Letter shown (A, B, X, or Y)
- `previous_stimulus`: Previous letter shown
- `trial_type`: AX, BX, AY, BY, or NONE
- `response`: 1 if participant responded, 0 if not
- `correct`: 1 if response was correct, 0 if incorrect
- `reaction_time_ms`: Response time in milliseconds (empty if no response)
- `stimulus_onset_timestamp`: High-precision timestamp of stimulus onset

**Tracking Columns (if enabled):**
- `blink_count`: Number of blinks during trial
- `blink_rate`: Blinks per second during trial
- `mean_head_distance`: Average distance from camera (normalized)
- `head_movement_variance`: Measure of head stability (lower = more stable)
- `looking_away_count`: Number of frames where head turned >30° away
- `frames_tracked`: Number of frames successfully tracked

### Frame-Level Tracking Data (if enabled)
File: `results/ax_cpt_tracking_frames_YYYY-MM-DDTHH-MM-SS.csv`

High-frequency data (~30 Hz) for detailed analysis:
- `timestamp`: Frame timestamp
- `trial_index`: Associated trial number
- `head_x`, `head_y`, `head_z`: Head position (normalized)
- `head_pitch`, `head_yaw`, `head_roll`: Head orientation in degrees
- `left_eye_aspect_ratio`, `right_eye_aspect_ratio`: Eye openness (lower = more closed)
- `is_blinking`: Boolean indicating blink state

### Session Summary (if enabled)
File: `results/ax_cpt_tracking_session_YYYY-MM-DDTHH-MM-SS.csv`

Aggregate metrics for the entire session:
- `total_blinks`: Total blinks across all trials
- `total_frames_tracked`: Total frames processed
- `total_trials_tracked`: Number of trials with tracking data
- `overall_blink_rate`: Average blinks per second
- `mean_head_stability`: Overall head movement variance
- `engagement_score`: Derived metric (0-1, higher = better engagement)
- `fatigue_indicator`: Change in blink rate over time (positive = increased fatigue)
- `session_duration_seconds`: Total tracking duration

## Analyzing Results

### Jupyter Notebook Analysis

A comprehensive Jupyter notebook is provided for analyzing your results:

```bash
# Install analysis dependencies (if not already installed)
pip install pandas matplotlib seaborn jupyter

# Launch Jupyter
jupyter notebook analyze_axcpt_results.ipynb
```

The notebook includes:
- **Behavioral Performance Analysis**: Accuracy, reaction times, error patterns
- **Tracking Data Visualization**: Blink rates, head movement, engagement metrics
- **Combined Analysis**: Correlation between attention and performance
- **Trial-by-Trial Deep Dive**: Detailed inspection of individual trials
- **Professional Visualizations**: Publication-ready charts and graphs

The notebook automatically loads the most recent session data from the `results/` folder.

## Controls

- **SPACEBAR** (or configured key): Respond to target
- **ESC**: Quit task (data will be saved)

## Webcam Tracking Features

The task now includes **optional webcam-based behavioral tracking** using MediaPipe for real-time face and eye detection.

### What's Tracked (Tier 1 - Currently Implemented)

**Blink Detection:**
- Automatic blink detection using Eye Aspect Ratio (EAR)
- Blink count and rate per trial
- Can identify fatigue patterns over time

**Head Pose Tracking:**
- 3D head position (X, Y, Z coordinates)
- Head orientation (pitch, yaw, roll in degrees)
- Head movement stability/variance
- Detection of looking away from screen

### Research Applications

This data enables analysis of:
- **Attention lapses**: Correlation between looking away and missed targets
- **Fatigue**: Increased blink rate over time
- **Engagement**: Head stability and posture changes
- **Restlessness**: Head movement patterns (relevant for ADHD research)
- **Error prediction**: Behavioral markers before incorrect responses

### Testing Tracking

Test the tracking system without running the full experiment:
```bash
python test_tracking.py
```

This will run a 15-second test (3 trials × 5 seconds) and save test data files.

### Disabling Tracking

To run the task without webcam tracking, set in `config.json`:
```json
"webcam_tracking": {
  "enabled": false
}
```

### Future Enhancements (Tier 2/3)

The architecture is designed to easily add:
- **Gaze tracking**: Where on screen the participant is looking
- **Pupil dilation**: Cognitive load and arousal measurement
- **Facial expressions**: Emotion detection (frustration, boredom)
- **Advanced attention metrics**: Fixation stability, saccade detection

## Project Structure

```
/ax-cpt
  ├── main.py                    # Main game loop and rendering
  ├── config.json                # Configuration parameters
  ├── stimulus_generator.py      # Trial sequence generation
  ├── logger.py                  # CSV data logging (behavioral + tracking)
  ├── utils.py                   # Timing and utility functions
  ├── webcam_tracker.py          # Webcam tracking with MediaPipe
  ├── tracking_logger.py         # Multi-level tracking data logger
  ├── test_tracking.py           # Test script for tracking system
  ├── requirements.txt           # Python dependencies
  └── README.md                  # This file
```

## Notes

- The task uses high-precision timing (`time.perf_counter()`)
- Frame-locked rendering at 60 FPS
- No feedback is provided during trials
- First trial cannot be AX (no prior context)
- Trial sequences are randomized while maintaining target probability

