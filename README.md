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

### Data Hierarchy

**Understanding the three levels of data:**

- **FRAME**: Single webcam capture (~60 per second)
  - Contains: Head position, eye state at one moment in time
  - Example: At 12.345 seconds, head was at position (0.5, 0.2, 2.8), eyes open

- **TRIAL**: One cue-probe pair (~4 seconds, ~240 frames)
  - Contains: Aggregated metrics from all frames in that trial
  - Example: During trial #42, participant blinked 3 times, head moved 0.15 units

- **SESSION**: Complete test run (~20 minutes, ~300 trials)
  - Contains: Overall statistics aggregated from all trials
  - Example: Across entire session, average blink rate was 36/min, posture consistency 0.83

### Main Trial Data
File: `results/TIMESTAMP_DURATION/trial_data.csv`

One row per trial (cue-probe pair). ~300 rows for a 20-minute session.

**Behavioral Columns:**
- `trial_index`: Trial number (one trial = one cue-probe pair)
- `stimulus`: Letter shown (A, B, X, or Y)
- `previous_stimulus`: Previous letter shown
- `trial_type`: AX, BX, AY, BY, or NONE
- `response`: 1 if participant responded, 0 if not
- `correct`: 1 if response was correct, 0 if incorrect
- `reaction_time_ms`: Response time in milliseconds (empty if no response)
- `stimulus_onset_timestamp`: High-precision timestamp of stimulus onset

**Tracking Columns (if enabled, aggregated from ~240 frames per trial):**
- `blink_count`: Number of blinks during this trial
- `blink_rate`: Blinks per second during this trial
- `mean_head_distance`: Average distance from camera during this trial
- `head_movement_variance`: Head position variance during this trial (lower = more stable)
- `looking_away_count`: Number of frames where head turned >30° away
- `frames_tracked`: Number of frames successfully tracked in this trial

### Frame-Level Tracking Data (if enabled)
File: `results/TIMESTAMP_DURATION/tracking_frames.csv`

High-frequency data (~60 FPS) for detailed analysis. Each row = one webcam frame. ~18,000 rows for a 20-minute session.

- `timestamp`: Frame timestamp
- `trial_index`: Which trial this frame belongs to
- `head_x`, `head_y`, `head_z`: Head position at this moment (z = distance from camera)
- `head_pitch`, `head_yaw`, `head_roll`: Head orientation at this moment (degrees)
- `left_eye_aspect_ratio`, `right_eye_aspect_ratio`: Eye openness at this moment (lower = more closed)
- `is_blinking`: Whether eyes were closed at this moment

### Session Summary (if enabled)
File: `results/TIMESTAMP_DURATION/tracking_session.csv`

Aggregate metrics for the entire session. One row total, aggregated from ~300 trials.

- `total_blinks`: Total blinks across all trials
- `total_frames_tracked`: Total frames processed (~18,000 for 20-minute session)
- `total_trials_tracked`: Number of trials with tracking data (~300)
- `blink_rate_per_minute`: Average blinks per minute across entire session
- `mean_head_movement`: Overall head movement across entire session (lower = more stable)
- `posture_consistency`: Posture consistency metric (0-1, higher = better consistency)
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
- **Posture Consistency**: Head movement and posture changes
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
- **Blink duration**: Measure how long each blink lasts (longer blinks may indicate fatigue)
- **PERCLOS (Percentage of Eye Closure)**: Percentage of time eyes are >80% closed (drowsiness indicator)
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
  ├── logger.py                  # CSV data logging (behavioral data)
  ├── utils.py                   # Timing and utility functions
  ├── webcam_tracker.py          # Webcam tracking with MediaPipe
  ├── tracking_logger.py         # Multi-level tracking data logger
  ├── questionnaire.py           # Pre-test questionnaire for metadata
  ├── session_metadata.py        # Session metadata management
  ├── summary_report.py          # ASCII summary report generator
  ├── requirements.txt           # Python dependencies
  ├── README.md                  # This file
  ├── tests/                     # Test scripts
  │   ├── test_tracking.py       # Test script for tracking system
  │   ├── test_summary.py        # Test script for summary reports
  │   └── test_generator.py      # Test script for stimulus generation
  └── results/                   # Session data (timestamped folders)
      └── YYYY-MM-DDTHH-MM-SS_XXm/
          ├── config.json        # Session configuration snapshot
          ├── trial_data.csv     # Behavioral performance data
          ├── tracking_frames.csv      # Frame-level tracking data
          ├── tracking_session.csv     # Session-level tracking summary
          ├── session_metadata.csv     # Pre-test questionnaire responses
          └── summary.txt        # Human-readable summary report
```

## Notes

- The task uses high-precision timing (`time.perf_counter()`)
- Frame-locked rendering at 60 FPS
- No feedback is provided during trials
- First trial cannot be AX (no prior context)
- Trial sequences are randomized while maintaining target probability

