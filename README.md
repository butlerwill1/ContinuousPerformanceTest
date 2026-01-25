# AX-CPT (Context-Dependent Continuous Performance Test)

A Pygame implementation of the AX-CPT task for measuring sustained attention, context maintenance, and inhibitory control.

## Requirements

- Python 3.7+
- Pygame

Install dependencies:
```bash
pip install pygame
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

## Data Output

Results are automatically saved to a CSV file with timestamp:
```
ax_cpt_results_2026-01-24T21-03-55.csv
```

### CSV Columns

- `trial_index`: Trial number
- `stimulus`: Letter shown (A, B, X, or Y)
- `previous_stimulus`: Previous letter shown
- `trial_type`: AX, BX, AY, BY, or NONE
- `response`: 1 if participant responded, 0 if not
- `correct`: 1 if response was correct, 0 if incorrect
- `reaction_time_ms`: Response time in milliseconds (empty if no response)
- `stimulus_onset_timestamp`: High-precision timestamp of stimulus onset

## Controls

- **SPACEBAR** (or configured key): Respond to target
- **ESC**: Quit task (data will be saved)

## Project Structure

```
/cpt
  ├── main.py                 # Main game loop and rendering
  ├── config.json             # Configuration parameters
  ├── stimulus_generator.py   # Trial sequence generation
  ├── logger.py               # CSV data logging
  ├── utils.py                # Timing and utility functions
  └── README.md               # This file
```

## Notes

- The task uses high-precision timing (`time.perf_counter()`)
- Frame-locked rendering at 60 FPS
- No feedback is provided during trials
- First trial cannot be AX (no prior context)
- Trial sequences are randomized while maintaining target probability

