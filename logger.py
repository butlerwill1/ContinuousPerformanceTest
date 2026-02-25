"""
Data logging for AX-CPT task
"""
import csv
import os
from typing import List, Dict, Any, Optional
from utils import get_filename_timestamp


class TrialLogger:
    """Logs trial data to CSV file."""

    HEADERS = [
        "trial_index",
        "stimulus",
        "previous_stimulus",
        "trial_type",
        "response",
        "correct",
        "reaction_time_ms",
        "stimulus_onset_timestamp",
        # Tracking data (optional, will be empty if tracking disabled)
        "blink_count",
        "blink_rate",
        "mean_head_distance",
        "head_movement_variance",
        "looking_away_count",
        "frames_tracked"
    ]
    
    def __init__(self):
        """Initialize trial logger."""
        self.trials: List[Dict[str, Any]] = []
        
    def log_trial(
        self,
        trial_index: int,
        stimulus: str,
        previous_stimulus: Optional[str],
        trial_type: str,
        response: int,
        correct: int,
        reaction_time_ms: Optional[float],
        stimulus_onset_timestamp: float,
        tracking_data: Optional[Dict[str, Any]] = None
    ):
        """
        Log a single trial.

        Args:
            trial_index: Index of the trial
            stimulus: Current stimulus letter
            previous_stimulus: Previous stimulus letter (or None)
            trial_type: Trial type (AX, BX, AY, BY, NONE)
            response: 1 if responded, 0 if no response
            correct: 1 if correct, 0 if incorrect
            reaction_time_ms: Reaction time in milliseconds (None if no response)
            stimulus_onset_timestamp: Timestamp when stimulus appeared
            tracking_data: Optional dictionary of tracking metrics from webcam
        """
        trial_data = {
            "trial_index": trial_index,
            "stimulus": stimulus,
            "previous_stimulus": previous_stimulus if previous_stimulus else "",
            "trial_type": trial_type,
            "response": response,
            "correct": correct,
            "reaction_time_ms": f"{reaction_time_ms:.2f}" if reaction_time_ms is not None else "",
            "stimulus_onset_timestamp": f"{stimulus_onset_timestamp:.6f}"
        }

        # Add tracking data if available
        if tracking_data:
            trial_data["blink_count"] = tracking_data.get("blink_count", "")
            trial_data["blink_rate"] = f"{tracking_data.get('blink_rate', 0):.3f}" if tracking_data.get('blink_rate') else ""
            trial_data["mean_head_distance"] = f"{tracking_data.get('mean_head_distance', 0):.3f}" if tracking_data.get('mean_head_distance') else ""
            trial_data["head_movement_variance"] = f"{tracking_data.get('head_movement_variance', 0):.6f}" if tracking_data.get('head_movement_variance') else ""
            trial_data["looking_away_count"] = tracking_data.get("looking_away_count", "")
            trial_data["frames_tracked"] = tracking_data.get("frames_tracked", "")
        else:
            # Empty tracking data
            trial_data["blink_count"] = ""
            trial_data["blink_rate"] = ""
            trial_data["mean_head_distance"] = ""
            trial_data["head_movement_variance"] = ""
            trial_data["looking_away_count"] = ""
            trial_data["frames_tracked"] = ""

        self.trials.append(trial_data)
    
    def save_to_csv(self, session_dir: Optional[str] = None, filename: Optional[str] = None):
        """
        Save logged trials to CSV file.

        Args:
            session_dir: Directory for this session (auto-generated if None)
            filename: Output filename (defaults to 'trial_data.csv')
        """
        # Create session directory if not provided
        if session_dir is None:
            timestamp = get_filename_timestamp()
            session_dir = f"results/{timestamp}"

        os.makedirs(session_dir, exist_ok=True)

        if filename is None:
            filename = f"{session_dir}/trial_data.csv"

        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.HEADERS)
            writer.writeheader()
            writer.writerows(self.trials)

        print(f"Trial data saved to: {filename}")
        return session_dir
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Calculate summary statistics from logged trials."""
        if not self.trials:
            return {}

        total_trials = len(self.trials)
        responses = sum(1 for t in self.trials if t["response"] == 1)
        correct = sum(1 for t in self.trials if t["correct"] == 1)

        # Calculate reaction time statistics (only for trials with responses)
        reaction_times = []
        for t in self.trials:
            rt = t["reaction_time_ms"]
            # Handle None, empty string, or convert to float
            if rt is not None and rt != "" and rt != "None":
                try:
                    reaction_times.append(float(rt))
                except (ValueError, TypeError):
                    pass  # Skip invalid values

        if reaction_times:
            sorted_rts = sorted(reaction_times)
            median_rt = sorted_rts[len(sorted_rts) // 2]
            mean_rt = sum(reaction_times) / len(reaction_times)
        else:
            median_rt = None
            mean_rt = None

        # Calculate by trial type
        trial_types = {}
        for trial in self.trials:
            tt = trial["trial_type"]
            if tt not in trial_types:
                trial_types[tt] = {"total": 0, "correct": 0, "responses": 0}
            trial_types[tt]["total"] += 1
            trial_types[tt]["correct"] += trial["correct"]
            trial_types[tt]["responses"] += trial["response"]

        return {
            "total_trials": total_trials,
            "total_responses": responses,
            "total_correct": correct,
            "accuracy": correct / total_trials if total_trials > 0 else 0,
            "median_rt_ms": median_rt,
            "mean_rt_ms": mean_rt,
            "by_trial_type": trial_types
        }

