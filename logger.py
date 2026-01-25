"""
Data logging for AX-CPT task
"""
import csv
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
        "stimulus_onset_timestamp"
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
        stimulus_onset_timestamp: float
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
        self.trials.append(trial_data)
    
    def save_to_csv(self, filename: Optional[str] = None):
        """
        Save logged trials to CSV file.
        
        Args:
            filename: Output filename (auto-generated if None)
        """
        if filename is None:
            timestamp = get_filename_timestamp()
            filename = f"ax_cpt_results_{timestamp}.csv"
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.HEADERS)
            writer.writeheader()
            writer.writerows(self.trials)
        
        print(f"Data saved to: {filename}")
        return filename
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Calculate summary statistics from logged trials."""
        if not self.trials:
            return {}
        
        total_trials = len(self.trials)
        responses = sum(1 for t in self.trials if t["response"] == 1)
        correct = sum(1 for t in self.trials if t["correct"] == 1)
        
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
            "by_trial_type": trial_types
        }

