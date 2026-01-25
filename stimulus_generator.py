"""
Stimulus sequence generator for AX-CPT task
"""
import random
from typing import List, Tuple


class StimulusGenerator:
    """Generates trial sequences for AX-CPT task."""
    
    STIMULI = ["A", "B", "X", "Y"]
    CUE_STIMULI = ["A", "B"]
    PROBE_STIMULI = ["X", "Y"]
    
    def __init__(self, total_trials: int, target_probability: float = 0.15):
        """
        Initialize stimulus generator.
        
        Args:
            total_trials: Total number of trials to generate
            target_probability: Probability of AX (target) trials
        """
        self.total_trials = total_trials
        self.target_probability = target_probability
        
    def generate_sequence(self) -> List[str]:
        """
        Generate a sequence of stimuli following AX-CPT constraints.
        
        Returns:
            List of stimuli strings
        """
        sequence = []
        
        # Calculate approximate trial type counts
        num_ax = int(self.total_trials * self.target_probability)
        remaining = self.total_trials - num_ax
        
        # Distribute remaining trials among BX, AY, BY
        # Use roughly equal proportions for non-target types
        num_bx = remaining // 3
        num_ay = remaining // 3
        num_by = remaining - num_bx - num_ay
        
        # Create trial type list
        trial_types = (
            ["AX"] * num_ax +
            ["BX"] * num_bx +
            ["AY"] * num_ay +
            ["BY"] * num_by
        )
        
        # Shuffle to randomize order
        random.shuffle(trial_types)
        
        # Convert trial types to stimulus sequence
        for trial_type in trial_types:
            sequence.append(trial_type[0])  # Cue (A or B)
            sequence.append(trial_type[1])  # Probe (X or Y)
        
        return sequence
    
    def get_trial_type(self, current_stimulus: str, previous_stimulus: str) -> str:
        """
        Determine trial type based on current and previous stimulus.
        
        Args:
            current_stimulus: Current stimulus letter
            previous_stimulus: Previous stimulus letter (or None)
            
        Returns:
            Trial type string (AX, BX, AY, BY, or NONE)
        """
        if previous_stimulus is None:
            return "NONE"
        
        if current_stimulus in self.PROBE_STIMULI and previous_stimulus in self.CUE_STIMULI:
            return f"{previous_stimulus}{current_stimulus}"
        
        return "NONE"
    
    def is_target_trial(self, trial_type: str) -> bool:
        """Check if trial type is a target (AX) trial."""
        return trial_type == "AX"

