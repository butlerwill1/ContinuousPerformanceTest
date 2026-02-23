"""
Multi-level logging for webcam tracking data
Handles frame-level, trial-level, and session-level data
"""
import csv
import os
from typing import List, Dict, Any, Optional
from dataclasses import asdict
from utils import get_filename_timestamp
from webcam_tracker import FrameMetrics
import numpy as np


class TrackingLogger:
    """Logs tracking data at three levels of granularity."""
    
    # Headers for frame-level data (high frequency)
    FRAME_HEADERS = [
        "timestamp",
        "trial_index",
        "head_x",
        "head_y",
        "head_z",
        "head_pitch",
        "head_yaw",
        "head_roll",
        "left_eye_aspect_ratio",
        "right_eye_aspect_ratio",
        "is_blinking"
    ]
    
    # Headers for trial-level data (aggregated per trial)
    TRIAL_HEADERS = [
        "blink_count",
        "blink_rate",
        "mean_head_distance",
        "head_movement_variance",
        "looking_away_count",
        "frames_tracked"
    ]
    
    def __init__(self, enabled: bool = True):
        """
        Initialize tracking logger.
        
        Args:
            enabled: Whether tracking logging is enabled
        """
        self.enabled = enabled
        
        # Storage for different levels
        self.frame_data: List[FrameMetrics] = []
        self.trial_data: List[Dict[str, Any]] = []
        
        # Session-level accumulators
        self.session_stats = {
            'total_blinks': 0,
            'total_frames': 0,
            'total_trials_tracked': 0,
            'head_distances': [],
            'blink_rates_over_time': []
        }
    
    def log_frame(self, frame_metrics: FrameMetrics):
        """
        Log a single frame of tracking data.
        
        Args:
            frame_metrics: Metrics from a single frame
        """
        if not self.enabled or frame_metrics is None:
            return
        
        self.frame_data.append(frame_metrics)
        self.session_stats['total_frames'] += 1
    
    def log_trial(self, trial_metrics: Dict[str, Any]):
        """
        Log aggregated trial-level metrics.
        
        Args:
            trial_metrics: Dictionary of trial-level metrics
        """
        if not self.enabled:
            return
        
        self.trial_data.append(trial_metrics)
        
        # Update session statistics
        self.session_stats['total_blinks'] += trial_metrics.get('blink_count', 0)
        self.session_stats['total_trials_tracked'] += 1
        
        if trial_metrics.get('mean_head_distance', 0) > 0:
            self.session_stats['head_distances'].append(
                trial_metrics['mean_head_distance']
            )
        
        if trial_metrics.get('blink_rate', 0) > 0:
            self.session_stats['blink_rates_over_time'].append(
                trial_metrics['blink_rate']
            )
    
    def save_frame_data(self, filename: Optional[str] = None) -> str:
        """
        Save frame-level data to CSV.

        Args:
            filename: Output filename (auto-generated if None)

        Returns:
            Filename where data was saved
        """
        if not self.enabled or not self.frame_data:
            return ""

        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)

        if filename is None:
            timestamp = get_filename_timestamp()
            filename = f"results/ax_cpt_tracking_frames_{timestamp}.csv"

        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.FRAME_HEADERS)
            writer.writeheader()

            for frame in self.frame_data:
                # Convert dataclass to dict and write
                frame_dict = asdict(frame)
                writer.writerow(frame_dict)

        print(f"Frame-level tracking data saved to: {filename}")
        return filename
    
    def get_trial_data(self) -> List[Dict[str, Any]]:
        """
        Get trial-level data (to be merged with main trial logger).
        
        Returns:
            List of trial-level tracking metrics
        """
        return self.trial_data
    
    def calculate_session_summary(self) -> Dict[str, Any]:
        """
        Calculate session-level summary statistics.
        
        Returns:
            Dictionary of session-level metrics
        """
        if not self.enabled or self.session_stats['total_trials_tracked'] == 0:
            return self._get_empty_session_summary()
        
        # Overall blink rate
        total_duration = 0
        if self.frame_data:
            total_duration = (self.frame_data[-1].timestamp - 
                            self.frame_data[0].timestamp)
        
        overall_blink_rate = (self.session_stats['total_blinks'] / total_duration 
                             if total_duration > 0 else 0.0)
        
        # Head stability (lower variance = more stable)
        mean_head_stability = 0.0
        if self.session_stats['head_distances']:
            mean_head_stability = np.var(self.session_stats['head_distances'])
        
        # Engagement score (derived metric: inverse of head movement + blink consistency)
        # Higher score = better engagement
        engagement_score = 0.0
        if mean_head_stability > 0:
            engagement_score = 1.0 / (1.0 + mean_head_stability)
        
        # Fatigue indicator: compare first half vs second half blink rates
        fatigue_indicator = 0.0
        blink_rates = self.session_stats['blink_rates_over_time']
        if len(blink_rates) >= 4:
            midpoint = len(blink_rates) // 2
            first_half_mean = np.mean(blink_rates[:midpoint])
            second_half_mean = np.mean(blink_rates[midpoint:])
            # Positive value = increased blinking in second half (fatigue)
            if first_half_mean > 0:
                fatigue_indicator = (second_half_mean - first_half_mean) / first_half_mean
        
        return {
            'total_blinks': self.session_stats['total_blinks'],
            'total_frames_tracked': self.session_stats['total_frames'],
            'total_trials_tracked': self.session_stats['total_trials_tracked'],
            'overall_blink_rate': overall_blink_rate,
            'mean_head_stability': mean_head_stability,
            'engagement_score': engagement_score,
            'fatigue_indicator': fatigue_indicator,
            'session_duration_seconds': total_duration
        }
    
    def _get_empty_session_summary(self) -> Dict[str, Any]:
        """Return empty session summary when no data available."""
        return {
            'total_blinks': 0,
            'total_frames_tracked': 0,
            'total_trials_tracked': 0,
            'overall_blink_rate': 0.0,
            'mean_head_stability': 0.0,
            'engagement_score': 0.0,
            'fatigue_indicator': 0.0,
            'session_duration_seconds': 0.0
        }
    
    def save_session_summary(self, filename: Optional[str] = None) -> str:
        """
        Save session-level summary to CSV.

        Args:
            filename: Output filename (auto-generated if None)

        Returns:
            Filename where data was saved
        """
        if not self.enabled:
            return ""

        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)

        if filename is None:
            timestamp = get_filename_timestamp()
            filename = f"results/ax_cpt_tracking_session_{timestamp}.csv"

        summary = self.calculate_session_summary()

        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=summary.keys())
            writer.writeheader()
            writer.writerow(summary)

        print(f"Session-level tracking summary saved to: {filename}")
        return filename

