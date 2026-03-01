"""
Multi-level logging for webcam tracking data.

Data Hierarchy:
- FRAME: Single webcam capture (~60 per second). Contains head pose, eye state at one moment.
- TRIAL: One cue-probe pair (~4 seconds). Aggregates ~240 frames into blink count, head movement stats.
- SESSION: Complete test run (~20 minutes). Aggregates ~300 trials into overall performance metrics.
"""
import csv
from typing import List, Dict, Any, Optional
from dataclasses import asdict
from webcam_tracker import FrameMetrics
import numpy as np


class TrackingLogger:
    """
    Logs tracking data at three levels of granularity.

    Levels:
    - Frame-level: Raw data from each webcam capture (~60 FPS)
    - Trial-level: Aggregated metrics per trial (one cue-probe pair)
    - Session-level: Overall statistics for entire test session
    """

    # Headers for frame-level data (one row per webcam frame, ~60 per second)
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
    
    # Headers for trial-level data (one row per trial, aggregated from ~240 frames)
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
        Initialize tracking logger. Called once in main.py at startup.

        Args:
            enabled: Whether tracking logging is enabled
        """
        self.enabled = enabled
        
        # Storage for different levels
        self.frame_data: List[FrameMetrics] = []
        self.trial_data: List[Dict[str, Any]] = []
        
        # Session-level accumulators (aggregated across all trials)
        self.session_stats = {
            'total_blinks': 0,
            'total_frames': 0,
            'total_trials_tracked': 0,
            'head_distances': [],
            'blink_rates_over_time': []
        }
    
    def log_frame(self, frame_metrics: FrameMetrics):
        """
        Log a single FRAME of tracking data. Called ~60 times per second during trials in main.py.

        A frame = one webcam capture with head position, eye state at a single moment in time.

        Args:
            frame_metrics: Metrics from webcam_tracker.process_frame()
        """
        if not self.enabled or frame_metrics is None:
            return
        
        self.frame_data.append(frame_metrics)
        self.session_stats['total_frames'] += 1
    
    def log_trial(self, trial_metrics: Dict[str, Any]):
        """
        Log aggregated TRIAL-level metrics. Called once per trial in main.py after trial ends.

        A trial = one cue-probe pair (~4 seconds, ~240 frames). Metrics are aggregated from all frames.

        Args:
            trial_metrics: Dictionary from webcam_tracker.end_trial() (blink count, head movement variance, etc.)
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
    
    def save_frame_data(self, session_dir: str, filename: Optional[str] = None) -> str:
        """
        Save all frame-level data to CSV. Called once at end of session in main.py.

        Args:
            session_dir: Session directory path
            filename: Optional custom filename (default: 'tracking_frames.csv')

        Returns:
            Path to saved file
        """
        if not self.enabled or not self.frame_data:
            return ""

        if filename is None:
            filename = f"{session_dir}/tracking_frames.csv"

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
        Get all trial-level data. Called by logger.py to merge with behavioral data.

        Returns:
            List of trial metrics from all trials
        """
        return self.trial_data
    
    def calculate_session_summary(self) -> Dict[str, Any]:
        """
        Calculate SESSION-level summary statistics. Called by summary_report.py and save_session_summary().

        A session = complete test run (~20 minutes, ~300 trials). Metrics are aggregated from all trials.

        Returns:
            Dictionary with blink rate, head movement, posture consistency, fatigue indicator
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

        # Convert to blinks per minute for display
        blink_rate_per_minute = overall_blink_rate * 60.0

        # Head movement metrics
        mean_head_movement = 0.0
        head_movement_variance = 0.0
        if self.session_stats['head_distances']:
            mean_head_movement = np.mean(self.session_stats['head_distances'])
            head_movement_variance = np.var(self.session_stats['head_distances'])

        # Count total looking away events from trial data
        total_looking_away_events = sum(
            trial.get('looking_away_count', 0) for trial in self.trial_data
        )

        # Posture consistency (derived metric: inverse of head movement variance)
        # Higher score = better posture consistency
        posture_consistency = 0.0
        if head_movement_variance > 0:
            posture_consistency = 1.0 / (1.0 + head_movement_variance)

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
            'blink_rate_per_minute': blink_rate_per_minute,
            'mean_head_movement': mean_head_movement,
            'total_looking_away_events': total_looking_away_events,
            'posture_consistency': posture_consistency,
            'fatigue_indicator': fatigue_indicator,
            'session_duration_seconds': total_duration
        }
    
    def _get_empty_session_summary(self) -> Dict[str, Any]:
        """Return empty session summary when no data available."""
        return {
            'total_blinks': 0,
            'total_frames_tracked': 0,
            'total_trials_tracked': 0,
            'blink_rate_per_minute': 0.0,
            'mean_head_movement': 0.0,
            'total_looking_away_events': 0,
            'posture_consistency': 0.0,
            'fatigue_indicator': 0.0,
            'session_duration_seconds': 0.0
        }
    
    def save_session_summary(self, session_dir: str, filename: Optional[str] = None) -> str:
        """
        Save session-level summary to CSV. Called once at end of session in main.py.

        Args:
            session_dir: Session directory path
            filename: Optional custom filename (default: 'tracking_session.csv')

        Returns:
            Path to saved file
        """
        if not self.enabled:
            return ""

        if filename is None:
            filename = f"{session_dir}/tracking_session.csv"

        summary = self.calculate_session_summary()

        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=summary.keys())
            writer.writeheader()
            writer.writerow(summary)

        print(f"Session-level tracking summary saved to: {filename}")
        return filename

