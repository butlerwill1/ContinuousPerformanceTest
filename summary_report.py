"""
Generate human-readable ASCII summary reports for AX-CPT sessions
"""
from typing import Dict, Any, Optional
from datetime import datetime
import csv


class SummaryReportGenerator:
    """Generates formatted ASCII summary reports from session data."""
    
    def __init__(self, session_dir: str):
        """
        Initialize summary report generator.
        
        Args:
            session_dir: Path to session directory containing data files
        """
        self.session_dir = session_dir
        
    def generate_report(
        self,
        performance_stats: Dict[str, Any],
        tracking_stats: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Generate complete ASCII summary report.
        
        Args:
            performance_stats: Performance metrics from TrialLogger.get_summary_stats()
            tracking_stats: Tracking metrics from TrackingLogger.calculate_session_summary()
            metadata: Session metadata from questionnaire
            
        Returns:
            Formatted ASCII report string
        """
        lines = []
        
        # Header
        lines.extend(self._generate_header(metadata))
        lines.append("")
        
        # Performance section
        lines.extend(self._generate_performance_section(performance_stats))
        lines.append("")
        
        # Attention tracking section (if available)
        if tracking_stats and tracking_stats.get('total_trials_tracked', 0) > 0:
            lines.extend(self._generate_tracking_section(tracking_stats))
            lines.append("")
        
        # Context section (if available)
        if metadata:
            lines.extend(self._generate_context_section(metadata))
            lines.append("")
        
        return "\n".join(lines)
    
    def _generate_header(self, metadata: Optional[Dict[str, str]]) -> list:
        """Generate report header."""
        lines = []
        lines.append("=" * 80)
        lines.append(" " * 24 + "AX-CPT SESSION SUMMARY")
        lines.append("=" * 80)
        lines.append("")
        
        # Extract session info from directory name
        session_name = self.session_dir.split('/')[-1]
        
        if metadata and 'timestamp' in metadata:
            timestamp_str = metadata['timestamp']
            try:
                dt = datetime.strptime(timestamp_str, "%Y-%m-%dT%H-%M-%S")
                date_str = dt.strftime("%B %d, %Y at %I:%M %p")
            except:
                date_str = timestamp_str
            lines.append(f"Session: {session_name}")
            lines.append(f"Date: {date_str}")
        else:
            lines.append(f"Session: {session_name}")
        
        # Extract duration from folder name if present
        if '_' in session_name:
            duration = session_name.split('_')[-1]
            # Format duration nicely
            if duration.endswith('m'):
                duration_text = duration.replace('m', ' minutes')
            elif duration.endswith('s'):
                duration_text = duration.replace('s', ' seconds')
            else:
                duration_text = duration
            lines.append(f"Duration: {duration_text}")
        
        return lines
    
    def _generate_performance_section(self, stats: Dict[str, Any]) -> list:
        """Generate performance metrics section."""
        lines = []
        lines.append("=" * 80)
        lines.append(" " * 26 + "PERFORMANCE METRICS")
        lines.append("=" * 80)
        lines.append("")
        
        # Overall metrics
        accuracy = stats.get('accuracy', 0)
        total_trials = stats.get('total_trials', 0)
        correct = stats.get('total_correct', 0)
        incorrect = total_trials - correct
        
        accuracy_bar = self._create_progress_bar(accuracy, width=20)
        accuracy_rating = self._get_accuracy_rating(accuracy)
        
        lines.append(f"Overall Accuracy:        {accuracy:.1%}  {accuracy_bar}  [{accuracy_rating}]")
        lines.append(f"Total Trials:            {total_trials}")
        lines.append(f"Correct Responses:       {correct}")
        lines.append(f"Incorrect Responses:     {incorrect}")
        lines.append("")

        # Reaction times
        median_rt = stats.get('median_rt_ms')
        mean_rt = stats.get('mean_rt_ms')

        if median_rt is not None:
            lines.append(f"Reaction Time (median):  {median_rt:.1f} ms")
        if mean_rt is not None:
            lines.append(f"Reaction Time (mean):    {mean_rt:.1f} ms")

        if median_rt is not None or mean_rt is not None:
            lines.append("")

        # Trial type breakdown
        by_type = stats.get('by_trial_type', {})
        if by_type:
            lines.append("Trial Type Breakdown:")

            # AX (Target)
            if 'AX' in by_type:
                ax = by_type['AX']
                ax_acc = ax['correct'] / ax['total'] if ax['total'] > 0 else 0
                lines.append(f"  AX (Target):           Accuracy: {ax_acc:.1%}  ({ax['correct']}/{ax['total']})  [Hit Rate]")

            # BX (Cue Lure)
            if 'BX' in by_type:
                bx = by_type['BX']
                bx_acc = bx['correct'] / bx['total'] if bx['total'] > 0 else 0
                lines.append(f"  BX (Cue Lure):         Accuracy: {bx_acc:.1%}  ({bx['correct']}/{bx['total']})  [Inhibition]")

            # AY (Probe Lure)
            if 'AY' in by_type:
                ay = by_type['AY']
                ay_acc = ay['correct'] / ay['total'] if ay['total'] > 0 else 0
                lines.append(f"  AY (Probe Lure):       Accuracy: {ay_acc:.1%}  ({ay['correct']}/{ay['total']})  [Context Use]")

            # BY (Non-target)
            if 'BY' in by_type:
                by = by_type['BY']
                by_acc = by['correct'] / by['total'] if by['total'] > 0 else 0
                lines.append(f"  BY (Non-target):       Accuracy: {by_acc:.1%}  ({by['correct']}/{by['total']})  [Baseline]")

        return lines

    def _generate_tracking_section(self, stats: Dict[str, Any]) -> list:
        """Generate attention tracking section."""
        lines = []
        lines.append("=" * 80)
        lines.append(" " * 28 + "ATTENTION TRACKING")
        lines.append("=" * 80)
        lines.append("")

        # Blink rate
        blink_rate = stats.get('blink_rate_per_minute', 0)
        blink_rating = self._get_blink_rate_rating(blink_rate)
        lines.append(f"Blink Rate:              {blink_rate:.1f} blinks/min  [{blink_rating}]")

        total_blinks = stats.get('total_blinks', 0)
        lines.append(f"Total Blinks:            {total_blinks}")

        looking_away = stats.get('total_looking_away_events', 0)
        lines.append(f"Looking Away Events:     {looking_away}")
        lines.append("")

        # Head movement
        head_movement = stats.get('mean_head_movement', 0)
        lines.append(f"Head Movement:           {head_movement:.2f}  [Lower = more stable]")

        # Posture consistency
        posture_consistency = stats.get('posture_consistency', 0)
        posture_bar = self._create_progress_bar(posture_consistency, width=20)
        posture_rating = self._get_posture_consistency_rating(posture_consistency)
        lines.append(f"Posture Consistency:     {posture_consistency:.2f}  {posture_bar}  [{posture_rating}]")

        # Fatigue indicator
        fatigue = stats.get('fatigue_indicator', 0)
        fatigue_pct = fatigue * 100
        fatigue_bar = self._create_progress_bar(abs(fatigue), width=20)
        fatigue_rating = self._get_fatigue_rating(fatigue)

        sign = "+" if fatigue >= 0 else ""
        lines.append(f"Fatigue Indicator:       {sign}{fatigue_pct:.1f}% {fatigue_bar}  [{fatigue_rating}]")

        return lines

    def _generate_context_section(self, metadata: Dict[str, str]) -> list:
        """Generate context and notes section."""
        lines = []
        lines.append("=" * 80)
        lines.append(" " * 28 + "CONTEXT & NOTES")
        lines.append("=" * 80)
        lines.append("")

        # ADHD Medication
        med_taken = metadata.get('adhd_med_taken', '').lower() == 'y'
        if med_taken:
            med_mg = metadata.get('adhd_med_mg', '')
            hours_since = metadata.get('hours_since_med', '')
            lines.append(f"ADHD Medication:         Yes - {med_mg}mg ({hours_since} hours ago)")
        else:
            lines.append(f"ADHD Medication:         No")

        # Sleep
        sleep_hours = metadata.get('sleep_hours', '')
        if sleep_hours:
            lines.append(f"Sleep:                   {sleep_hours} hours")

        # Mental fatigue
        mental_fatigue = metadata.get('mental_fatigue', '')
        if mental_fatigue:
            lines.append(f"Mental Fatigue:          {mental_fatigue}/10")

        # Caffeine
        caffeine_mg = metadata.get('caffeine_mg', '')
        if caffeine_mg:
            caffeine_hours = metadata.get('caffeine_hours_ago', '')
            lines.append(f"Caffeine:                {caffeine_mg}mg ({caffeine_hours} hours ago)")
        else:
            lines.append(f"Caffeine:                None")

        # Exercise
        exercise_hours = metadata.get('exercise_hours_ago', '')
        if exercise_hours:
            lines.append(f"Exercise:                {exercise_hours} hours ago")

        # Stress
        stress = metadata.get('stress_level', '')
        if stress:
            lines.append(f"Stress Level:            {stress}/10")

        # Notes
        notes = metadata.get('notes', '')
        if notes:
            lines.append("")
            lines.append(f"Notes: {notes}")

        lines.append("")
        lines.append("=" * 80)

        return lines

    def _create_progress_bar(self, value: float, width: int = 20) -> str:
        """
        Create ASCII progress bar.

        Args:
            value: Value between 0 and 1
            width: Width of the bar in characters

        Returns:
            ASCII progress bar string
        """
        filled = int(value * width)
        empty = width - filled
        return "█" * filled + "░" * empty

    def _get_accuracy_rating(self, accuracy: float) -> str:
        """Get rating label for accuracy."""
        if accuracy >= 0.90:
            return "Excellent"
        elif accuracy >= 0.85:
            return "Good"
        elif accuracy >= 0.75:
            return "Fair"
        else:
            return "Needs Improvement"

    def _get_blink_rate_rating(self, blink_rate: float) -> str:
        """Get rating label for blink rate (normal is 15-20/min)."""
        if blink_rate < 10:
            return "Very Low"
        elif blink_rate < 15:
            return "Low"
        elif blink_rate <= 25:
            return "Normal"
        elif blink_rate <= 35:
            return "Elevated"
        else:
            return "High"

    def _get_posture_consistency_rating(self, posture_consistency: float) -> str:
        """Get rating label for posture consistency score."""
        if posture_consistency >= 0.85:
            return "Excellent"
        elif posture_consistency >= 0.75:
            return "Good"
        elif posture_consistency >= 0.60:
            return "Fair"
        else:
            return "Low"

    def _get_fatigue_rating(self, fatigue: float) -> str:
        """Get rating label for fatigue indicator."""
        if fatigue < -0.10:
            return "Decreased (unusual)"
        elif fatigue < 0.05:
            return "Minimal"
        elif fatigue < 0.15:
            return "Slight increase"
        elif fatigue < 0.30:
            return "Moderate increase"
        else:
            return "Significant increase"

    def save_report(self, report: str, filename: str = "summary.txt") -> str:
        """
        Save report to file.

        Args:
            report: Report string to save
            filename: Output filename

        Returns:
            Full path to saved file
        """
        filepath = f"{self.session_dir}/{filename}"
        with open(filepath, 'w') as f:
            f.write(report)
        return filepath

    def load_metadata_from_csv(self) -> Optional[Dict[str, str]]:
        """Load metadata from session_metadata.csv."""
        filepath = f"{self.session_dir}/session_metadata.csv"
        try:
            metadata = {}
            with open(filepath, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    if len(row) >= 2:
                        metadata[row[0]] = row[1]
            return metadata
        except FileNotFoundError:
            return None


