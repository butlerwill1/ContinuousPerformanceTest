"""
Test script to generate a summary report from existing session data
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from summary_report import SummaryReportGenerator
import csv


def load_metadata(session_dir):
    """Load metadata from CSV."""
    filepath = f"{session_dir}/session_metadata.csv"
    metadata = {}
    try:
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 2:
                    metadata[row[0]] = row[1]
    except FileNotFoundError:
        pass
    return metadata


def load_tracking_stats(session_dir):
    """Load tracking stats from CSV."""
    filepath = f"{session_dir}/tracking_session.csv"
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            stats = next(reader)
            # Convert string values to appropriate types
            # Handle both old and new column names for backward compatibility
            return {
                'total_blinks': int(stats['total_blinks']),
                'total_frames_tracked': int(stats['total_frames_tracked']),
                'total_trials_tracked': int(stats['total_trials_tracked']),
                'blink_rate_per_minute': float(stats['blink_rate_per_minute']),
                'mean_head_movement': float(stats.get('mean_head_movement', stats.get('mean_head_distance', 0))),
                'total_looking_away_events': int(stats['total_looking_away_events']),
                'posture_consistency': float(stats.get('posture_consistency', stats.get('engagement_score', 0))),
                'fatigue_indicator': float(stats['fatigue_indicator']),
                'session_duration_seconds': float(stats['session_duration_seconds'])
            }
    except FileNotFoundError:
        return None


def calculate_performance_stats(session_dir):
    """Calculate performance stats from trial data CSV."""
    filepath = f"{session_dir}/trial_data.csv"
    
    trials = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            trials.append(row)
    
    total_trials = len(trials)
    responses = sum(1 for t in trials if t['response'] == '1')
    correct = sum(1 for t in trials if t['correct'] == '1')
    
    # Reaction times
    reaction_times = []
    for t in trials:
        rt = t['reaction_time_ms']
        if rt and rt != '':
            try:
                reaction_times.append(float(rt))
            except ValueError:
                pass
    
    median_rt = None
    mean_rt = None
    if reaction_times:
        sorted_rts = sorted(reaction_times)
        median_rt = sorted_rts[len(sorted_rts) // 2]
        mean_rt = sum(reaction_times) / len(reaction_times)
    
    # By trial type
    trial_types = {}
    for trial in trials:
        tt = trial['trial_type']
        if tt not in trial_types:
            trial_types[tt] = {'total': 0, 'correct': 0, 'responses': 0}
        trial_types[tt]['total'] += 1
        trial_types[tt]['correct'] += int(trial['correct'])
        trial_types[tt]['responses'] += int(trial['response'])
    
    return {
        'total_trials': total_trials,
        'total_responses': responses,
        'total_correct': correct,
        'accuracy': correct / total_trials if total_trials > 0 else 0,
        'median_rt_ms': median_rt,
        'mean_rt_ms': mean_rt,
        'by_trial_type': trial_types
    }


if __name__ == "__main__":
    # Test with the existing session
    session_dir = "results/2026-02-28T18-48-32_34m"
    
    print(f"Generating summary report for: {session_dir}")
    
    # Load data
    metadata = load_metadata(session_dir)
    tracking_stats = load_tracking_stats(session_dir)
    performance_stats = calculate_performance_stats(session_dir)
    
    # Generate report
    report_gen = SummaryReportGenerator(session_dir)
    report = report_gen.generate_report(
        performance_stats=performance_stats,
        tracking_stats=tracking_stats,
        metadata=metadata
    )
    
    # Print to console
    print("\n" + "=" * 80)
    print(report)
    print("=" * 80)
    
    # Save to file
    filepath = report_gen.save_report(report)
    print(f"\nReport saved to: {filepath}")

