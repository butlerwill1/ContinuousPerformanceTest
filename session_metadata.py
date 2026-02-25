"""
Session metadata logger for AX-CPT task
Handles pre-test questionnaire data collection and storage
"""
import csv
from typing import Dict, Any, Optional
from datetime import datetime


class SessionMetadata:
    """Collects and saves session metadata from pre-test questionnaire."""
    
    def __init__(self):
        """Initialize metadata storage."""
        self.metadata: Dict[str, Any] = {}
    
    def set_metadata(self, metadata: Dict[str, Any]):
        """
        Set metadata from questionnaire.
        
        Args:
            metadata: Dictionary of metadata fields and values
        """
        self.metadata = metadata
    
    def save_to_csv(self, session_dir: str, filename: Optional[str] = None):
        """
        Save metadata to CSV file.
        
        Args:
            session_dir: Directory for this session
            filename: Output filename (defaults to 'session_metadata.csv')
        """
        if filename is None:
            filename = f"{session_dir}/session_metadata.csv"
        
        # Write as key-value pairs
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['field', 'value'])
            
            for key, value in self.metadata.items():
                # Convert value to string, handle None
                value_str = '' if value is None else str(value)
                writer.writerow([key, value_str])
        
        print(f"Session metadata saved to: {filename}")
        return filename
    
    @staticmethod
    def create_empty_metadata(timestamp: str) -> Dict[str, Any]:
        """
        Create empty metadata dictionary with just timestamp.
        
        Args:
            timestamp: Session timestamp
            
        Returns:
            Dictionary with timestamp and empty fields
        """
        return {
            'timestamp': timestamp,
            'adhd_med_taken': '',
            'hours_since_med': '',
            'sleep_hours': '',
            'caffeine_hours_ago': '',
            'exercise_hours_ago': '',
            'stress_level': '',
            'mental_fatigue': '',
            'notes': ''
        }

