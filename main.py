"""
AX-CPT (Context-Dependent Continuous Performance Test)
Main game loop and rendering
"""
import pygame
import sys
import os
import shutil
import random
from typing import Optional, Tuple
from utils import load_config, get_timestamp, ms_to_seconds, calculate_elapsed_ms, get_filename_timestamp
from stimulus_generator import StimulusGenerator
from logger import TrialLogger
from webcam_tracker import WebcamTracker
from tracking_logger import TrackingLogger
from questionnaire import PreTestQuestionnaire
from session_metadata import SessionMetadata


class AXCPTGame:
    """Main game class for AX-CPT task."""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the game."""
        self.config = load_config(config_path)
        self.logger = TrialLogger()

        # Session management
        self.session_timestamp = None
        self.session_dir = None
        self.session_metadata = SessionMetadata()

        # Initialize webcam tracking
        tracking_config = self.config.get("webcam_tracking", {})
        self.tracking_enabled = tracking_config.get("enabled", False)

        if self.tracking_enabled:
            self.webcam_tracker = WebcamTracker(
                enabled=True,
                camera_index=tracking_config.get("camera_index", 0)
            )
            self.tracking_logger = TrackingLogger(enabled=True)
            print("Webcam tracking enabled")
        else:
            self.webcam_tracker = None
            self.tracking_logger = None
            print("Webcam tracking disabled")

        # Initialize Pygame
        pygame.init()

        # Set up display
        if self.config["fullscreen"]:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode((800, 600))

        pygame.display.set_caption("AX-CPT Task")

        # Get screen dimensions
        self.width, self.height = self.screen.get_size()
        self.center = (self.width // 2, self.height // 2)

        # Set up font
        self.font = pygame.font.Font(None, self.config["font_size"])
        self.instruction_font = pygame.font.Font(None, 36)

        # Colors
        self.bg_color = tuple(self.config["background_color"])
        self.stim_color = tuple(self.config["stimulus_color"])

        # Clock for frame timing
        self.clock = pygame.time.Clock()

        # Response key
        self.response_key = getattr(pygame, f"K_{self.config['response_key'].upper()}")

        # Calculate total trials from session duration
        time_per_trial_ms = (
            self.config["stimulus_duration_ms"] +
            self.config["response_window_ms"] +
            self.config["inter_stimulus_interval_ms"]
        )
        session_duration_ms = self.config["session_duration_minutes"] * 60 * 1000
        total_trials = int(session_duration_ms / time_per_trial_ms)

        print(f"Session duration: {self.config['session_duration_minutes']} minutes")
        print(f"Time per trial: {time_per_trial_ms}ms")
        print(f"Calculated trials: {total_trials}")

        # Generate stimulus sequence
        self.generator = StimulusGenerator(
            total_trials,
            self.config["target_probability"]
        )
        self.stimulus_sequence = self.generator.generate_sequence()

        # Trial state
        self.current_trial = 0
        self.previous_stimulus = None
        self.trial_start_time = None
        self.response_made = False
        self.response_time = None
        
    def show_instructions(self):
        """Display instruction screen."""
        self.screen.fill(self.bg_color)
        
        instructions = [
            "AX-CPT Task",
            "",
            "You will see letters appear one at a time.",
            "",
            f"Press {self.config['response_key'].upper()} ONLY when you see X after A.",
            "",
            "Do not respond to any other combination.",
            "",
            "Press SPACE to begin..."
        ]
        
        y_offset = self.height // 4
        for line in instructions:
            text_surface = self.instruction_font.render(line, True, self.stim_color)
            text_rect = text_surface.get_rect(center=(self.center[0], y_offset))
            self.screen.blit(text_surface, text_rect)
            y_offset += 50
        
        pygame.display.flip()
        
        # Wait for space key
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.quit_game()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        waiting = False
                    elif event.key == pygame.K_ESCAPE:
                        self.quit_game()
    
    def show_end_screen(self):
        """Display comprehensive end screen with performance summary."""
        self.screen.fill(self.bg_color)

        # Get performance stats
        stats = self.logger.get_summary_stats()

        # Get tracking stats if available
        tracking_stats = None
        if self.webcam_tracker and self.tracking_logger:
            tracking_stats = self.tracking_logger.calculate_session_summary()

        # Title - check if session was completed or ended early
        total_trials = stats.get('total_trials', 0)
        expected_trials = len(self.stimulus_sequence)
        completed_early = total_trials < expected_trials

        title_font = pygame.font.Font(None, 64)
        title_text = "Session Summary" if completed_early else "Task Complete!"
        title = title_font.render(title_text, True, self.stim_color)
        title_rect = title.get_rect(center=(self.center[0], 60))
        self.screen.blit(title, title_rect)

        # Show early completion notice if applicable
        if completed_early:
            subtitle_font = pygame.font.Font(None, 28)
            subtitle = subtitle_font.render(
                f"(Ended early: {total_trials}/{expected_trials} trials completed)",
                True, (255, 255, 100)  # Yellow color
            )
            subtitle_rect = subtitle.get_rect(center=(self.center[0], 95))
            self.screen.blit(subtitle, subtitle_rect)

        # Create two columns: Performance (left) and Tracking (right)
        left_x = self.width // 4
        right_x = 3 * self.width // 4
        start_y = 150 if completed_early else 140

        # === LEFT COLUMN: Performance Metrics ===
        self._draw_section_header("Performance", left_x, start_y)
        y = start_y + 50

        # Overall accuracy
        accuracy = stats.get('accuracy', 0)
        accuracy_color = self._get_performance_color(accuracy)
        self._draw_stat_line(f"Overall Accuracy: {accuracy:.1%}", left_x, y, accuracy_color)
        y += 40

        # Total trials
        self._draw_stat_line(f"Total Trials: {stats.get('total_trials', 0)}", left_x, y)
        y += 40

        # Reaction times
        median_rt = stats.get('median_rt_ms')
        mean_rt = stats.get('mean_rt_ms')
        if median_rt is not None:
            self._draw_stat_line(f"Median RT: {median_rt:.0f} ms", left_x, y)
            y += 35
        if mean_rt is not None:
            self._draw_stat_line(f"Mean RT: {mean_rt:.0f} ms", left_x, y)
            y += 45

        # Accuracy by trial type
        self._draw_stat_line("By Trial Type:", left_x, y, size=28)
        y += 35

        by_type = stats.get('by_trial_type', {})
        for trial_type in ['AX', 'BX', 'AY', 'BY']:
            if trial_type in by_type:
                type_stats = by_type[trial_type]
                type_acc = type_stats['correct'] / type_stats['total'] if type_stats['total'] > 0 else 0
                type_color = self._get_performance_color(type_acc)
                self._draw_stat_line(
                    f"  {trial_type}: {type_acc:.1%} ({type_stats['correct']}/{type_stats['total']})",
                    left_x, y, type_color, size=26
                )
                y += 32

        # === RIGHT COLUMN: Tracking Metrics (if available) ===
        if tracking_stats and tracking_stats.get('total_frames_tracked', 0) > 0:
            self._draw_section_header("Attention Tracking", right_x, start_y)
            y = start_y + 50

            # Blink metrics
            total_blinks = tracking_stats.get('total_blinks', 0)
            blink_rate = tracking_stats.get('blink_rate_per_minute', 0)
            self._draw_stat_line(f"Total Blinks: {total_blinks}", right_x, y)
            y += 35
            self._draw_stat_line(f"Blink Rate: {blink_rate:.1f}/min", right_x, y)
            y += 40

            # Head movement
            head_stability = tracking_stats.get('mean_head_distance', 0)
            self._draw_stat_line(f"Head Stability: {head_stability:.1f}", right_x, y)
            y += 35

            # Looking away
            looking_away = tracking_stats.get('total_looking_away_events', 0)
            self._draw_stat_line(f"Looking Away: {looking_away} times", right_x, y)
            y += 40

            # Engagement score
            engagement = tracking_stats.get('engagement_score', 0)
            engagement_color = self._get_performance_color(engagement)
            self._draw_stat_line(f"Engagement: {engagement:.1%}", right_x, y, engagement_color)
            y += 35

            # Fatigue indicator
            fatigue = tracking_stats.get('fatigue_indicator', 0)
            fatigue_color = self._get_fatigue_color(fatigue)
            self._draw_stat_line(f"Fatigue: {fatigue:.1%}", right_x, y, fatigue_color)

        # Bottom message
        bottom_y = self.height - 80
        self._draw_stat_line("Data has been saved to results/ folder", self.center[0], bottom_y, size=28)
        self._draw_stat_line("Press any key to exit...", self.center[0], bottom_y + 35, size=28)

        pygame.display.flip()

        # Wait for any key
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                    waiting = False

    def _draw_section_header(self, text: str, x: int, y: int):
        """Draw a section header on the end screen."""
        header_font = pygame.font.Font(None, 42)
        header_surface = header_font.render(text, True, self.stim_color)
        header_rect = header_surface.get_rect(center=(x, y))
        self.screen.blit(header_surface, header_rect)

        # Underline
        line_y = y + 20
        pygame.draw.line(self.screen, self.stim_color,
                        (x - 100, line_y), (x + 100, line_y), 2)

    def _draw_stat_line(self, text: str, x: int, y: int,
                       color: Optional[Tuple[int, int, int]] = None, size: int = 30):
        """Draw a single stat line on the end screen."""
        if color is None:
            color = self.stim_color

        stat_font = pygame.font.Font(None, size)
        stat_surface = stat_font.render(text, True, color)
        stat_rect = stat_surface.get_rect(center=(x, y))
        self.screen.blit(stat_surface, stat_rect)

    def _get_performance_color(self, accuracy: float) -> Tuple[int, int, int]:
        """Get color based on performance level (green=good, yellow=ok, red=poor)."""
        if accuracy >= 0.85:
            return (100, 255, 100)  # Green
        elif accuracy >= 0.70:
            return (255, 255, 100)  # Yellow
        else:
            return (255, 100, 100)  # Red

    def _get_fatigue_color(self, fatigue: float) -> Tuple[int, int, int]:
        """Get color based on fatigue level (green=low, yellow=medium, red=high)."""
        if fatigue <= 0.3:
            return (100, 255, 100)  # Green (low fatigue is good)
        elif fatigue <= 0.6:
            return (255, 255, 100)  # Yellow
        else:
            return (255, 100, 100)  # Red (high fatigue is bad)

    def _format_duration(self, seconds: float) -> str:

        if seconds < 60:
            return f"{int(round(seconds))}s"
        else:
            minutes = int(round(seconds / 60))
            return f"{minutes}m"

    def get_random_color(self) -> Tuple[int, int, int]:
        """Generate a random bright color."""
        # Generate bright, saturated colors
        return (
            random.randint(100, 255),
            random.randint(100, 255),
            random.randint(100, 255)
        )

    def draw_stimulus(self, stimulus: str):
        """Draw stimulus on screen."""
        self.screen.fill(self.bg_color)

        # Use random color if enabled, otherwise use default
        if self.config.get("random_colors", False):
            color = self.get_random_color()
        else:
            color = self.stim_color

        text_surface = self.font.render(stimulus, True, color)
        text_rect = text_surface.get_rect(center=self.center)
        self.screen.blit(text_surface, text_rect)
        pygame.display.flip()
    
    def draw_fixation(self):
        """Draw fixation cross."""
        if self.config.get("show_fixation", True):
            self.screen.fill(self.bg_color)
            fixation = self.config.get("fixation_symbol", "+")
            text_surface = self.font.render(fixation, True, self.stim_color)
            text_rect = text_surface.get_rect(center=self.center)
            self.screen.blit(text_surface, text_rect)
            pygame.display.flip()

    def draw_blank(self):
        """Draw blank screen."""
        self.screen.fill(self.bg_color)
        pygame.display.flip()

    def run_trial(self, stimulus: str, trial_index: int) -> Tuple[int, int, Optional[float]]:
        """
        Run a single trial.

        Args:
            stimulus: The stimulus to display
            trial_index: Index of current trial

        Returns:
            Tuple of (response, correct, reaction_time_ms, trial_type, stimulus_onset)
        """
        # Start tracking for this trial
        if self.tracking_enabled and self.webcam_tracker:
            self.webcam_tracker.start_trial(trial_index)

        # Determine trial type
        trial_type = self.generator.get_trial_type(stimulus, self.previous_stimulus)
        is_target = self.generator.is_target_trial(trial_type)

        # Reset trial state
        self.response_made = False
        self.response_time = None

        # Display stimulus
        self.draw_stimulus(stimulus)
        stimulus_onset = get_timestamp()

        # Stimulus duration
        stim_duration = ms_to_seconds(self.config["stimulus_duration_ms"])
        response_window = ms_to_seconds(self.config["response_window_ms"])

        # Wait for stimulus duration + response window
        total_duration = stim_duration + response_window
        elapsed = 0

        while elapsed < total_duration:
            # Process webcam frame
            if self.tracking_enabled and self.webcam_tracker:
                frame_metrics = self.webcam_tracker.process_frame(trial_index)
                if frame_metrics and self.tracking_logger:
                    self.tracking_logger.log_frame(frame_metrics)

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.quit_game()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.quit_game()
                    elif event.key == self.response_key and not self.response_made:
                        # Record response
                        self.response_made = True
                        self.response_time = get_timestamp()

            # Clear stimulus after stimulus duration
            current_time = get_timestamp()
            elapsed = current_time - stimulus_onset

            if elapsed >= stim_duration and elapsed < stim_duration + 0.016:
                # Clear screen after stimulus duration (with small buffer for frame timing)
                self.draw_blank()

            self.clock.tick(60)  # 60 FPS

        # Calculate response and correctness
        response = 1 if self.response_made else 0

        # Correct if: (target and responded) or (not target and not responded)
        correct = 1 if (is_target and response == 1) or (not is_target and response == 0) else 0

        # Calculate reaction time
        reaction_time_ms = None
        if self.response_made:
            reaction_time_ms = calculate_elapsed_ms(stimulus_onset, self.response_time)

        return response, correct, reaction_time_ms, trial_type, stimulus_onset

    def run_session(self):
        """Run the main experimental session."""
        for i, stimulus in enumerate(self.stimulus_sequence):
            # Run trial
            response, correct, rt_ms, trial_type, onset = self.run_trial(stimulus, i)

            # Get trial-level tracking data
            tracking_data = None
            if self.tracking_enabled and self.webcam_tracker and self.tracking_logger:
                tracking_data = self.webcam_tracker.end_trial()
                self.tracking_logger.log_trial(tracking_data)

            # Log trial (with tracking data if available)
            self.logger.log_trial(
                trial_index=i,
                stimulus=stimulus,
                previous_stimulus=self.previous_stimulus,
                trial_type=trial_type,
                response=response,
                correct=correct,
                reaction_time_ms=rt_ms,
                stimulus_onset_timestamp=onset,
                tracking_data=tracking_data
            )

            # Update previous stimulus
            self.previous_stimulus = stimulus

            # Inter-stimulus interval
            self.draw_fixation()
            isi_duration = ms_to_seconds(self.config["inter_stimulus_interval_ms"])
            isi_start = get_timestamp()

            while get_timestamp() - isi_start < isi_duration:
                # Check for quit events during ISI
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.quit_game()
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        self.quit_game()

                self.clock.tick(60)

    def quit_game(self):
        """Save data, show summary, and quit."""
        # Rename folder with actual duration (even if quit early)
        if self.session_dir and len(self.logger.trials) > 0:
            self._rename_session_folder_with_duration()

        # Save data to session directory
        if self.session_dir:
            self.logger.save_to_csv(session_dir=self.session_dir)
            self._save_tracking_data()
            self.session_metadata.save_to_csv(self.session_dir)
        else:
            # Fallback if session_dir not set (shouldn't happen)
            self.logger.save_to_csv()
            self._save_tracking_data()

        # Show summary screen even when quitting early
        if len(self.logger.trials) > 0:  # Only show if at least one trial was completed
            self.show_end_screen()

        self._cleanup_tracking()
        pygame.quit()
        sys.exit()

    def _save_tracking_data(self):
        """Save all tracking data."""
        if not self.tracking_enabled or not self.tracking_logger:
            return

        if not self.session_dir:
            return

        tracking_config = self.config.get("webcam_tracking", {})

        # Save frame-level data
        if tracking_config.get("save_frame_data", True):
            self.tracking_logger.save_frame_data(self.session_dir)

        # Save session summary
        if tracking_config.get("save_session_summary", True):
            self.tracking_logger.save_session_summary(self.session_dir)

    def _cleanup_tracking(self):
        """Release webcam and cleanup tracking resources."""
        if self.webcam_tracker:
            self.webcam_tracker.release()

    def _rename_session_folder_with_duration(self):
        """
        Rename session folder to include actual duration based on trials completed.
        Calculates duration from: trials_completed × time_per_trial
        """
        if not self.session_dir:
            return

        # Calculate duration from trials completed
        trials_completed = len(self.logger.trials)
        if trials_completed == 0:
            return  # No trials, don't rename

        time_per_trial_ms = (
            self.config["stimulus_duration_ms"] +
            self.config["response_window_ms"] +
            self.config["inter_stimulus_interval_ms"]
        )
        actual_duration_seconds = (trials_completed * time_per_trial_ms) / 1000

        # Format duration
        duration_str = self._format_duration(actual_duration_seconds)

        # Create new folder name with duration
        old_dir = self.session_dir
        new_dir = f"{self.session_dir}_{duration_str}"

        # Rename the folder
        try:
            os.rename(old_dir, new_dir)
            self.session_dir = new_dir
            print(f"Session folder renamed to: {new_dir}")
        except OSError as e:
            print(f"Warning: Could not rename session folder: {e}")

    def run(self):
        """Run the complete experiment."""
        try:
            # Create session timestamp and directory
            self.session_timestamp = get_filename_timestamp()
            self.session_dir = f"results/{self.session_timestamp}"
            os.makedirs(self.session_dir, exist_ok=True)

            # Copy config file to session directory for reproducibility
            shutil.copy('config.json', f'{self.session_dir}/config.json')
            print(f"Session directory created: {self.session_dir}")

            # Show pre-test questionnaire
            questionnaire = PreTestQuestionnaire(
                self.screen,
                bg_color=tuple(self.config["background_color"]),
                text_color=tuple(self.config["stimulus_color"])
            )

            metadata_responses = questionnaire.show(self.session_timestamp)

            if metadata_responses is None:
                # User skipped questionnaire - create empty metadata
                metadata_responses = SessionMetadata.create_empty_metadata(self.session_timestamp)

            self.session_metadata.set_metadata(metadata_responses)

            # Run the test
            self.show_instructions()
            self.run_session()

            # Rename folder with actual duration
            self._rename_session_folder_with_duration()

            # Save all data (to renamed folder)
            self.logger.save_to_csv(session_dir=self.session_dir)
            self._save_tracking_data()
            self.session_metadata.save_to_csv(self.session_dir)

            # Show summary
            self.show_end_screen()
        finally:
            self._cleanup_tracking()
            pygame.quit()


def main():
    """Entry point for the program."""
    game = AXCPTGame()
    game.run()


if __name__ == "__main__":
    main()

