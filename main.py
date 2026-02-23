"""
AX-CPT (Context-Dependent Continuous Performance Test)
Main game loop and rendering
"""
import pygame
import sys
import random
from typing import Optional, Tuple
from utils import load_config, get_timestamp, ms_to_seconds, calculate_elapsed_ms
from stimulus_generator import StimulusGenerator
from logger import TrialLogger
from webcam_tracker import WebcamTracker
from tracking_logger import TrackingLogger


class AXCPTGame:
    """Main game class for AX-CPT task."""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the game."""
        self.config = load_config(config_path)
        self.logger = TrialLogger()

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

        # Generate stimulus sequence
        self.generator = StimulusGenerator(
            self.config["total_trials"],
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
        """Display end screen."""
        self.screen.fill(self.bg_color)
        
        stats = self.logger.get_summary_stats()
        
        lines = [
            "Task Complete!",
            "",
            f"Total Trials: {stats.get('total_trials', 0)}",
            f"Accuracy: {stats.get('accuracy', 0):.1%}",
            "",
            "Data has been saved.",
            "",
            "Press any key to exit..."
        ]
        
        y_offset = self.height // 3
        for line in lines:
            text_surface = self.instruction_font.render(line, True, self.stim_color)
            text_rect = text_surface.get_rect(center=(self.center[0], y_offset))
            self.screen.blit(text_surface, text_rect)
            y_offset += 45
        
        pygame.display.flip()
        
        # Wait for any key
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                    waiting = False
    
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
        """Save data and quit."""
        self.logger.save_to_csv()
        self._save_tracking_data()
        self._cleanup_tracking()
        pygame.quit()
        sys.exit()

    def _save_tracking_data(self):
        """Save all tracking data."""
        if not self.tracking_enabled or not self.tracking_logger:
            return

        tracking_config = self.config.get("webcam_tracking", {})

        # Save frame-level data
        if tracking_config.get("save_frame_data", True):
            self.tracking_logger.save_frame_data()

        # Save session summary
        if tracking_config.get("save_session_summary", True):
            self.tracking_logger.save_session_summary()

    def _cleanup_tracking(self):
        """Release webcam and cleanup tracking resources."""
        if self.webcam_tracker:
            self.webcam_tracker.release()

    def run(self):
        """Run the complete experiment."""
        try:
            self.show_instructions()
            self.run_session()
            self.logger.save_to_csv()
            self._save_tracking_data()
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

